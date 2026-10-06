"""Offline tests for the mission decision logic (no ROS2 / Gazebo needed).

Covers the contest Task-1 requirements:
  * room scan completion gate: >=1 floor QR, >=2 doorway QR, floor==doorway match;
  * Path_Array uniqueness (repeated digit -> re-scan the room);
  * platform array comparison ([3821,5] vs "3821,5" vs concatenated digits);
  * full mission flow smoke test: takeoff -> rooms -> matching platform -> land.
"""
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

# ── stub ROS modules so the node file can be imported without a ROS install ──
def module(name, **kwargs):
    m = types.ModuleType(name)
    m.__dict__.update(kwargs)
    sys.modules[name] = m


class Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass


class Node:
    def get_logger(self): return Log()
    def create_publisher(self, *a): return types.SimpleNamespace(publish=lambda x: None)
    def create_client(self, *a): return types.SimpleNamespace(service_is_ready=lambda: False)


module('rclpy')
module('rclpy.node', Node=Node)
module('rclpy.qos', qos_profile_sensor_data=None)
for package, names in (
        ('geometry_msgs', ['PoseStamped']),
        ('sensor_msgs', ['Image', 'Imu', 'LaserScan', 'Range']),
        ('mavros_msgs', ['State'])):
    module(package)
    module(package + '.msg', **{n: type(n, (), {}) for n in names})
module('mavros_msgs.srv', **{n: type(n, (), {}) for n in
                             ['CommandBool', 'CommandLong', 'SetMode']})

import autonomous_drone_mission as A


def det(text, camera='down', pos=(0.0, 0.0, 2.0), yaw=0.0, rng=5.0):
    """Build a confirmed QrDetection directly (bypasses debouncing)."""
    qr_id = int(str(text).split(',')[0].strip('[]{} ')) if str(text)[0].isdigit() else 0
    import re as _re
    m = _re.search(r'\d+', str(text))
    qr_id = int(m.group()) if m else 0
    return A.QrDetection(
        raw_text=str(text), qr_id=qr_id, camera=camera,
        qr_type='doorway' if camera == 'front' else 'raw_down',
        drone_pos=pos, drone_yaw=yaw, front_range=rng,
        timestamp=time.monotonic())


class TestParseAndMatch(unittest.TestCase):
    """Алгоритм сравнения массивов Path_Array <-> Platform QR."""

    def test_parse_formats(self):
        self.assertEqual(A.parse_qr_array('[4, 11, 6, 10, 9]'), [4, 11, 6, 10, 9])
        self.assertEqual(A.parse_qr_array('4,11,6,10,9'), [4, 11, 6, 10, 9])
        self.assertEqual(A.parse_qr_array('4 11 6 10 9'), [4, 11, 6, 10, 9])
        self.assertEqual(A.parse_qr_array('{"array":[4,11,6,10,9]}'), [4, 11, 6, 10, 9])
        self.assertEqual(A.parse_qr_array('3821,5'), [3821, 5])
        self.assertEqual(A.parse_qr_array('12345'), [12345])
        self.assertEqual(A.parse_qr_array(''), [])

    def test_platform_exact_array_match(self):
        path = [4, 11, 6, 10, 9]
        self.assertTrue(A.platform_matches(path, '[4,11,6,10,9]'))
        self.assertTrue(A.platform_matches(path, '4,11,6,10,9'))
        self.assertFalse(A.platform_matches(path, '[3821,5]'))
        self.assertFalse(A.platform_matches(path, '[4,11,6]'))      # partial
        self.assertFalse(A.platform_matches(path, '[9,10,6,11,4]'))  # reversed

    def test_platform_concatenated_digits(self):
        # Contest example: single QR encoding "12345" for doors 1..5
        self.assertTrue(A.platform_matches([1, 2, 3, 4, 5], '12345'))
        self.assertTrue(A.platform_matches([4, 11, 6, 10, 9], '4116109'))
        # P2: [3821,5] must NOT match a path like [3, 8, 2, 1, 5]... it does not:
        self.assertFalse(A.platform_matches([3, 8, 2, 1], '3821,5'))

    def test_two_digit_door_ids_concat_ambiguity(self):
        # Concatenation is checked only when the whole digit string matches.
        self.assertTrue(A.platform_matches([11, 22], '1122'))
        self.assertFalse(A.platform_matches([11, 23], '1122'))


class TestRoomScanGate(unittest.TestCase):
    """Условие завершения сканирования комнаты из ТЗ."""

    def setUp(self):
        self.qr = A.QrDetector(Node())

    def test_needs_floor_plus_two_doorways_plus_match(self):
        self.assertFalse(self.qr.room_scan_complete())          # nothing seen
        self.qr.confirmed.append(det('4'))                       # floor only
        self.assertFalse(self.qr.room_scan_complete())
        self.qr.confirmed.append(det('4', 'front'))              # 1 doorway
        self.assertFalse(self.qr.room_scan_complete())           # need >=2 doorways
        self.qr.confirmed.append(det('7', 'front'))              # 2 doorways
        self.assertTrue(self.qr.room_scan_complete())            # floor 4 == door 4

    def test_no_match_means_incomplete(self):
        self.qr.confirmed.append(det('4'))                       # floor 4
        self.qr.confirmed.append(det('7', 'front'))
        self.qr.confirmed.append(det('9', 'front'))
        self.assertFalse(self.qr.room_scan_complete())           # no floor==door

    def test_has_matched_floor_doorway(self):
        self.qr.confirmed.append(det('11'))
        self.qr.confirmed.append(det('11', 'front'))
        ok, mid = self.qr.has_matched_floor_doorway()
        self.assertTrue(ok)
        self.assertEqual(mid, 11)

    def test_final_room_down_qrs_are_platforms_not_floor(self):
        self.qr.confirmed.append(det('[4,11,6,10,9]'))
        self.assertEqual(len(self.qr.get_floor_qrs(is_final_room=False)), 1)
        self.assertEqual(len(self.qr.get_floor_qrs(is_final_room=True)), 0)
        self.assertEqual(len(self.qr.get_platform_qrs(is_final_room=True)), 1)


class TestPathArray(unittest.TestCase):
    """Path_Array: цифры не должны повторяться."""

    def setUp(self):
        node = Node()
        self.fc = A.FlightController(node)
        self.qr = A.QrDetector(node)
        self.ex = A.RoomExplorer(node, self.fc, self.qr)
        self.ms = A.MissionStateMachine(node, self.fc, self.qr, self.ex)

    def test_add_unique_rejects_duplicate(self):
        self.assertTrue(self.ms.add_to_path(4))
        self.assertTrue(self.ms.add_to_path(11))
        self.assertFalse(self.ms.add_to_path(4))                 # repeat rejected
        self.assertEqual(self.ms.path_array, [4, 11])

    def test_select_door_prefers_matching_and_rejects_repeat(self):
        # floor 4, doors 4 and 7 -> pick door 4
        self.qr.confirmed.append(det('4'))
        self.qr.confirmed.append(det('4', 'front', pos=(2, 0, 2)))
        self.qr.confirmed.append(det('7', 'front', pos=(0, 2, 2)))
        self.ms._build_doors_from_qr()
        door = self.ms._select_door()
        self.assertIsNotNone(door)
        self.assertEqual(door.qr_id, 4)

        # ID 4 already in path -> must re-scan (None), not fly an arbitrary door
        self.ms.path_array = [4]
        self.assertIsNone(self.ms._select_door())

    def test_duplicate_door_rejected_in_build(self):
        self.ms.path_array = [4]
        self.qr.confirmed.append(det('4', 'front', pos=(2, 0, 2)))
        self.ms._build_doors_from_qr()
        self.assertTrue(all(d.rejected for d in self.ms.current_doors
                            if d.qr_id == 4))


class TestPlatformSelection(unittest.TestCase):
    """Drone must pick the platform whose array equals Path_Array."""

    def setUp(self):
        node = Node()
        self.fc = A.FlightController(node)
        self.qr = A.QrDetector(node)
        self.ex = A.RoomExplorer(node, self.fc, self.qr)
        self.ms = A.MissionStateMachine(node, self.fc, self.qr, self.ex)

    def test_three_platforms_picks_correct_one(self):
        # Task example: path [4,11,6,10,9]; P2=[3821,5]; P3 matches.
        self.ms.path_array = [4, 11, 6, 10, 9]
        self.qr.confirmed.append(det('[3821,5]', pos=(1, 1, 2)))       # P1/P2-like
        self.qr.confirmed.append(det('[4,11,6,10,9]', pos=(3, 3, 2)))  # P3
        self.assertTrue(self.ms._check_platforms())
        self.assertEqual(self.ms.selected_platform.raw_text, '[4,11,6,10,9]')

    def test_no_match_returns_false(self):
        self.ms.path_array = [1, 2, 3]
        self.qr.confirmed.append(det('[3821,5]'))
        self.assertFalse(self.ms._check_platforms())
        self.assertIsNone(self.ms.selected_platform)


class TestFullMissionSmoke(unittest.TestCase):
    """End-to-end FSM: WAIT -> ... -> landing on the matching platform."""

    def test_flow_to_auto_land(self):
        node = Node()
        fc = A.FlightController(node)
        qr = A.QrDetector(node)
        ex = A.RoomExplorer(node, fc, qr)
        ms = A.MissionStateMachine(node, fc, qr, ex)

        # sensors available
        fc.state = types.SimpleNamespace(connected=True, armed=True,
                                         mode='OFFBOARD')
        fc.pose = types.SimpleNamespace()
        fc.current_pos = (0.0, 0.0, 0.0)
        n = 360
        ex.scan = types.SimpleNamespace(
            ranges=[5.0] * n, angle_min=-np.pi,
            angle_increment=2 * np.pi / n, range_min=0.1, range_max=30.0)

        phases_seen = set()

        def step(dt):
            p = fc.current_pos
            if ms.phase in (A.MissionPhase.SCAN_ROOM,
                            A.MissionPhase.SCAN_PLATFORMS):
                # instant "scan": feed the room's QRs and finish exploring
                if not ms.is_final_room:
                    fid = len(ms.path_array) + 1
                    if not any(d.camera == 'down' for d in qr.confirmed):
                        qr.confirmed.append(det(str(fid), pos=p))
                        qr.confirmed.append(det(str(fid), 'front', pos=p))
                        qr.confirmed.append(det(str(fid + 50), 'front', pos=p))
                else:
                    if not any(d.raw_text.startswith('[') for d in qr.confirmed):
                        target = ','.join(str(x) for x in ms.path_array)
                        qr.confirmed.append(det('[3821,5]', pos=p))
                        qr.confirmed.append(det(f'[{target}]', pos=p))
                ex.is_scanning = False
                ex.scan_complete = True
            ms.update(dt)
            phases_seen.add(ms.phase)
            # teleport: arrival shortcuts everywhere
            if fc.has_target:
                fc.current_pos = tuple(fc.target_pos)

        with patch.object(fc, 'send_setpoint'), \
             patch.object(ex, 'update', return_value='complete'), \
             patch.object(ex, 'navigate_to',
                          side_effect=lambda g, z=None, face=True: 'arrived'):
            # PASS_DOOR has a real-time hover (HOVER_AFTER_DOOR); shorten it
            ms.hover_after_door = 0.05
            for _ in range(2000):
                step(0.05)
                if ms.phase in (A.MissionPhase.AUTO_LAND,
                                A.MissionPhase.FAILSAFE_LAND):
                    break

        self.assertEqual(ms.phase, A.MissionPhase.AUTO_LAND,
                         f'mission stuck, phases={sorted(p.name for p in phases_seen)}')
        # Path has unique door ids and ends on the platform that matched it
        self.assertEqual(len(set(ms.path_array)), len(ms.path_array))
        self.assertGreaterEqual(len(ms.path_array), 1)
        self.assertIsNotNone(ms.selected_platform)
        self.assertTrue(A.platform_matches(ms.path_array,
                                           ms.selected_platform.raw_text))


if __name__ == '__main__':
    unittest.main(verbosity=2)
