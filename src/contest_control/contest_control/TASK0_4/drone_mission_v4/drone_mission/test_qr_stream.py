"""Offline tests: decoding and ROS adapter, no Gazebo/PX4 required."""
import sys
import types
import time
import unittest
from queue import Queue
from unittest.mock import patch

import cv2
import numpy as np
from qr_stream import RobustDecoder, CameraWorker, FrameJob, DecodeResult


class Log:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass


class Node:
    def get_logger(self): return Log()
    def create_publisher(self, *a): return types.SimpleNamespace(publish=lambda x: None)
    def create_client(self, *a): return types.SimpleNamespace(service_is_ready=lambda: False)


def module(name, **kwargs):
    m = types.ModuleType(name)
    m.__dict__.update(kwargs)
    sys.modules[name] = m


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


def qr_image(text):
    qr = cv2.QRCodeEncoder_create().encode(text)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (240, 240), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)


class Tests(unittest.TestCase):
    def setUp(self):
        self.node = Node()
        self.qr = A.QrDetector(self.node)
        self.corners = np.array([[30, 30], [100, 30], [100, 100], [30, 100]],
                                np.float32)
        self.frame = np.full((160, 160, 3), 255, np.uint8)

    def tearDown(self):
        self.qr.close()

    def inject(self, text='11', seq=1, generation=None, age=0.0, camera='down'):
        job = FrameJob(self.frame, time.monotonic() - age, (1., 2., 2.),
                       0.5, 4., self.qr.generation if generation is None else generation,
                       seq)
        decoded = [] if text is None else [(text, self.corners)]
        result = DecodeResult(camera, job, decoded, [], time.monotonic(), 0.01)
        self.qr.results.put(result)
        self.qr.poll()

    def test_real_decoding_formats_and_rotation(self):
        dec = RobustDecoder()
        for text in ('4', '11', '[4,11,6,10,9]', '4116109'):
            frame = cv2.rotate(qr_image(text), cv2.ROTATE_90_CLOCKWISE)
            items, _ = dec.decode(frame)
            self.assertIn(text, [raw for raw, _ in items])

    def test_two_distinct_frames_and_capture_pose(self):
        self.inject(seq=1)
        self.inject(seq=1)  # same frame is not confirmation
        self.assertEqual(len(self.qr.confirmed), 0)
        self.qr.update_pose((99, 99, 99), 3.)
        self.inject(seq=2)
        self.assertEqual(len(self.qr.confirmed), 1)
        self.assertEqual(self.qr.confirmed[0].drone_pos, (1., 2., 2.))
        self.assertEqual(self.qr.confirmed[0].drone_yaw, 0.5)

    def test_missing_frame_removes_servo_offset(self):
        self.inject()
        self.assertIn('11', self.qr.last_down_offsets)
        self.inject(text=None, seq=2)
        self.assertFalse(self.qr.last_down_offsets)

    def test_stale_and_previous_room_are_rejected(self):
        self.inject(age=5.)
        self.assertFalse(self.qr._candidates)
        old = self.qr.generation
        self.qr.clear_room_scan()
        self.inject(generation=old, seq=2)
        self.assertFalse(self.qr._candidates)

    def test_down_and_front_are_independent(self):
        for camera in ('down', 'front'):
            self.inject(camera=camera, seq=1)
            self.inject(camera=camera, seq=2)
        self.assertEqual(len(self.qr.get_floor_qrs()), 1)
        self.assertEqual(len(self.qr.get_doorway_qrs()), 1)

    def test_enter_room_retains_inflight_observation(self):
        self.inject(seq=1)
        self.inject(seq=2)
        fc = A.FlightController(self.node)
        fc.current_pos = (1., 2., 2.)
        fc.pose = types.SimpleNamespace()
        fc.set_target(1, 2, 2)
        ex = A.RoomExplorer(self.node, fc, self.qr)
        ms = A.MissionStateMachine(self.node, fc, self.qr, ex)
        ms.room_center = (1, 2)
        ms.set_phase(A.MissionPhase.ENTER_ROOM)
        with patch.object(fc, 'send_setpoint'):
            ms.update(0.05)
        self.assertEqual(ms.phase, A.MissionPhase.SCAN_ROOM)
        self.assertEqual(len(self.qr.get_floor_qrs()), 1)

    def test_qr_outline_pauses_room_transit(self):
        fc = A.FlightController(self.node)
        fc.current_pos = (1., 2., 2.)
        ex = A.RoomExplorer(self.node, fc, self.qr)
        ex.is_scanning = True
        ex.explorer_phase = A.ExplorerPhase.GO_VIEWPOINT
        self.qr.down_candidate_at = time.monotonic()
        self.qr.down_candidate_sequence = 5
        result = ex.update(0.05)
        self.assertEqual(result, 'qr_read_hover')
        self.assertEqual(fc.target_pos, (1., 2., 2.))

    def test_room_epoch_changes_only_once_at_door_plane(self):
        fc = A.FlightController(self.node)
        ex = A.RoomExplorer(self.node, fc, self.qr)
        ms = A.MissionStateMachine(self.node, fc, self.qr, ex)
        ms.selected_door = A.DoorwayInfo(4, (0., 0.), (-1., 0., 2.), (1.6, 0., 2.))
        ms.set_phase(A.MissionPhase.PASS_DOOR)
        fc.current_pos = (-0.5, 0., 2.)
        epoch = self.qr.generation
        with patch.object(fc, 'send_setpoint'):
            ms.update(0.05)
            self.assertEqual(self.qr.generation, epoch)
            fc.current_pos = (0.1, 0., 2.)
            ms.update(0.05)
            self.assertEqual(self.qr.generation, epoch + 1)
            ms.update(0.05)
            self.assertEqual(self.qr.generation, epoch + 1)

    def test_decoder_never_blocks_submit_and_cameras_work_together(self):
        class Slow:
            def decode(self, image):
                time.sleep(0.15)
                return [('4', np.zeros((4, 2)))], []
        output = Queue(maxsize=20)
        workers = [CameraWorker(name, output, decoder=Slow()) for name in ('down', 'front')]
        try:
            start = time.monotonic()
            for seq in range(30):
                for w in workers:
                    w.submit(FrameJob(self.frame, time.monotonic(), (0, 0, 2),
                                      0., 3., 0, seq))
            self.assertLess(time.monotonic() - start, 0.1)
            names = set()
            deadline = time.monotonic() + 2.
            while len(names) < 2 and time.monotonic() < deadline:
                names.add(output.get(timeout=1).camera)
            self.assertEqual(names, {'down', 'front'})
        finally:
            for w in workers:
                w.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
