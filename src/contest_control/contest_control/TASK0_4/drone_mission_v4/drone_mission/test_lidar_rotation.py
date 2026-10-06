"""Офлайн-тесты: карта лидара не должна «смешиваться» при развороте дрона.

Проблема из полёта: в окне 'lidar_map' карта обновлялась с каждым разворотом
дрона на месте — стены рисовались повёрнутыми, комната превращалась в «звезду»
из наслоенных сканов, A* строил путь сквозь мнимые стены и дрон не мог лететь.

Причина: LaserScan задаётся в системе координат корпуса; при интегрировании
использовался текущий yaw позы. При вращении на месте (x,y не меняются) каждый
скан добавлял точки стен под новым углом в ту же карту -> смазывание.

Решение (LidarMapper._stabilize_yaw): сканы, полученные во время вращения на
месте, доворачиваются к прежнему «стабильному» yaw; скачки yaw выше физического
предела при неподвижной позиции отбрасываются как шум pose.
"""
import math
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


# ── заглушки ROS, чтобы импортировать autonomous_drone_mission без ROS2 ──
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

import lidar_mapper as LM
import autonomous_drone_mission as A


# ─────────────────── синтетическая прямоугольная комната ───────────────────
WALLS = [(0, 0, 10, 0), (10, 0, 10, 8), (10, 8, 0, 8), (0, 8, 0, 0)]


def make_scan(x, y, yaw, n=360, range_max=30.0):
    """LaserScan прямоугольной комнаты: дальности до 4 стен из точки (x,y).

    Стены сдвинуты на +0.2 м наружу от координатной сетки — иначе лучи,
    идущие ровно вдоль стены, не дают точек отражения.
    """
    ws = [(ax - 0.2 * (bx - ax), ay - 0.2 * (by - ay),
           bx + 0.2 * (bx - ax), by + 0.2 * (by - ay))
          for ax, ay, bx, by in WALLS]
    angs = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ranges = []
    for a in angs:
        wx, wy = math.cos(a), math.sin(a)
        tmin = float('inf')
        for ax, ay, bx, by in ws:
            dx, dy = bx - ax, by - ay
            det = wx * dy - wy * dx
            if abs(det) < 1e-9:
                continue
            t = ((ax - x) * dy - (ay - y) * dx) / det      # вдоль луча
            s = ((ax - x) * wy - (ay - y) * wx) / det      # вдоль стены
            if t > 1e-6 and 0.0 <= s <= 1.0:
                tmin = min(tmin, t)
        ranges.append(min(tmin, range_max) if math.isfinite(tmin) else range_max)
    return np.asarray(ranges, np.float32), angs


def integrate_room(mapper, x, y, yaw):
    r, angs = make_scan(x, y, yaw)
    mapper.integrate(r, float(angs[0]), float(angs[1] - angs[0]),
                     0.12, 30.0, x, y, yaw)


def hit_count(mapper):
    return int((mapper.hits >= 2).sum())


class TestRotationStabilization(unittest.TestCase):
    def test_spin_in_place_does_not_smear_map(self):
        """Разворот на месте: число пикселей-стен НЕ растёт (карта чистая)."""
        good = LM.LidarMapper(size_m=20.0)
        bad = LM.LidarMapper(size_m=20.0)
        # эталон: один скан без вращения
        integrate_room(good, 5.0, 4.0, 0.0)
        base_hits = hit_count(good)
        self.assertGreater(base_hits, 50)   # стены реально нарисованы

        # старый режим (без компенсации): 8 шагов вращения на месте
        for k in range(1, 9):
            integrate_room(bad, 5.0, 4.0, k * math.radians(45))
        smeared = hit_count(bad)
        self.assertGreater(smeared, base_hits * 2)   # воспроизводим баг: карта смешана

        # новый режим: те же 8 шагов, позиция та же -> компенсация
        integrate_room(good, 5.0, 4.0, 0.0)          # тот же старт
        for k in range(1, 9):
            integrate_room(good, 5.0, 4.0, k * math.radians(45))
        self.assertEqual(hit_count(good), base_hits)  # ни лишнего пикселя стены
        self.assertGreater(good.n_rot_corrected, 0)

    def test_translation_still_integrates_normally(self):
        """Обычный полёт (движение + поворот) — сканы интегрируются штатно."""
        m = LM.LidarMapper(size_m=20.0)
        prev = 0
        for k in range(10):
            integrate_room(m, 3.0 + 0.3 * k, 4.0, math.radians(20 * k))
            h = hit_count(m)
            self.assertGreaterEqual(h, prev)
            prev = h
        self.assertEqual(m.n_rot_dropped, 0)
        self.assertGreater(m.n_scans, 5)

    def test_pose_jump_noise_is_dropped(self):
        """Скачок yaw > предела при той же позиции — скан отбрасывается."""
        m = LM.LidarMapper(size_m=20.0)
        integrate_room(m, 5.0, 4.0, 0.0)
        base = hit_count(m)
        integrate_room(m, 5.0, 4.0, math.radians(170))   # мгновенный разворот
        self.assertEqual(m.n_rot_dropped, 1)
        self.assertEqual(hit_count(m), base)             # карта не изменилась

    def test_render_raw_points_follow_actual_yaw(self):
        """Красные точки последнего скана в окне — фактическое направление взора."""
        m = LM.LidarMapper(size_m=20.0)
        integrate_room(m, 5.0, 4.0, 0.0)
        pts = m.last_scan_xy
        # при yaw=0 луч вперёд (+X) бьёт в стену x=10: есть точка около (10,4)
        d = np.linalg.norm(pts - np.array([10.0, 4.0]), axis=1).min()
        self.assertLess(d, 0.2)

    def test_reset_motion_state_on_new_room(self):
        m = LM.LidarMapper(size_m=20.0)
        integrate_room(m, 5.0, 4.0, 0.0)
        integrate_room(m, 5.0, 4.0, math.radians(30))
        self.assertIsNotNone(m._stable_yaw)
        m.reset_motion_state()
        self.assertIsNone(m._stable_yaw)
        self.assertIsNone(m.last_pose_xy)


class _FakeQr:
    """Заглушка QrDetector для RoomExplorer.on_scan."""
    def __init__(self):
        self.confirmed = []
    def room_scan_complete(self, final=False): return False
    def get_floor_qrs(self, final=False): return []
    def get_doorway_qrs(self): return []


class _FakeFc:
    def __init__(self, pos=(5.0, 4.0, 2.0), yaw=0.0):
        self.current_pos = pos
        self.current_yaw = yaw
        self.pose = True
        self.tilt = 0.0
        self.scan_z = 2.0
    def set_target(self, *a, **k): pass
    def at_yaw(self, tol=0.15): return True


class _Msg:
    pass


class TestExplorerIntegration(unittest.TestCase):
    """RoomExplorer.on_scan должен защищать карту от вращательных артефактов."""

    def _make_explorer(self):
        node = Node()
        fc = _FakeFc()
        qr = _FakeQr()
        ex = A.RoomExplorer(node, fc, qr)
        return ex, fc

    def test_no_reextract_during_pure_yaw_change(self):
        """Чистое изменение yaw (разворот) не вызывает пересборку формы комнаты."""
        ex, fc = self._make_explorer()
        ex.start_room_scan((5.0, 4.0))
        msg = types.SimpleNamespace(
            ranges=[], angle_min=0.0, angle_increment=0.0,
            range_min=0.12, range_max=30.0)
        with patch.object(type(ex.mapper), 'integrate', return_value=None):
            # первый скан фиксирует позу
            fc.current_yaw = 0.0
            ex.on_scan(msg)
            calls = []
            with patch.object(ex.mapper, 'extract_room',
                              side_effect=lambda *a: calls.append(a)):
                fc.current_yaw = math.radians(90)   # только разворот, позиция та же
                ex.on_scan(msg)
                self.assertEqual(calls, [])          # форма НЕ пересобирается
                fc.current_pos = (6.0, 4.0, 2.0)     # реальное перемещение
                ex.on_scan(msg)
                self.assertEqual(len(calls), 1)      # после движения — можно

    def test_mapper_tracks_yaw_between_scans(self):
        ex, fc = self._make_explorer()
        msg = types.SimpleNamespace(
            ranges=np.zeros(0, np.float32), angle_min=0.0,
            angle_increment=0.0, range_min=0.12, range_max=30.0)
        fc.current_yaw = 0.0
        ex.on_scan(msg)
        fc.current_yaw = math.radians(45)
        ex.on_scan(msg)
        self.assertAlmostEqual(ex.mapper.last_yaw, math.radians(45))


if __name__ == '__main__':
    unittest.main(verbosity=2)
