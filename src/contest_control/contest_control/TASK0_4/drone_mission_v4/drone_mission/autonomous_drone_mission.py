#!/usr/bin/env python3
"""
autonomous_drone_mission.py — ROS2 узел автономного полёта дрона (Gazebo + PX4 + MAVROS).

Функционал:
  1. Взлёт в центре комнаты, сканирование QR-кодов (пол + проёмы).
  2. Навигация между комнатами: сопоставление ID пола с ID проёма.
  3. Формирование Path_Array без повторов.
  4. Посадка на платформу с совпадающим массивом QR.

Архитектура (один узел, разбитый на классы):
  - QrDetector        — обработка камер, распознавание QR с дебаунсом.
  - FlightController  — управление полётом через MAVROS setpoint'ы.
  - RoomExplorer      — осмотр комнаты по лидару: стены -> углы (2 м от стен) -> точки осмотра.
  - LidarMapper (lidar_mapper.py) — карта, форма комнаты, проёмы ~2 м, A*, окно 'lidar_map'.
  - MissionStateMachine — конечный автомат всей миссии.

Топики (как в рабочем примере):
  /uav1/mavros/state, /uav1/mavros/local_position/pose,
  /uav1/scan, /uav1/rangefinder, /uav1/mavros/imu/data,
  /uav1/camera_down (вниз), /uav1/camera (вперёд).

Запуск:
  ros2 run <pkg> autonomous_drone_mission
  (или python3 autonomous_drone_mission.py после source setup.bash)

Зависимости: rclpy, mavros_msgs, geometry_msgs, sensor_msgs, cv2 (opencv-contrib-python),
             numpy, cv_bridge (опционально — используется ручное преобразование).

Предположения:
  - Координаты двери оцениваются по позе дрона + дальности лидара в момент детекта QR.
  - Высота сканирования фиксирована (scan_z = home_z + SCAN_ALT).
  - Темы камер и сенсоров соответствуют рабочему примеру пользователя.
  - Для распознавания платформ-массивов используется cv2.QRCodeDetector (OpenCV).
  - Посадка на платформу: visual servoing по нижней камере (QR в центре кадра + снижение).
"""

import math
import time
import json
import re
from queue import Queue, Empty
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Set

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, Imu, LaserScan, Range
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandLong, SetMode

# Карта лидара, форма комнаты, углы, проёмы, A* (лежит рядом с этим файлом)
from lidar_mapper import LidarMapper, Viewpoint
from qr_stream import CameraWorker, FrameJob

# ──────────────────────────────────────────────
#  Константы конфигурации
# ──────────────────────────────────────────────

NS = '/uav1'                       # namespace дрона
SCAN_ALT = 2.0                     # высота сканирования над точкой старта, м
MOVE_SPEED = 0.6                   # скорость подъёма, м/с
SP_RATE_HZ = 20                    # частота публикации setpoint, Гц
SP_TIMER_DT = 1.0 / SP_RATE_HZ    # период таймера, с

# Покрытие комнаты
GRID_RES = 0.5                    # размер ячейки сетки покрытия, м
COVERAGE_RADIUS = 4.0             # радиус сканирования от центра комнаты, м
YAW_SCAN_STEPS = 6                # (не используется в новом режиме)
YAW_DWELL = 0.6                   # время удержания на каждом шаге yaw, с
WAYPOINT_REACHED_DWELL = 0.3      # пауза после достижения waypoint, с

# QR дебаунс
QR_CONFIRM_COUNT = 1             # два разных кадра для подтверждения QR
QR_CONFIRM_WINDOW = 3.0           # окно времени для подтверждения, с
QR_RESULT_MAX_AGE = 1.5           # старые результаты не управляют полётом
QR_HOLD_TIME = 1.5                # зависание для чтения контура QR на полу
QR_HOLD_COOLDOWN = 4.0            # не зависать непрерывно над одним QR
QR_SERVO_MAX_AGE = 0.35           # предельный возраст QR для снижения

# Пороги
POS_TOL = 0.2                    # tolerance достижения точки, м
YAW_TOL = 0.15                   # tolerance достижения yaw, рад (~8.6°)
DOOR_APPROACH_DIST = 0.5         # дистанция от двери до точки approach, м
DOOR_PASS_DIST = 1.5             # дистанция пролёта через дверь, м
HOVER_AFTER_DOOR = 2.0           # зависание после пролёта двери, с
LAND_DESCEND_STEP = 0.15         # шаг снижения при visual servoing, м
LAND_DESCEND_RATE = 0.3          # скорость снижения, м/с
LAND_FINAL_ALT = 0.3             # высота перехода на AUTO.LAND, м

SHOW_WINDOWS = False              # окна камер OpenCV (False для headless)
SHOW_LIDAR_WINDOW = True         # отдельное окно карты лидара 'lidar_map'
LIDAR_WINDOW_HZ = 5.0            # частота обновления окна лидара

# Быстрый осмотр комнаты по лидару (углы — см. lidar_mapper.CORNER_OFFSET = 2 м,
# ширина проёма — lidar_mapper.DOOR_WIDTH = 2 м)
SHAPE_SCAN_TIME = 1.5            # зависание для замера стен лидаром, с
MAX_VIEWPOINTS = 8               # максимум точек осмотра на комнату
FLY_STEP = 1.0                   # «морковка»: шаг setpoint вперёд по пути, м (≈ скорость)
NAV_TIMEOUT = 30.0               # таймаут перелёта к одной точке, с
MAX_TILT_FOR_MAP = math.radians(10)  # при большем крене лидар не пишет в карту
FORCE_ARM = True                 # принудительный arm для симулятора

# Безопасность
SAFETY_MARGIN = 0.5              # минимальное расстояние до препятствия при движении, м


# ──────────────────────────────────────────────
#  Вспомогательные функции
# ──────────────────────────────────────────────

def normalize_angle(a: float) -> float:
    """Нормализация угла в [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def angle_diff(a: float, b: float) -> float:
    """Разница углов a-b, нормализованная в [-pi, pi]."""
    return normalize_angle(a - b)


def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    """Извлечение yaw из кватерниона."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Кватернион из yaw (roll=pitch=0)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (0.0, 0.0, sy, cy)


def parse_qr_array(raw_text: str) -> List[int]:
    """
    Разбор текста QR в массив целых чисел.

    Поддерживаемые форматы:
      [4, 11, 6, 10, 9]       -> [4, 11, 6, 10, 9]
      4,11,6,10,9             -> [4, 11, 6, 10, 9]
      4 11 6 10 9             -> [4, 11, 6, 10, 9]
      {"array":[4,11,6,10,9]} -> [4, 11, 6, 10, 9]
      4116109                 -> [4116109] (одно число без разделителей)
      127310                  -> [127310]  (одно число)

    ВАЖНО: строка без разделителей (4116109) неоднозначна.
    Для сравнения платформ используйте platform_matches(), которая проверяет
    и массив, и строковое представление.
    """
    text = raw_text.strip()
    if not text:
        return []

    # JSON
    if text.startswith('{') or text.startswith('['):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [int(x) for x in v]
            elif isinstance(data, list):
                return [int(x) for x in data]
        except (json.JSONDecodeError, ValueError):
            pass

    # Разделённый запятыми/пробелами
    parts = re.split(r'[,\s]+', text)
    if len(parts) > 1:
        try:
            return [int(p) for p in parts if p.strip()]
        except ValueError:
            pass

    # Одно число
    try:
        val = int(text)
        return [val]
    except ValueError:
        pass

    return []


def platform_matches(path_array: List[int], raw_text: str) -> bool:
    """
    Проверка совпадения Path_Array с QR-кодом платформы.

    Сравнивает двумя способами:
      1. Парсинг текста в массив и прямое сравнение.
      2. Строковое сравнение: если текст без разделителей,
         то конкатенация Path_Array как строк должна совпасть.

    Пример: Path_Array=[4,11,6,10,9], QR="4116109"
      Способ 1: parse("4116109") = [4116109] ≠ [4,11,6,10,9] → False
      Способ 2: "4116109" == "4116109" → True
    """
    # Способ 1: прямой массив
    arr = parse_qr_array(raw_text)
    if arr == path_array:
        return True

    # Способ 2: строковая конкатенация
    path_str = ''.join(str(x) for x in path_array)
    raw_digits = re.sub(r'\D', '', raw_text)
    if path_str == raw_digits and path_str:
        return True

    return False


def line_of_sight_clear(start_xy: Tuple[float, float],
                        end_xy: Tuple[float, float],
                        obstacle_cells: Set[Tuple[int, int]],
                        margin: float = SAFETY_MARGIN) -> bool:
    """
    Проверка отсутствия препятствий на отрезке start→end по occupancy-сетке.

    Дискретизирует отрезок и проверяет каждую точку на наличие препятствий
    в соседних ячейках.
    """
    sx, sy = start_xy
    ex, ey = end_xy
    dist = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    if dist < 0.01:
        return True

    steps = max(int(dist / (GRID_RES * 0.5)), 2)
    for i in range(steps + 1):
        t = i / steps
        px = sx + (ex - sx) * t
        py = sy + (ey - sy) * t
        gx = int(round(px / GRID_RES))
        gy = int(round(py / GRID_RES))
        # Проверяем ячейку и соседние
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if (gx + dx, gy + dy) in obstacle_cells:
                    return False
    return True


# ──────────────────────────────────────────────
#  Структуры данных
# ──────────────────────────────────────────────

@dataclass
class QrDetection:
    """Одно подтверждённое обнаружение QR-кода."""
    raw_text: str                     # оригинальный текст QR
    qr_id: int                        # первое число из текста (для floor/doorway)
    camera: str                       # 'down' или 'front'
    qr_type: str                      # 'floor', 'doorway', 'platform', 'raw_down'
    drone_pos: Tuple[float, float, float]
    drone_yaw: float
    front_range: float                # дальность лидара вперёд в момент детекта, м
    timestamp: float


@dataclass
class DoorwayInfo:
    """Информация о проёме с QR."""
    qr_id: int
    door_xy: Tuple[float, float]               # координаты двери
    approach_pos: Tuple[float, float, float]   # точка перед дверью (со стороны текущей комнаты)
    pass_pos: Tuple[float, float, float]       # точка после двери (в новой комнате)
    visited: bool = False
    rejected: bool = False                     # если ведёт к повтору ID


@dataclass
class PlatformInfo:
    """Информация о платформе."""
    name: str
    raw_text: str
    array: List[int]
    drone_pos: Tuple[float, float, float]      # позиция дрона при детекте
    pixel_offset: Tuple[float, float] = (0.0, 0.0)  # смещение QR от центра кадра


# ──────────────────────────────────────────────
#  Класс: QrDetector
# ──────────────────────────────────────────────

class QrDetector:
    """
    Обработка изображений с камер и распознавание QR-кодов с дебаунсом.

    QR считается подтверждённым, если распознан QR_CONFIRM_COUNT раз
    в течение QR_CONFIRM_WINDOW секунд.

    Ключ дебаунса — (raw_text, camera): разные QR с одним ID (например,
    floor QR "4" и platform QR "[4,11,6,10,9]") не подавляют друг друга.

    Классификация QR:
      - front камера → 'doorway'
      - down камера  → 'raw_down' (классификация floor/platform выполняется
        контекстно в MissionStateMachine, т.к. без разделителей невозможно
        отличить floor QR от platform QR)
    """

    def __init__(self, node: Node):
        self.node = node
        self.results = Queue(maxsize=12)
        self.workers = {name: CameraWorker(name, self.results)
                        for name in ('down', 'front')}
        self.generation = 0
        self.received_at = {'down': 0.0, 'front': 0.0}
        self.processed = {'down': 0, 'front': 0}
        # Время (monotonic) последнего обработанного результата по камерам —
        # для диагностики: кадры могут приходить, но обрабатываться с задержкой.
        self.processed_at = {'down': 0.0, 'front': 0.0}
        self.decode_ms = {'down': 0.0, 'front': 0.0}
        self._candidate_jobs = {}
        self._candidate_sequences = {}
        self.down_candidate_at = 0.0
        self.down_candidate_sequence = 0
        self.last_down_at = 0.0
        self.frames: Dict[str, Optional[np.ndarray]] = {'down': None, 'front': None}
        self.nframes: Dict[str, int] = {'down': 0, 'front': 0}

        # Кандидаты: { (raw_text, camera): [timestamps] }
        self._candidates: Dict[Tuple[str, str], List[float]] = {}

        # Подтверждённые обнаружения
        self.confirmed: List[QrDetection] = []

        # Текущая поза дрона
        self.drone_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.drone_yaw: float = 0.0
        self.front_range: float = 5.0  # дальность лидара вперёд

        # Последний кадр нижней камеры (для visual servoing)
        self.last_down_frame: Optional[np.ndarray] = None
        self.last_qr_offset: Tuple[float, float] = (0.0, 0.0)  # устаревший — для обратной совместимости
        self.last_qr_in_frame: bool = False
        self.last_down_offsets: Dict[str, Tuple[float, float]] = {}  # raw_text → offset

    def update_pose(self, pos: Tuple[float, float, float], yaw: float,
                    front_range: float = 5.0):
        self.drone_pos = pos
        self.drone_yaw = yaw
        self.front_range = front_range

    def on_image(self, name: str, msg: Image):
        """Быстро скопировать свежий кадр; распознавание работает независимо."""
        self.nframes[name] += 1
        now = time.monotonic()
        self.received_at[name] = now
        enc = msg.encoding.lower()
        ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc)
        if ch is None:
            self.node.get_logger().warn(f'Unsupported camera encoding: {name}={enc}',
                                        throttle_duration_sec=5.0)
            return

        try:
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        except ValueError:
            self.node.get_logger().warn(f'Invalid image buffer: {name}',
                                        throttle_duration_sec=5.0)
            return
        a = a[:, :msg.width * ch].reshape(msg.height, msg.width, ch)

        if ch == 1:
            bgr = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        elif enc == 'rgb8':
            bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        elif enc == 'rgba8':
            bgr = cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
        elif enc == 'bgra8':
            bgr = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
        else:
            bgr = np.ascontiguousarray(a)

        bgr = np.ascontiguousarray(bgr).copy()
        # Capture pose now, not later when the worker completes decoding.
        fc = getattr(self.node, 'fc', None)
        position = tuple(fc.current_pos) if fc else tuple(self.drone_pos)
        yaw = fc.current_yaw if fc else self.drone_yaw
        self.workers[name].submit(FrameJob(
            bgr, now, position, yaw, self.front_range,
            self.generation, self.nframes[name]))

    def poll(self):
        """ROS thread: apply decoded results in any phase, including flight."""
        now = time.monotonic()
        while True:
            try:
                result = self.results.get_nowait()
            except Empty:
                break
            name, job = result.camera, result.job
            self.processed[name] += 1
            self.processed_at[name] = now
            self.decode_ms[name] = result.elapsed * 1000.0
            if job.generation != self.generation or now - job.captured_at > QR_RESULT_MAX_AGE:
                continue
            if result.error:
                self.node.get_logger().warn(f'QR {name}: {result.error}',
                                            throttle_duration_sec=5.0)
            bgr = job.frame.copy()
            for corners in result.outlines:
                cv2.polylines(bgr, [corners.astype(np.int32)], True, (0, 180, 255), 2)
            if name == 'down':
                self.last_down_at = job.captured_at
                self.last_down_frame = bgr
                self.last_down_offsets.clear()
                self.last_qr_in_frame = False
                if result.outlines:
                    self.down_candidate_at = job.captured_at
                    self.down_candidate_sequence = job.sequence
            for raw, corners in result.decoded:
                self._process_qr_text(raw, name, job.captured_at, bgr, corners, job)
                if name == 'down':
                    offset = (float(corners[:, 0].mean() - bgr.shape[1] / 2),
                              float(corners[:, 1].mean() - bgr.shape[0] / 2))
                    self.last_down_offsets[raw] = offset
                    self.last_qr_offset = offset
                    self.last_qr_in_frame = True
            label = (f'{name}: RX={self.nframes[name]} decoded frames={self.processed[name]} '
                     f'{self.decode_ms[name]:.0f}ms')
            cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)
            self.frames[name] = bgr
        if now - self.last_down_at > QR_SERVO_MAX_AGE:
            self.last_down_offsets.clear()
            self.last_qr_in_frame = False

    def show_windows(self):
        if not SHOW_WINDOWS:
            return
        for name, frame in self.frames.items():
            if frame is not None:
                cv2.imshow(f'camera_{name}', frame)
        cv2.waitKey(1)

    def close(self):
        for worker in self.workers.values():
            worker.close()

    def _process_qr_text(self, text: str, camera: str, now: float,
                         bgr: np.ndarray, pts: np.ndarray, job=None):
        """Обработка одного распознанного QR-текста."""
        # Извлекаем ID (первое число)
        id_match = re.search(r'\d+', text)
        if not id_match:
            return
        qr_id = int(id_match.group())

        # Рисуем рамку
        if SHOW_WINDOWS and pts is not None:
            cv2.polylines(bgr, [pts.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(bgr, text, tuple(pts[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Дебаунс по (raw_text, camera)
        key = (text, camera)
        if job is not None and self._candidate_sequences.get(key) == job.sequence:
            return
        self._candidate_sequences[key] = job.sequence if job else None
        if key not in self._candidates:
            self._candidates[key] = []
        if not self._candidates[key] or now - self._candidates[key][-1] >= QR_CONFIRM_WINDOW:
            self._candidate_jobs[key] = job
        self._candidates[key].append(now)
        self._candidates[key] = [t for t in self._candidates[key]
                                  if now - t < QR_CONFIRM_WINDOW]

        if len(self._candidates[key]) >= QR_CONFIRM_COUNT:
            # Проверяем, не добавлен ли уже этот QR (по raw_text + camera)
            already = any(d.raw_text == text and d.camera == camera
                          for d in self.confirmed)
            if not already:
                qr_type = 'doorway' if camera == 'front' else 'raw_down'
                capture = self._candidate_jobs.get(key) or job
                det = QrDetection(
                    raw_text=text,
                    qr_id=qr_id,
                    camera=camera,
                    qr_type=qr_type,
                    drone_pos=capture.position if capture else self.drone_pos,
                    drone_yaw=capture.yaw if capture else self.drone_yaw,
                    front_range=capture.front_range if capture else self.front_range,
                    timestamp=now,
                )
                self.confirmed.append(det)
                self.node.get_logger().info(
                    f'[QR] Подтверждён: text="{text}" ID={qr_id} '
                    f'тип={qr_type} камера={camera} '
                    f'поза=({self.drone_pos[0]:.1f},{self.drone_pos[1]:.1f},'
                    f'{self.drone_pos[2]:.1f}) yaw={math.degrees(self.drone_yaw):.0f}° '
                    f'front_range={self.front_range:.1f}м')

    # ── Классификация по контексту ──

    def classify_down_qr(self, det: QrDetection, is_final_room: bool) -> str:
        """
        Контекстная классификация QR с нижней камеры.

        В финальной комнате все down-QR — платформы.
        В обычных комнатах down-QR — пол.
        """
        if is_final_room:
            return 'platform'
        return 'floor'

    def get_floor_qrs(self, is_final_room: bool = False) -> List[QrDetection]:
        """QR на полу (в не-финальной комнате)."""
        result = []
        for d in self.confirmed:
            if d.camera == 'down':
                qtype = self.classify_down_qr(d, is_final_room)
                if qtype == 'floor':
                    result.append(d)
        return result

    def get_doorway_qrs(self) -> List[QrDetection]:
        return [d for d in self.confirmed if d.qr_type == 'doorway']

    def get_platform_qrs(self, is_final_room: bool = True) -> List[QrDetection]:
        """QR на платформах (в финальной комнате)."""
        result = []
        for d in self.confirmed:
            if d.camera == 'down':
                qtype = self.classify_down_qr(d, is_final_room)
                if qtype == 'platform':
                    result.append(d)
        return result

    def get_all_down_qrs(self) -> List[QrDetection]:
        """Все QR с нижней камеры (raw_down)."""
        return [d for d in self.confirmed if d.camera == 'down']

    def has_matched_floor_doorway(self, is_final_room: bool = False) \
            -> Tuple[bool, Optional[int]]:
        """
        Проверка: есть ли QR на полу и QR над проёмом с одинаковым ID.
        """
        floor_ids = {d.qr_id for d in self.get_floor_qrs(is_final_room)}
        doorway_ids = {d.qr_id for d in self.get_doorway_qrs()}
        matched = floor_ids & doorway_ids
        if matched:
            return True, matched.pop()
        return False, None

    def room_scan_complete(self, is_final_room: bool = False) -> bool:
        """
        Условие завершения сканирования комнаты:
        - Найден хотя бы 1 QR на полу (ID_floor)
        - Найдено не менее 2 QR над проёмами (ID_doorway)
        - Найдено совпадение ID пола с ID проёма
        """
        has_floor = len(self.get_floor_qrs(is_final_room)) >= 1
        has_doorways = len(self.get_doorway_qrs()) >= 2
        matched, _ = self.has_matched_floor_doorway(is_final_room)
        return has_floor and has_doorways and matched

    def clear_room_scan(self):
        """New room epoch only at crossing, never on a room rescan."""
        self.generation += 1
        self.confirmed.clear()
        self._candidates.clear()
        self._candidate_jobs.clear()
        self._candidate_sequences.clear()
        self.last_down_offsets.clear()
        self.last_qr_in_frame = False
        self.down_candidate_at = 0.0

    def reset_all(self):
        self.clear_room_scan()


# ──────────────────────────────────────────────
#  Класс: FlightController
# ──────────────────────────────────────────────

class FlightController:
    """
    Управление полётом через MAVROS: публикация setpoint'ов,
    переключение режимов, arm/disarm.
    """

    def __init__(self, node: Node):
        self.node = node
        self.pub = node.create_publisher(
            PoseStamped, f'{NS}/mavros/setpoint_position/local', 10)
        self.cli_mode = node.create_client(SetMode, f'{NS}/mavros/set_mode')
        self.cli_arm = node.create_client(CommandBool, f'{NS}/mavros/cmd/arming')
        self.cli_cmd = node.create_client(CommandLong, f'{NS}/mavros/cmd/command')

        self.state: Optional[State] = None
        self.pose: Optional[PoseStamped] = None
        self.current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_yaw: float = 0.0
        self.tilt: float = 0.0

        self.target_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.target_yaw: float = 0.0
        self.has_target: bool = False

        # Высота сканирования (абсолютная)
        self.scan_z: float = SCAN_ALT

    def on_state(self, msg: State):
        self.state = msg

    def on_pose(self, msg: PoseStamped):
        self.pose = msg
        p = msg.pose.position
        self.current_pos = (p.x, p.y, p.z)
        o = msg.pose.orientation
        self.current_yaw = yaw_from_quaternion(o.x, o.y, o.z, o.w)
        # наклон корпуса (угол между осью Z дрона и вертикалью)
        self.tilt = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (o.x * o.x + o.y * o.y))))

    def set_target(self, x: float, y: float, z: float, yaw: Optional[float] = None):
        self.target_pos = (x, y, z)
        if yaw is not None:
            self.target_yaw = yaw
        self.has_target = True

    def set_yaw_target(self, yaw: float):
        """Изменить только yaw, сохранив позицию."""
        self.target_yaw = yaw
        self.has_target = True

    def send_setpoint(self):
        if not self.has_target:
            return
        m = PoseStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.pose.position.x = self.target_pos[0]
        m.pose.position.y = self.target_pos[1]
        m.pose.position.z = self.target_pos[2]
        _, _, sz, sw = quaternion_from_yaw(self.target_yaw)
        m.pose.orientation.z = sz
        m.pose.orientation.w = sw
        self.pub.publish(m)

    def at_target(self, tol: float = POS_TOL) -> bool:
        if not self.has_target or self.pose is None:
            return False
        dx = self.target_pos[0] - self.current_pos[0]
        dy = self.target_pos[1] - self.current_pos[1]
        dz = self.target_pos[2] - self.current_pos[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz) < tol

    def at_yaw(self, tol: float = YAW_TOL) -> bool:
        return abs(angle_diff(self.current_yaw, self.target_yaw)) < tol

    def set_mode(self, mode: str):
        if self.cli_mode.service_is_ready():
            r = SetMode.Request()
            r.custom_mode = mode
            self.cli_mode.call_async(r)
            self.node.get_logger().info(f'[FC] set_mode → {mode}')

    def arm(self):
        if FORCE_ARM and self.cli_cmd.service_is_ready():
            r = CommandLong.Request()
            r.command = 400
            r.param1 = 1.0
            r.param2 = 21196.0
            self.cli_cmd.call_async(r)
            self.node.get_logger().info('[FC] Force arm')
        elif self.cli_arm.service_is_ready():
            r = CommandBool.Request()
            r.value = True
            self.cli_arm.call_async(r)
            self.node.get_logger().info('[FC] Arm')

    def disarm(self):
        if self.cli_arm.service_is_ready():
            r = CommandBool.Request()
            r.value = False
            self.cli_arm.call_async(r)

    def land(self):
        self.set_mode('AUTO.LAND')

    def is_armed(self) -> bool:
        return self.state is not None and self.state.armed

    def is_offboard(self) -> bool:
        return self.state is not None and self.state.mode == 'OFFBOARD'

    def is_connected(self) -> bool:
        return self.state is not None and self.state.connected


# ──────────────────────────────────────────────
#  Класс: RoomExplorer
# ──────────────────────────────────────────────

class ExplorerPhase(Enum):
    """Внутренние фазы RoomExplorer."""
    IDLE = auto()
    SHAPE_SCAN = auto()       # зависание: лидар измеряет все стены комнаты
    LOOK_AROUND = auto()      # поворот камеры на проёмы из стартовой точки
    GO_VIEWPOINT = auto()     # перелёт к точке осмотра (A* по карте)
    LOOK_VIEWPOINT = auto()   # осмотр из точки (2-3 направления)
    DONE = auto()


class RoomExplorer:
    """
    Эффективный осмотр комнаты по форме, измеренной лидаром.

      1. SHAPE_SCAN (~1.5 с): лидар (360°) измеряет расстояния до всех стен,
         строится многоугольник комнаты, находятся проёмы ~2 м.
      2. LOOK_AROUND: из стартовой точки фронтальная камера поворачивается
         прямо на найденные проёмы (QR над проёмами читаются сразу).
      3. Точки осмотра: в каждом угле — точка в 2 м от ОБЕИХ стен,
         взгляд внутрь комнаты (3 направления по ±40°). Для «тени»
         Г-образной комнаты — точка у открытой стороны, взгляд в неизвестную часть.
      4. После каждой точки карта пересчитывается — новые углы второго крыла
         Г-комнаты добавляются автоматически. Посещённые точки не повторяются.
      5. Как только выполнены условия комнаты (QR пола + ≥2 QR проёмов + совпадение)
         — осмотр прекращается досрочно.
    """

    def __init__(self, node: Node, fc: FlightController, qr: QrDetector):
        self.node = node
        self.fc = fc
        self.qr = qr
        self.scan: Optional[LaserScan] = None
        self.mapper = LidarMapper(origin_xy=(0.0, 0.0))

        self.room_center: Tuple[float, float] = (0.0, 0.0)
        self.scan_z: float = SCAN_ALT
        self.explorer_phase: ExplorerPhase = ExplorerPhase.IDLE
        self.is_scanning: bool = False
        self.scan_complete: bool = False
        self.is_final_room: bool = False

        self.viewpoints: List[Viewpoint] = []     # для окна лидара (посещённые + текущая)
        self.current_vp_idx: int = -1
        self.visited: List[Tuple[float, float]] = []
        self.target_vp: Optional[Viewpoint] = None

        self.headings: List[float] = []
        self.heading_idx: int = 0
        self.dwell: float = 0.0
        self.t_phase: float = 0.0
        self.scans_at_start: int = 0
        self.hold_xy: Tuple[float, float] = (0.0, 0.0)

        # навигация
        self.nav_goal: Optional[Tuple[float, float]] = None
        self.nav_path: Optional[List[Tuple[float, float]]] = None
        self.nav_idx: int = 0
        self.nav_t0: float = 0.0
        self.qr_hold_until = 0.0
        self.qr_hold_xy = None
        self.qr_hold_yaw = 0.0
        self.qr_hold_last = -1e9
        self.qr_hold_sequence = -1

    # ── лидар ──
    def on_scan(self, msg: LaserScan):
        self.scan = msg
        if self.fc.pose is None:
            return
        # не рисуем карту на земле и при сильном крене (лидар «видит» пол)
        if self.fc.current_pos[2] < self.fc.scan_z - 0.8:
            return
        if self.fc.tilt > MAX_TILT_FOR_MAP:
            return
        self.mapper.integrate(msg.ranges, msg.angle_min, msg.angle_increment,
                              msg.range_min, msg.range_max,
                              self.fc.current_pos[0], self.fc.current_pos[1],
                              self.fc.current_yaw)

    def _get_front_range(self) -> float:
        """Минимальная дальность лидара вперёд (±15°)."""
        if self.scan is None:
            return 5.0
        r = np.asarray(self.scan.ranges, np.float32)
        ang = self.scan.angle_min + np.arange(r.size) * self.scan.angle_increment
        ok = np.isfinite(r) & (r > 0.15) & (r < self.scan.range_max)
        d = np.abs((ang + np.pi) % (2 * np.pi) - np.pi)
        sel = r[ok & (d < math.radians(15))]
        return float(sel.min()) if sel.size else self.scan.range_max

    # ── запуск ──
    def start_room_scan(self, center: Tuple[float, float], scan_z: float = None):
        self.room_center = center
        if scan_z is not None:
            self.scan_z = scan_z
        self.viewpoints = []
        self.current_vp_idx = -1
        self.visited = []
        self.target_vp = None
        self.qr_hold_until = 0.0
        self.reset_nav()
        self.is_scanning = True
        self.scan_complete = False
        self.hold_xy = (self.fc.current_pos[0], self.fc.current_pos[1])
        self.scans_at_start = self.mapper.n_scans
        self._set_phase(ExplorerPhase.SHAPE_SCAN)
        self.node.get_logger().info(
            f'[RE] Новая комната: лидар-замер стен в ({self.hold_xy[0]:.1f},'
            f'{self.hold_xy[1]:.1f})')

    def _set_phase(self, ph: ExplorerPhase):
        self.explorer_phase = ph
        self.t_phase = time.monotonic()

    def _finish(self, why: str) -> str:
        self.scan_complete = True
        self.is_scanning = False
        self._set_phase(ExplorerPhase.DONE)
        self.node.get_logger().info(f'[RE] Осмотр завершён: {why}')
        self.mapper.extract_room(self.fc.current_pos[0], self.fc.current_pos[1])
        self._log_room()   # итоговая форма (после всех точек — полная)
        return 'complete'

    # ── навигация по карте (A* + «морковка») ──
    def reset_nav(self):
        self.nav_goal = None
        self.nav_path = None
        self.nav_idx = 0

    def navigate_to(self, goal: Tuple[float, float], z: Optional[float] = None,
                    face: bool = True) -> str:
        """Возвращает 'moving' | 'arrived' | 'fail'."""
        z = self.scan_z if z is None else z
        x, y = self.fc.current_pos[0], self.fc.current_pos[1]
        if self.nav_goal is None or math.hypot(goal[0] - self.nav_goal[0],
                                               goal[1] - self.nav_goal[1]) > 0.3:
            self.mapper.extract_room(x, y)
            path = self.mapper.plan_path((x, y), goal)
            if path is None:
                self.node.get_logger().warn(
                    f'[RE] A*: путь к ({goal[0]:.1f},{goal[1]:.1f}) не найден — '
                    'лечу напрямую')
                path = [tuple(goal)]
            self.nav_goal = tuple(goal)
            self.nav_path = path
            self.nav_idx = 0
            self.nav_t0 = time.monotonic()
        if time.monotonic() - self.nav_t0 > NAV_TIMEOUT:
            self.reset_nav()
            return 'fail'
        sub = self.nav_path[self.nav_idx]
        dx, dy = sub[0] - x, sub[1] - y
        d = math.hypot(dx, dy)
        last = self.nav_idx == len(self.nav_path) - 1
        if d < (POS_TOL if last else 0.4):
            if last:
                self.fc.set_target(sub[0], sub[1], z)
                return 'arrived'
            self.nav_idx += 1
            return 'moving'
        step = min(d, FLY_STEP)
        tx, ty = x + dx / d * step, y + dy / d * step
        yaw = math.atan2(dy, dx) if (face and d > 0.6) else None
        self.fc.set_target(tx, ty, z, yaw=yaw)
        return 'moving'

    # ── осмотр направлений ──
    def _do_headings(self, dt: float) -> bool:
        """Поворачивает камеру по self.headings. True — все направления осмотрены."""
        if self.heading_idx >= len(self.headings):
            return True
        h = self.headings[self.heading_idx]
        self.fc.set_target(self.hold_xy[0], self.hold_xy[1], self.scan_z, yaw=h)
        if self.fc.at_yaw():
            self.dwell += dt
            if self.dwell >= YAW_DWELL:
                self.dwell = 0.0
                self.heading_idx += 1
        return False

    def _start_look(self, headings: List[float], phase: ExplorerPhase):
        self.hold_xy = (self.fc.current_pos[0], self.fc.current_pos[1])
        # порядок направлений — от ближайшего к текущему yaw
        cur = self.fc.current_yaw
        hs = sorted(headings, key=lambda h: (normalize_angle(h - cur)) % (2 * math.pi))
        self.headings = hs
        self.heading_idx = 0
        self.dwell = 0.0
        self._set_phase(phase)

    def _next_viewpoint(self) -> str:
        x, y = self.fc.current_pos[0], self.fc.current_pos[1]
        n_done = sum(1 for v in self.viewpoints if v.visited)
        if n_done >= MAX_VIEWPOINTS:
            return self._finish(f'лимит точек ({MAX_VIEWPOINTS})')
        plan = self.mapper.plan_viewpoints(x, y, self.visited,
                                           max_n=MAX_VIEWPOINTS - n_done)
        if not plan:
            return self._finish('все углы и тени осмотрены')
        vp = plan[0]
        self.target_vp = vp
        self.viewpoints = [v for v in self.viewpoints if v.visited] + plan
        self.current_vp_idx = len([v for v in self.viewpoints if v.visited])
        self.reset_nav()
        self._set_phase(ExplorerPhase.GO_VIEWPOINT)
        self.node.get_logger().info(
            f'[RE] Точка осмотра #{self.current_vp_idx + 1} ({vp.kind}) '
            f'({vp.x:.1f},{vp.y:.1f}); запланировано ещё {len(plan) - 1}')
        return 'moving'

    def _log_room(self):
        r = self.mapper.room
        if r is None:
            self.node.get_logger().warn('[RE] Не удалось построить форму комнаты')
            return
        self.node.get_logger().info(
            f'[RE] Комната {r.shape}: {r.size[0]:.1f}x{r.size[1]:.1f} м, '
            f'S={r.area:.1f} м², углов={len(r.corners)}, вогнутых={len(r.concave)}, '
            f'проёмов={len(self.mapper.confirmed_doors())}')
        self.node.get_logger().info(
            '[RE] До стен по осям комнаты: '
            + ' / '.join(f'{d:.2f}' for d in r.axis_dists) + ' м')
        for k, w in enumerate(r.walls):
            self.node.get_logger().info(
                f'[RE]   стена {k + 1}: {w.kind:5s} длина={w.length:.1f} м '
                f'расстояние={w.dist:.2f} м')
        for d in self.mapper.confirmed_doors():
            self.node.get_logger().info(
                f'[RE]   проём: центр=({d.center[0]:.1f},{d.center[1]:.1f}) '
                f'ширина={d.width:.2f} м')

    # ── главный update ──
    def update(self, dt: float) -> str:
        if not self.is_scanning:
            return 'idle'
        self.qr.update_pose(self.fc.current_pos, self.fc.current_yaw,
                            self._get_front_range())
        if self.qr.room_scan_complete(self.is_final_room):
            return self._finish('условия комнаты выполнены')

        ph = self.explorer_phase
        # Brief hold only inside a room, never during door crossing or landing.
        if ph == ExplorerPhase.GO_VIEWPOINT:
            now = time.monotonic()
            candidate_fresh = now - self.qr.down_candidate_at < 0.7
            if (candidate_fresh and
                    self.qr.down_candidate_sequence != self.qr_hold_sequence and
                    now - self.qr_hold_last > QR_HOLD_COOLDOWN):
                self.qr_hold_until = now + QR_HOLD_TIME
                self.qr_hold_last = now
                self.qr_hold_sequence = self.qr.down_candidate_sequence
                self.qr_hold_xy = self.fc.current_pos[:2]
                self.qr_hold_yaw = self.fc.current_yaw
                self.node.get_logger().info(
                    '[QR/down] Вижу QR-контур: короткое зависание для чтения')
            if now < self.qr_hold_until:
                self.fc.set_target(*self.qr_hold_xy, self.scan_z, yaw=self.qr_hold_yaw)
                return 'qr_read_hover'
        if ph == ExplorerPhase.SHAPE_SCAN:
            self.fc.set_target(self.hold_xy[0], self.hold_xy[1], self.scan_z)
            enough = self.mapper.n_scans - self.scans_at_start >= 8
            if time.monotonic() - self.t_phase >= SHAPE_SCAN_TIME and enough:
                self.mapper.extract_room(*self.hold_xy)
                self._log_room()
                self.visited.append(self.hold_xy)
                center_vp = Viewpoint(self.hold_xy[0], self.hold_xy[1], [], 'center',
                                      visited=True)
                self.viewpoints = [center_vp]
                hs = self.mapper.door_view_heading(*self.hold_xy)
                if not hs:   # проёмы не видны — 4 направления по осям комнаты
                    a = self.mapper.room.axis_angle if self.mapper.room else 0.0
                    hs = [a + q * math.pi / 2 for q in range(4)]
                center_vp.headings = hs
                self._start_look(hs, ExplorerPhase.LOOK_AROUND)
            return 'shape_scanning'

        if ph in (ExplorerPhase.LOOK_AROUND, ExplorerPhase.LOOK_VIEWPOINT):
            if self._do_headings(dt):
                return self._next_viewpoint()
            return 'yaw_scanning'

        if ph == ExplorerPhase.GO_VIEWPOINT:
            vp = self.target_vp
            res = self.navigate_to((vp.x, vp.y))
            if res == 'arrived':
                vp.visited = True
                self.visited.append((vp.x, vp.y))
                self._start_look(vp.headings, ExplorerPhase.LOOK_VIEWPOINT)
                return 'yaw_scanning'
            if res == 'fail':
                self.node.get_logger().warn('[RE] Точка недостижима — пропуск')
                vp.visited = True
                self.visited.append((vp.x, vp.y))
                return self._next_viewpoint()
            return 'moving'

        return self._finish('нет активной фазы')

    def remaining_path(self):
        if self.nav_path is None:
            return None
        return self.nav_path[self.nav_idx:]


# ──────────────────────────────────────────────
#  Класс: MissionStateMachine
# ──────────────────────────────────────────────

class MissionPhase(Enum):
    WAIT = auto()
    PRESTREAM = auto()
    ARMING = auto()
    TAKEOFF = auto()
    ENTER_ROOM = auto()
    SCAN_ROOM = auto()
    SELECT_DOOR = auto()
    GO_TO_DOOR = auto()
    PASS_DOOR = auto()
    VERIFY_NEW_ROOM = auto()
    SCAN_PLATFORMS = auto()
    LAND_ON_PLATFORM = auto()
    AUTO_LAND = auto()
    FAILSAFE_LAND = auto()


class MissionStateMachine:
    """
    Конечный автомат миссии.

    Полный цикл:
      WAIT → PRESTREAM → ARMING → TAKEOFF →
      ENTER_ROOM → SCAN_ROOM → SELECT_DOOR → GO_TO_DOOR →
      PASS_DOOR → VERIFY_NEW_ROOM → (ENTER_ROOM или SCAN_PLATFORMS) →
      LAND_ON_PLATFORM → AUTO_LAND

    Логика Path_Array:
      - При пролёте через дверь добавляем ID doorway QR в Path_Array.
      - Если ID уже есть в Path_Array — дверь отвергается.
      - Если все двери ведут к повтору — повторное обследование комнаты.

    Логика платформ:
      - При точном совпадении Path_Array с массивом платформы → посадка.
      - При отсутствии точного совпадения — повторное сканирование или failsafe.
      - Посадка: visual servoing по нижней камере (QR в центре кадра + снижение).
    """

    def __init__(self, node: Node, fc: FlightController,
                 qr: QrDetector, explorer: RoomExplorer):
        self.node = node
        self.fc = fc
        self.qr = qr
        self.explorer = explorer

        self.phase = MissionPhase.WAIT
        self.phase_start = time.monotonic()
        self.last_request_time = 0.0
        self.last_report_time = 0.0

        # Глобальный путь
        self.path_array: List[int] = []
        self.room_count: int = 0
        self.max_rooms: int = 10

        # Двери текущей комнаты
        self.current_doors: List[DoorwayInfo] = []
        self.selected_door: Optional[DoorwayInfo] = None
        self.room_center: Tuple[float, float] = (0.0, 0.0)

        # Платформы
        self.platforms: Dict[str, PlatformInfo] = {}
        self.selected_platform: Optional[PlatformInfo] = None
        self.is_final_room: bool = False
        self.platform_rescan_count: int = 0
        self.max_platform_rescans: int = 3

        # Домашняя позиция
        self.home: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.sp_z: float = 0.0
        self.n_prestream: int = 0

        # PASS_DOOR
        self.door_path_committed: bool = False
        self.door_pass_start: float = 0.0
        self.door_hover_start: float = 0.0
        self.room_qr_reset_at_crossing = False

        # LAND_ON_PLATFORM (visual servoing)
        self.land_descend_z: float = 0.0
        self.land_phase: str = 'approach'  # 'approach' → 'descend' → 'final_land'

    def set_phase(self, phase: MissionPhase):
        self.node.get_logger().info(
            f'[MSM] {self.phase.name} → {phase.name}')
        self.phase = phase
        self.phase_start = time.monotonic()
        if phase == MissionPhase.PASS_DOOR:
            self.room_qr_reset_at_crossing = False

    def phase_elapsed(self) -> float:
        return time.monotonic() - self.phase_start

    # ── Логика Path_Array ──

    def add_to_path(self, qr_id: int) -> bool:
        """Добавление ID в Path_Array. Возвращает True если добавлен."""
        if qr_id in self.path_array:
            self.node.get_logger().warn(
                f'[MSM] ID {qr_id} уже в Path_Array {self.path_array} — повтор!')
            return False
        self.path_array.append(qr_id)
        self.node.get_logger().info(
            f'[MSM] Path_Array обновлён: {self.path_array}')
        return True

    # ── Логика дверей ──

    def _build_doors_from_qr(self):
        """
        Список дверей из QR над проёмами.

        Сначала QR связывается с проёмом, найденным лидаром (луч камеры в момент
        детекта проходит через центр проёма ±25°). Тогда известен точный центр
        проёма и его нормаль: approach — 1.2 м до проёма, pass — 1.6 м после.
        Если лидар проём не нашёл — оценка по yaw + дальности лидара вперёд.
        """
        self.current_doors.clear()
        mapper = self.explorer.mapper
        room = mapper.room
        for d in self.qr.get_doorway_qrs():
            ld = mapper.door_along_ray(d.drone_pos[0], d.drone_pos[1], d.drone_yaw)
            if ld is not None:
                c = ld.center
                t = (ld.b - ld.a) / max(ld.width, 1e-6)
                n = np.array([-t[1], t[0]])
                ref = room.centroid if room is not None else np.array(d.drone_pos[:2])
                if np.dot(n, c - ref) < 0:
                    n = -n                         # нормаль наружу из комнаты
                door_x, door_y = float(c[0]), float(c[1])
                approach = (door_x - 1.2 * n[0], door_y - 1.2 * n[1], self.fc.scan_z)
                pass_pos = (door_x + 1.6 * n[0], door_y + 1.6 * n[1], self.fc.scan_z)
                src = f'лидар, ширина {ld.width:.2f} м'
            else:
                ux, uy = math.cos(d.drone_yaw), math.sin(d.drone_yaw)
                door_x = d.drone_pos[0] + d.front_range * ux
                door_y = d.drone_pos[1] + d.front_range * uy
                approach = (door_x - DOOR_APPROACH_DIST * ux,
                            door_y - DOOR_APPROACH_DIST * uy, self.fc.scan_z)
                pass_pos = (door_x + DOOR_PASS_DIST * ux,
                            door_y + DOOR_PASS_DIST * uy, self.fc.scan_z)
                src = 'оценка по камере'
            door = DoorwayInfo(qr_id=d.qr_id, door_xy=(door_x, door_y),
                               approach_pos=approach, pass_pos=pass_pos)
            if d.qr_id in self.path_array:
                door.rejected = True
            self.current_doors.append(door)
            self.node.get_logger().info(
                f'[MSM] Проём ID={d.qr_id}: ({door_x:.1f},{door_y:.1f}) [{src}]')

    def _select_door(self) -> Optional[DoorwayInfo]:
        """
        Выбор двери для пролёта.

        Приоритет: дверь с ID, совпадающим с floor QR.
        Если этот ID уже в Path_Array — дверь отвергается, и если других
        подходящих дверей нет, возвращается None (повторное сканирование).
        По требованию задачи: если цифра повторяется, нужно ещё раз
        обследовать комнату, а не лететь через произвольную дверь.
        """
        matched, matched_id = self.qr.has_matched_floor_doorway(self.is_final_room)

        if matched and matched_id is not None:
            # Проверяем, есть ли этот ID уже в пути
            if matched_id in self.path_array:
                # Помечаем дверь с этим ID как rejected
                for door in self.current_doors:
                    if door.qr_id == matched_id:
                        door.rejected = True
                self.node.get_logger().warn(
                    f'[MSM] Совпадающий ID {matched_id} уже в пути — '
                    'дверь rejected, требуется повторное сканирование')
                return None

            # ID не в пути — выбираем дверь с этим ID
            for door in self.current_doors:
                if door.qr_id == matched_id and not door.rejected:
                    self.node.get_logger().info(
                        f'[MSM] Выбрана дверь ID={door.qr_id} '
                        f'(совпадает с полом, координаты двери='
                        f'({door.door_xy[0]:.1f},{door.door_xy[1]:.1f}))')
                    return door

        # Если нет совпадения floor/doorway — нет подходящей двери
        # По логике задачи дрон должен найти совпадение, иначе сканировать заново
        self.node.get_logger().warn(
            '[MSM] Нет двери с совпадающим floor ID — повторное сканирование')
        return None

    # ── Логика платформ ──

    def _check_platforms(self) -> bool:
        """
        Проверка платформ: сравнение Path_Array с массивами на платформах.
        Возвращает True если найдено ТОЧНОЕ совпадение.
        """
        self.platforms.clear()

        down_qrs = self.qr.get_all_down_qrs()
        for i, d in enumerate(down_qrs):
            pname = f'P{i + 1}'
            arr = parse_qr_array(d.raw_text)
            pinfo = PlatformInfo(
                name=pname,
                raw_text=d.raw_text,
                array=arr,
                drone_pos=d.drone_pos)
            self.platforms[pname] = pinfo
            self.node.get_logger().info(
                f'[MSM] Платформа {pname}: raw="{d.raw_text}" '
                f'parsed={arr} поза=({d.drone_pos[0]:.1f},'
                f'{d.drone_pos[1]:.1f})')

        if not self.platforms:
            self.node.get_logger().warn('[MSM] Платформы не найдены')
            return False

        # Сравнение через platform_matches()
        self.node.get_logger().info(
            f'[MSM] Path_Array={self.path_array}')
        for pname, pinfo in self.platforms.items():
            match = platform_matches(self.path_array, pinfo.raw_text)
            self.node.get_logger().info(
                f'[MSM] Платформа {pname}: {pinfo.array} '
                f'raw="{pinfo.raw_text}" '
                f'{"✓ СОВПАДЕНИЕ" if match else "✗"}')
            if match:
                self.selected_platform = pinfo
                self.node.get_logger().info(
                    f'[MSM] Точное совпадение с платформой {pname}!')
                return True

        return False

    # ── Visual servoing для посадки ──

    def _visual_servo_step(self, dt: float) -> str:
        """
        Один шаг visual servoing для посадки на платформу.

        Алгоритм:
          1. Если QR выбранной платформы в кадре нижней камеры — центрироваться.
          2. Если QR отцентрирован — снижаться.
          3. На высоте LAND_FINAL_ALT — переход на AUTO.LAND.
        """
        if self.selected_platform is None:
            return 'fail'

        target = self.selected_platform.drone_pos
        raw_text = self.selected_platform.raw_text

        # Проверяем, виден ли QR выбранной платформы
        if raw_text in self.qr.last_down_offsets:
            offset_x, offset_y = self.qr.last_down_offsets[raw_text]

            # Коррекция позиции по смещению QR в кадре
            correction_gain = 0.005  # м на пиксель (приближённо)
            corr_x = -offset_x * correction_gain * math.cos(self.fc.current_yaw) \
                     + offset_y * correction_gain * math.sin(self.fc.current_yaw)
            corr_y = -offset_x * correction_gain * math.sin(self.fc.current_yaw) \
                     - offset_y * correction_gain * math.cos(self.fc.current_yaw)

            new_x = self.fc.current_pos[0] + corr_x
            new_y = self.fc.current_pos[1] + corr_y
            self.fc.set_target(new_x, new_y, self.land_descend_z)

            # Если QR отцентрирован (смещение мало) — снижаемся
            if abs(offset_x) < 30 and abs(offset_y) < 30:
                self.land_descend_z = max(
                    self.land_descend_z - LAND_DESCEND_RATE * dt,
                    self.home[2] + LAND_FINAL_ALT)
                self.fc.set_target(new_x, new_y, self.land_descend_z)
                self.node.get_logger().debug(
                    f'[MSM] Снижение: z={self.land_descend_z:.2f} '
                    f'offset=({offset_x:.0f},{offset_y:.0f})')

                if self.land_descend_z <= self.home[2] + LAND_FINAL_ALT + 0.05:
                    self.node.get_logger().info(
                        '[MSM] Достигнута финальная высота — AUTO.LAND')
                    return 'final_land'
            return 'servoing'
        else:
            # Never descend on missing/stale QR. Reacquire at the same height.
            self.land_descend_z = max(self.land_descend_z, self.fc.current_pos[2])
            self.fc.set_target(target[0], target[1], self.land_descend_z)
            return 'approach'

    # ── Главный цикл ──

    def update(self, dt: float):
        now = time.monotonic()

        # Обновление позы для QR-детектора
        front_range = self.explorer._get_front_range() if self.explorer.scan else 5.0
        self.qr.update_pose(self.fc.current_pos, self.fc.current_yaw, front_range)

        # Публикация setpoint во всех активных фазах
        if self.phase not in (MissionPhase.WAIT, MissionPhase.AUTO_LAND,
                              MissionPhase.FAILSAFE_LAND):
            self.fc.send_setpoint()

        # ── WAIT ──
        if self.phase == MissionPhase.WAIT:
            if (self.fc.is_connected() and self.fc.pose is not None
                    and self.explorer.scan is not None):
                p = self.fc.current_pos
                self.home = (p[0], p[1], p[2])
                self.room_center = (p[0], p[1])
                self.fc.scan_z = p[2] + SCAN_ALT
                self.sp_z = p[2]
                self.land_descend_z = self.fc.scan_z
                self.set_phase(MissionPhase.PRESTREAM)
            else:
                self.node.get_logger().info(
                    'Жду MAVROS, pose и лидар...', throttle_duration_sec=2.0)
            return

        # ── PRESTREAM ──
        if self.phase == MissionPhase.PRESTREAM:
            self.fc.set_target(self.home[0], self.home[1], self.home[2])
            self.n_prestream += 1
            if self.n_prestream > 40:
                self.set_phase(MissionPhase.ARMING)
            return

        # ── ARMING ──
        if self.phase == MissionPhase.ARMING:
            self.fc.set_target(self.home[0], self.home[1], self.home[2])
            if self.fc.is_offboard() and self.fc.is_armed():
                self.set_phase(MissionPhase.TAKEOFF)
            elif now - self.last_request_time > 2.0:
                self.last_request_time = now
                if not self.fc.is_offboard():
                    self.fc.set_mode('OFFBOARD')
                if not self.fc.is_armed():
                    self.fc.arm()
            return

        # ── TAKEOFF ──
        if self.phase == MissionPhase.TAKEOFF:
            target_z = self.fc.scan_z
            self.sp_z = min(self.sp_z + MOVE_SPEED * dt, target_z)
            self.fc.set_target(self.home[0], self.home[1], self.sp_z)
            if self.fc.current_pos[2] > target_z - 0.15:
                self.node.get_logger().info(
                    f'[MSM] Взлёт завершён, z={self.fc.current_pos[2]:.2f}')
                self.set_phase(MissionPhase.ENTER_ROOM)
            return

        # ── ENTER_ROOM ──
        if self.phase == MissionPhase.ENTER_ROOM:
            self.fc.set_target(
                self.room_center[0], self.room_center[1], self.fc.scan_z)
            if self.fc.at_target():
                self.room_count += 1
                self.node.get_logger().info(
                    f'[MSM] Комната #{self.room_count}, '
                    f'центр=({self.room_center[0]:.1f},'
                    f'{self.room_center[1]:.1f})')
                if self.room_count > self.max_rooms:
                    self.node.get_logger().error(
                        '[MSM] Превышено максимальное число комнат!')
                    self.set_phase(MissionPhase.FAILSAFE_LAND)
                    return
                # QR seen during takeoff/entry/flight are retained in this room.
                self.explorer.is_final_room = self.is_final_room
                self.explorer.start_room_scan(self.room_center, self.fc.scan_z)
                self.set_phase(MissionPhase.SCAN_ROOM)
            return

        # ── SCAN_ROOM ──
        if self.phase == MissionPhase.SCAN_ROOM:
            status = self.explorer.update(dt)

            # Периодический отчёт
            if now - self.last_report_time > 3.0:
                self.last_report_time = now
                floor_ids = [d.qr_id for d in
                             self.qr.get_floor_qrs(self.is_final_room)]
                door_ids = [d.qr_id for d in self.qr.get_doorway_qrs()]
                self.node.get_logger().info(
                    f'[MSM] Сканирование: {status} | '
                    f'floor_QR={floor_ids} doorway_QR={door_ids} | '
                    f'vp={self.explorer.current_vp_idx}/'
                    f'{len(self.explorer.viewpoints)} '
                    f'phase={self.explorer.explorer_phase.name}')

            if status == 'complete':
                # Проверяем, финальная ли это комната (платформы)
                down_qrs = self.qr.get_all_down_qrs()

                # Проверяем точное совпадение платформы
                exact_platform_seen = (
                    len(self.path_array) > 0 and
                    any(platform_matches(self.path_array, d.raw_text)
                        for d in down_qrs)
                )
                # Также проверяем признаки платформ: ≥2 down QR
                # или QR с разделителями (JSON/запятые)
                platform_room_seen = (
                    len(self.path_array) > 0 and (
                        exact_platform_seen
                        or len(down_qrs) >= 2
                        or any(('[' in d.raw_text or ',' in d.raw_text)
                               for d in down_qrs)
                    )
                )

                if platform_room_seen:
                    self.is_final_room = True
                    self.explorer.is_final_room = True
                    self.node.get_logger().info(
                        '[MSM] Найдена финальная комната / платформы!')
                    if self._check_platforms():
                        self.set_phase(MissionPhase.LAND_ON_PLATFORM)
                    else:
                        self.set_phase(MissionPhase.SCAN_PLATFORMS)
                else:
                    self._build_doors_from_qr()
                    self.set_phase(MissionPhase.SELECT_DOOR)
            return

        # ── SELECT_DOOR ──
        if self.phase == MissionPhase.SELECT_DOOR:
            door = self._select_door()
            if door is not None:
                self.selected_door = door
                self.door_path_committed = False
                self.set_phase(MissionPhase.GO_TO_DOOR)
            else:
                # Все двери ведут к повтору — повторное сканирование
                self.node.get_logger().info(
                    '[MSM] Повторное сканирование комнаты')
                # Keep already confirmed QR on rescan of the SAME room.
                self.explorer.is_final_room = self.is_final_room
                self.explorer.start_room_scan(self.room_center, self.fc.scan_z)
                self.set_phase(MissionPhase.SCAN_ROOM)
            return

        # ── GO_TO_DOOR ──
        if self.phase == MissionPhase.GO_TO_DOOR:
            if self.selected_door is None:
                self.set_phase(MissionPhase.SELECT_DOOR)
                return
            door = self.selected_door
            res = self.explorer.navigate_to(door.approach_pos[:2])
            if res == 'arrived':
                self.node.get_logger().info(
                    f'[MSM] Достигнут approach для двери ID={door.qr_id}')
                self.explorer.reset_nav()
                self.door_pass_start = now
                self.set_phase(MissionPhase.PASS_DOOR)
            elif res == 'fail':
                self.fc.set_target(*door.approach_pos)
                if self.fc.at_target():
                    self.explorer.reset_nav()
                    self.door_pass_start = now
                    self.set_phase(MissionPhase.PASS_DOOR)
            return

        # ── PASS_DOOR ──
        if self.phase == MissionPhase.PASS_DOOR:
            if self.selected_door is None:
                self.set_phase(MissionPhase.SELECT_DOOR)
                return
            door = self.selected_door
            # Room boundary: reset once at the door plane, before observing the
            # new room. Do not clear later in ENTER_ROOM (that loses in-flight QR).
            if not self.room_qr_reset_at_crossing:
                cx, cy = door.door_xy
                vx = door.pass_pos[0] - door.approach_pos[0]
                vy = door.pass_pos[1] - door.approach_pos[1]
                side = ((self.fc.current_pos[0] - cx) * vx +
                        (self.fc.current_pos[1] - cy) * vy)
                if side >= 0:
                    self.qr.clear_room_scan()
                    self.room_qr_reset_at_crossing = True

            # Летим через дверь по прямой, носом в проём
            yaw_pass = math.atan2(door.pass_pos[1] - door.approach_pos[1],
                                  door.pass_pos[0] - door.approach_pos[0])
            self.fc.set_target(door.pass_pos[0], door.pass_pos[1],
                               door.pass_pos[2], yaw=yaw_pass)

            # Добавляем ID в Path_Array ровно один раз
            if self.fc.at_target() and not self.door_path_committed:
                added = self.add_to_path(door.qr_id)
                self.door_path_committed = True
                self.door_hover_start = now  # отсчёт hover с момента commit
                if not added:
                    # Повтор ID — возврат
                    door.rejected = True
                    self.node.get_logger().warn(
                        '[MSM] Повтор ID при пролёте — возврат')
                    self.fc.set_target(door.approach_pos[0],
                                       door.approach_pos[1],
                                       door.approach_pos[2])
                    self.set_phase(MissionPhase.SELECT_DOOR)
                    return

                # Обновляем центр комнаты
                self.room_center = (door.pass_pos[0], door.pass_pos[1])
                self.node.get_logger().info(
                    f'[MSM] Пролёт через дверь ID={door.qr_id} завершён. '
                    f'Path_Array={self.path_array} '
                    f'Новый центр=({self.room_center[0]:.1f},'
                    f'{self.room_center[1]:.1f})')

            # Зависание после пролёта (отсчёт от момента commit)
            if self.door_path_committed and now - self.door_hover_start > HOVER_AFTER_DOOR:
                self.set_phase(MissionPhase.VERIFY_NEW_ROOM)
            return

        # ── VERIFY_NEW_ROOM ──
        if self.phase == MissionPhase.VERIFY_NEW_ROOM:
            self.fc.set_target(
                self.room_center[0], self.room_center[1], self.fc.scan_z)
            if self.fc.at_target():
                self.node.get_logger().info(
                    '[MSM] Новая комната подтверждена')
                self.set_phase(MissionPhase.ENTER_ROOM)
            return

        # ── SCAN_PLATFORMS ──
        if self.phase == MissionPhase.SCAN_PLATFORMS:
            # Перезапускаем сканирование для поиска всех платформ
            if not self.explorer.is_scanning:
                self.explorer.is_final_room = True
                self.explorer.start_room_scan(self.room_center, self.fc.scan_z)

            status = self.explorer.update(dt)

            if now - self.last_report_time > 3.0:
                self.last_report_time = now
                down_qrs = self.qr.get_all_down_qrs()
                self.node.get_logger().info(
                    f'[MSM] Сканирование платформ: {status} | '
                    f'down_QR={[(d.raw_text, d.qr_id) for d in down_qrs]}')

            # Проверяем платформы
            if self._check_platforms():
                self.set_phase(MissionPhase.LAND_ON_PLATFORM)
            elif status == 'complete':
                # Сканирование завершено, но точного совпадения нет
                self.platform_rescan_count += 1
                if self.platform_rescan_count >= self.max_platform_rescans:
                    self.node.get_logger().error(
                        f'[MSM] Достигнут лимит повторных сканирований '
                        f'платформ ({self.max_platform_rescans}). '
                        'Точного совпадения не найдено — failsafe.')
                    self.set_phase(MissionPhase.FAILSAFE_LAND)
                else:
                    self.node.get_logger().warn(
                        f'[MSM] Точного совпадения платформ не найдено. '
                        f'Повторное сканирование '
                        f'({self.platform_rescan_count}/{self.max_platform_rescans})...')
                    # Keep platform QR already read during this room's flight.
                    self.explorer.start_room_scan(
                        self.room_center, self.fc.scan_z)
            return

        # ── LAND_ON_PLATFORM ──
        if self.phase == MissionPhase.LAND_ON_PLATFORM:
            if self.selected_platform is None:
                # Нет точного совпадения — нельзя сажать
                self.node.get_logger().error(
                    '[MSM] Нет платформы с точным совпадением — failsafe')
                self.set_phase(MissionPhase.FAILSAFE_LAND)
                return

            if self.land_phase == 'approach':
                # Летим к платформе
                target = self.selected_platform.drone_pos
                res = self.explorer.navigate_to(target[:2], face=False)
                if res == 'fail':
                    self.fc.set_target(target[0], target[1], self.fc.scan_z)
                if res == 'arrived' or (res == 'fail' and self.fc.at_target()):
                    self.explorer.reset_nav()
                    self.land_phase = 'descend'
                    self.land_descend_z = self.fc.scan_z
                return

            if self.land_phase == 'descend':
                result = self._visual_servo_step(dt)
                if result == 'final_land':
                    self.land_phase = 'final_land'
                    self.fc.land()
                    self.set_phase(MissionPhase.AUTO_LAND)
                elif result == 'fail':
                    self.set_phase(MissionPhase.FAILSAFE_LAND)
                return

        # ── AUTO_LAND ──
        if self.phase == MissionPhase.AUTO_LAND:
            if now - self.last_request_time > 1.0:
                self.last_request_time = now
                if self.fc.state and self.fc.state.mode != 'AUTO.LAND':
                    self.fc.set_mode('AUTO.LAND')
                if self.fc.state and not self.fc.state.armed:
                    self.node.get_logger().info(
                        '[MSM] Посадка завершена. Миссия выполнена!')
                    self.node.get_logger().info(
                        f'[MSM] Итоговый Path_Array: {self.path_array}')
                    if self.selected_platform:
                        self.node.get_logger().info(
                            f'[MSM] Посадка на платформу '
                            f'{self.selected_platform.name}')
            return

        # ── FAILSAFE_LAND ──
        if self.phase == MissionPhase.FAILSAFE_LAND:
            self.fc.land()
            self.node.get_logger().error(
                '[MSM] Аварийная посадка активирована')
            self.set_phase(MissionPhase.AUTO_LAND)
            return


# ──────────────────────────────────────────────
#  Главный узел
# ──────────────────────────────────────────────

class DroneMissionNode(Node):
    """Главный ROS2 узел, объединяющий все компоненты."""

    def __init__(self):
        super().__init__('drone_mission')
        q = qos_profile_sensor_data

        self.fc = FlightController(self)
        self.qr = QrDetector(self)
        self.explorer = RoomExplorer(self, self.fc, self.qr)
        self.mission = MissionStateMachine(self, self.fc, self.qr, self.explorer)

        # Подписки
        self.create_subscription(State, f'{NS}/mavros/state',
                                 self.fc.on_state, q)
        self.create_subscription(PoseStamped, f'{NS}/mavros/local_position/pose',
                                 self.fc.on_pose, q)
        self.create_subscription(LaserScan, f'{NS}/scan',
                                 self.explorer.on_scan, q)
        self.create_subscription(Range, f'{NS}/rangefinder',
                                 self._on_rangefinder, q)
        self.create_subscription(Imu, f'{NS}/mavros/imu/data',
                                 self._on_imu, q)
        self.create_subscription(Image, f'{NS}/camera_down',
                                 lambda m: self.qr.on_image('down', m), q)
        self.create_subscription(Image, f'{NS}/camera',
                                 lambda m: self.qr.on_image('front', m), q)

        self.rangefinder: Optional[Range] = None
        self.imu: Optional[Imu] = None

        self.last_time = time.monotonic()
        self.create_timer(SP_TIMER_DT, self._tick)
        if SHOW_LIDAR_WINDOW:
            self.create_timer(1.0 / LIDAR_WINDOW_HZ, self._show_lidar)
        if SHOW_WINDOWS:
            self.create_timer(0.1, self._show_cameras)
        self.create_timer(2.0, self._camera_diagnostics)

        self.get_logger().info(
            'DroneMissionNode запущен. Ожидание MAVROS...')

    def _on_rangefinder(self, msg: Range):
        self.rangefinder = msg

    def _on_imu(self, msg: Imu):
        self.imu = msg

    def _tick(self):
        now = time.monotonic()
        dt = min(now - self.last_time, 0.1)  # ограничение dt для стабильности
        self.last_time = now
        self.qr.update_pose(self.fc.current_pos, self.fc.current_yaw,
                            self.explorer._get_front_range())
        self.qr.poll()
        self.mission.update(dt)

    def _show_cameras(self):
        try:
            self.qr.show_windows()
        except cv2.error as exc:
            self.get_logger().warn(f'camera windows: {exc}', throttle_duration_sec=5.0)

    def _camera_diagnostics(self):
        now = time.monotonic()
        for name in ('down', 'front'):
            worker = self.qr.workers[name]
            received = self.qr.received_at[name]
            # Возраст считается по последнему ОБРАБОТАННОМУ кадру: пока worker
            # ещё декодирует предыдущий кадр, новые кадры уже принимаются.
            processed_age = (now - self.qr.processed_at[name]
                             if self.qr.processed_at[name] else float('inf'))
            rx_age = now - received if received else float('inf')
            pending = worker.pending.qsize() + worker.inflight
            self.get_logger().info(
                f'[CAM/{name}] rx={self.qr.nframes[name]} '
                f'processed={self.qr.processed[name]} decode={self.qr.decode_ms[name]:.0f}ms '
                f'age={processed_age:.2f}s replaced={worker.dropped} pending={pending}')
            
            # Проверяем здоровье конвейера по возрaсту ОБРАБОТКИ, а не получения.
            # Если кадры приходят, но не обрабатываются — это проблема с CPU/decoder.
            # Если кадры вообще не приходят — проблема с topic/bridge/QoS.
            if rx_age > 3.0 and processed_age > 3.0:
                # Длительное отсутствие кадров на входе И в обработке.
                self.get_logger().warn(
                    f'[CAM/{name}] Нет свежих кадров: проверьте camera topic/bridge/QoS')
            elif processed_age > max(2.5, 4.0 * self.qr.decode_ms[name] / 1000.0):
                # Кадры приходят регулярно, но конвейер не успевает их обрабатывать.
                self.get_logger().warn(
                    f'[CAM/{name}] Конвейер QR отстаёт: decode='
                    f'{self.qr.decode_ms[name]:.0f}ms при rx_age={rx_age:.2f}s; '
                    f'проверьте загрузку CPU / уменьшите разрешение камеры')

    def _show_lidar(self):
        """Отдельное окно: карта лидара, стены, углы, проёмы, точки осмотра."""
        if self.fc.pose is None:
            return
        try:
            ex, ms = self.explorer, self.mission
            floor = [d.qr_id for d in self.qr.get_floor_qrs(ms.is_final_room)]
            doors = [d.qr_id for d in self.qr.get_doorway_qrs()]
            info = [
                '',
                f'Миссия: {ms.phase.name}',
                f'Осмотр: {ex.explorer_phase.name}',
                f'Высота: {self.fc.current_pos[2]:.2f} м',
                f'QR пол: {floor}',
                f'QR проёмы: {doors}',
                f'Path_Array: {ms.path_array}',
            ]
            img = ex.mapper.render(self.fc.current_pos[0], self.fc.current_pos[1],
                                   self.fc.current_yaw, ex.viewpoints,
                                   ex.current_vp_idx, ex.remaining_path(), info)
            cv2.imshow('lidar_map', img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'lidar window: {e}', throttle_duration_sec=5.0)

    def shutdown(self):
        self.fc.land()
        self.qr.close()
        if SHOW_WINDOWS or SHOW_LIDAR_WINDOW:
            cv2.destroyAllWindows()
        self.destroy_node()


def main():
    rclpy.init()
    node = DroneMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # При Ctrl+C контекст может быть уже невалидным для публикации логов.
        # Используем print вместо logger, чтобы избежать ошибки.
        print('[drone_mission] Прерывание пользователем (Ctrl+C)')
        try:
            node.fc.land()
        except Exception as e:
            print(f'[drone_mission] Ошибка при посадке: {e}')
        # Даем немного времени на обработку последних setpoint'ов
        try:
            rclpy.spin_once(node, timeout_sec=0.3)
        except Exception:
            pass
    finally:
        try:
            node.shutdown()
        except Exception as e:
            print(f'[drone_mission] Ошибка при shutdown: {e}')
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
