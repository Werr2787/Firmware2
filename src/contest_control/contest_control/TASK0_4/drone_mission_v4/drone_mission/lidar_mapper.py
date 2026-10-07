#!/usr/bin/env python3
"""
lidar_mapper.py — 2D-карта по лидару: форма комнаты, стены, углы, проёмы,
точки осмотра и планирование пути. Чистый numpy + OpenCV (без ROS),
поэтому модуль можно проверять офлайн.

Логика осмотра комнаты:
  1. Из 360° лидара строится карта (свободно / стена).
  2. Находятся проёмы шириной ~2 м (разрыв в стене, за которым лучи уходят дальше).
     Проём «закрывается» виртуальной стенкой, чтобы комната не «протекала» в соседнюю.
  3. Контур комнаты -> многоугольник -> список стен с длиной и расстоянием до дрона.
  4. Точки осмотра:
       - УГОЛ: выпуклый угол между двумя настоящими стенами -> точка на расстоянии
         CORNER_OFFSET (2 м) от ОБЕИХ стен (по биссектрисе: s = d / sin(θ/2)).
       - FRONTIER: «открытая» сторона многоугольника (тень за выступом Г-образной
         комнаты или предел дальности лидара) -> точка в 1 м от неё внутрь,
         взгляд в неизвестную часть. После прилёта карта обновляется,
         и появляются новые углы (второе крыло Г-комнаты).
  5. Путь между точками — A* по безопасной зоне (>= SAFETY_MARGIN от стен).
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────── параметры ────────────────────────────
RES = 0.1                 # м/пиксель карты
MAP_SIZE_M = 60.0         # размер карты, м (квадрат с центром в origin)
MAX_FREE_RANGE = 10.0     # луч без отражения считаем свободным до этой дальности, м

DOOR_WIDTH = 2.0          # ожидаемая ширина проёма, м
DOOR_MIN = 1.3            # допуск ширины проёма, м
DOOR_MAX = 2.8
DOOR_JUMP = 0.8           # скачок дальности на краю проёма, м
DOOR_CONFIRM = 3          # сколько сканов должно увидеть проём

CORNER_OFFSET = 2.0       # точка осмотра угла: 2 м от каждой из двух стен
FRONTIER_OFFSET = 1.0     # точка осмотра «тени»: 1 м внутрь от открытой стороны
SAFETY_MARGIN = 0.6       # минимум до стены для точек и пути, м
MERGE_DIST = 1.2          # точки ближе этого объединяются, м
MIN_OPEN_EDGE = 0.8       # открытая сторона короче — игнорируется, м

# ── компенсация поворота дрона (защита от «смешивания» карты) ──
# При развороте на месте позиция в ENU не меняется, а LaserScan приходит с
# новым yaw. Если интегрировать такой скан как есть, лучи рисуются там, где
# их «увидел» поворот, и карта смазывается при каждом развороте. Поэтому
# сканы, полученные во время вращения (yaw изменился больше YAW_ROT_THRESH
# между соседними сканами), доворачиваются к прежнему ориентированию дрона —
# геометрия комнаты остаётся неподвижной в мировых координатах.
YAW_ROT_THRESH = math.radians(10.0)   # считаем дрон «вращающимся»
YAW_SLEW_LIMIT = math.radians(90.0)   # физический предел поворота за скан;
                                      # скачок больше — это шум/потеря pose,
                                      # скан отбрасывается целиком


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


@dataclass
class Door:
    a: np.ndarray
    b: np.ndarray
    hits: int = 1

    @property
    def center(self) -> np.ndarray:
        return (self.a + self.b) / 2.0

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.a - self.b))

    @property
    def confirmed(self) -> bool:
        return self.hits >= DOOR_CONFIRM


@dataclass
class Wall:
    p1: np.ndarray
    p2: np.ndarray
    kind: str            # 'wall' | 'door' | 'open'
    length: float
    dist: float          # расстояние от дрона до стены, м


@dataclass
class Viewpoint:
    x: float
    y: float
    headings: List[float]
    kind: str            # 'center' | 'corner' | 'frontier' | 'door'
    visited: bool = False


@dataclass
class RoomModel:
    polygon: np.ndarray                  # Nx2 мировые координаты
    walls: List[Wall]
    corners: List[np.ndarray]            # выпуклые углы между стенами
    concave: List[np.ndarray]            # вогнутые (внутренний угол Г-комнаты)
    centroid: np.ndarray
    area: float
    size: Tuple[float, float]            # стороны minAreaRect, м
    axis_angle: float                    # ориентация комнаты, рад
    axis_dists: List[float]              # до стен по 4 осям комнаты, м
    shape: str                           # 'прямоугольная' | 'Г-образная' | ...
    mask: np.ndarray = field(repr=False)     # uint8 маска комнаты (вся карта)
    dist: np.ndarray = field(repr=False)     # distance transform, м


class LidarMapper:
    def __init__(self, origin_xy=(0.0, 0.0), size_m=MAP_SIZE_M, res=RES):
        self.res = res
        self.n = int(size_m / res)
        self.ox = origin_xy[0] - size_m / 2.0
        self.oy = origin_xy[1] - size_m / 2.0
        self.hits = np.zeros((self.n, self.n), np.uint16)
        self.free = np.zeros((self.n, self.n), np.uint16)
        self.doors: List[Door] = []
        self.room: Optional[RoomModel] = None
        self.last_scan_xy: Optional[np.ndarray] = None
        self.n_scans = 0
        # ── компенсация вращения дрона (карта не должна «смешиваться»
        #    при разворотах) ──
        self.last_pose_xy: Optional[np.ndarray] = None
        self.last_yaw: Optional[float] = None
        self._stable_yaw: Optional[float] = None   # yaw после последнего
                                                   # перемещения дрона
        self.n_rot_corrected = 0     # сколько сканов довернуто
        self.n_rot_dropped = 0       # сколько сканов отброшено (скачок yaw)
        self.n_skipped = 0           # всего сканов, не попавших в карту

    # ───────────── координаты ─────────────
    def w2p(self, x, y) -> Tuple[int, int]:
        return int((x - self.ox) / self.res), int((y - self.oy) / self.res)

    def p2w(self, ix, iy) -> Tuple[float, float]:
        return self.ox + (ix + 0.5) * self.res, self.oy + (iy + 0.5) * self.res

    def _inside(self, ix, iy) -> bool:
        return 0 <= ix < self.n and 0 <= iy < self.n

    # ───────────── интеграция скана ─────────────
    def _stabilize_yaw(self, x, y, yaw):
        """Компенсация поворота дрона на месте.

        Лидар в Gazebo «вращает» лучи вместе с корпусом: если дрон разворачивается,
        не меняя позиции (x, y), следующий LaserScan рисует те же стены в других
        мировых точках — карта смазывается и A* начинает водить дрона врезаясь.

        Правило:
          * дрон переместился (>= MOVE_EPS)   -> yaw обновляется штатно;
          * позиция та же, |Δyaw| <= YAW_SLEW_LIMIT -> считаем это вращением:
            скан доворачиваем к прежнему стабильному yaw (геометрия комнаты
            остаётся неподвижной);
          * скачок yaw больше физического предела при той же позиции -> шум pose,
            скан отбрасываем целиком.

        Возвращает (use: bool, yaw_eff: float).
        """
        MOVE_EPS = 0.15   # м — меньше этого считаем, что дрон «на месте»
        if self.last_pose_xy is None or self._stable_yaw is None:
            use, ye = True, yaw
        else:
            moved = float(np.linalg.norm(np.array([x, y]) - self.last_pose_xy))
            dyaw = abs(math.atan2(math.sin(yaw - self.last_yaw),
                                  math.cos(yaw - self.last_yaw)))
            if moved >= MOVE_EPS:
                use, ye = True, yaw                 # обычное движение
            elif dyaw > YAW_SLEW_LIMIT:
                use, ye = False, self._stable_yaw   # скачок — мусор, режем
                self.n_rot_dropped += 1
            elif dyaw > YAW_ROT_THRESH:
                use, ye = True, self._stable_yaw    # вращение на месте —
                self.n_rot_corrected += 1           # доворачиваем к старому yaw
            else:
                use, ye = True, yaw                 # стоит почти прямо
        self.last_pose_xy = np.array([x, y])
        self.last_yaw = yaw
        if use:
            self._stable_yaw = ye
        return use, ye

    def reset_motion_state(self):
        """Сброс трекера вращения (например, при старте новой комнаты)."""
        self.last_pose_xy = None
        self.last_yaw = None
        self._stable_yaw = None

    def integrate(self, ranges, angle_min, angle_inc, range_min, range_max,
                  x, y, yaw):
        r = np.asarray(ranges, np.float32)
        if r.size < 10:
            return
        use, yaw_eff = self._stabilize_yaw(x, y, yaw)
        if not use:
            self.n_skipped += 1
            return
        ang = angle_min + np.arange(r.size, dtype=np.float32) * angle_inc + yaw_eff
        rmax = min(float(range_max), MAX_FREE_RANGE)
        valid = np.isfinite(r) & (r > range_min) & (r < range_max * 0.98)
        far = (np.isinf(r) | (np.isfinite(r) & (r >= range_max * 0.98)))
        # дальность для области «свободно»
        rf = np.where(valid, np.maximum(r - 1.5 * self.res, 0.0),
                      np.where(far, rmax, float(range_min)))
        rf = np.minimum(rf, rmax)
        fx = x + rf * np.cos(ang)
        fy = y + rf * np.sin(ang)
        poly = np.stack([(fx - self.ox) / self.res, (fy - self.oy) / self.res],
                        axis=1).astype(np.int32)
        mask = np.zeros((self.n, self.n), np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        np.add(self.free, mask, out=self.free, where=self.free < 60000)

        # стены — концы лучей
        hx = x + r[valid] * np.cos(ang[valid])
        hy = y + r[valid] * np.sin(ang[valid])
        ix = ((hx - self.ox) / self.res).astype(np.int32)
        iy = ((hy - self.oy) / self.res).astype(np.int32)
        ok = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        np.add.at(self.hits, (iy[ok], ix[ok]), 1)
        # «Сырые» точки последнего скана тоже доворачиваем к yaw_eff, иначе в
        # окне lidar_map красные точки прыгают по кругу при каждом развороте.
        raw_ang = angle_min + np.arange(r.size, dtype=np.float32) * angle_inc + yaw
        hxr = x + r[valid] * np.cos(raw_ang[valid])
        hyr = y + r[valid] * np.sin(raw_ang[valid])
        self.last_scan_xy = np.stack([hxr, hyr], axis=1)

        # проёмы
        rd = np.where(valid, r, np.where(far, float(range_max), np.nan))
        self._detect_doors(rd, ang, valid, x, y)
        self.n_scans += 1

    def _detect_doors(self, rd, ang, valid, x, y):
        """Проём = скачок «близко -> далеко -> близко», края на расстоянии ~2 м."""
        n = rd.size
        r2 = np.concatenate([rd, rd])
        a2 = np.concatenate([ang, ang])
        v2 = np.concatenate([valid, valid])
        found = []
        i = 0
        max_span = n // 3
        while i < n:
            if not (v2[i] and np.isfinite(r2[i]) and np.isfinite(r2[i + 1])):
                i += 1
                continue
            if r2[i + 1] - r2[i] > DOOR_JUMP:            # близко -> далеко
                near_i = r2[i]
                j = i + 1
                while j < i + max_span and not (
                        np.isfinite(r2[j + 1]) and np.isfinite(r2[j])
                        and r2[j] - r2[j + 1] > DOOR_JUMP):
                    j += 1
                if j < i + max_span and v2[j + 1]:
                    near_j = r2[j + 1]
                    gap = r2[i + 1:j + 1]
                    gap = gap[np.isfinite(gap)]
                    if gap.size and gap.min() > max(near_i, near_j) + 0.3:
                        A = np.array([x + near_i * math.cos(a2[i]),
                                      y + near_i * math.sin(a2[i])])
                        B = np.array([x + near_j * math.cos(a2[j + 1]),
                                      y + near_j * math.sin(a2[j + 1])])
                        w = float(np.linalg.norm(A - B))
                        if DOOR_MIN <= w <= DOOR_MAX:
                            found.append((A, B))
                    i = j + 1
                    continue
            i += 1
        for A, B in found:
            c = (A + B) / 2.0
            for d in self.doors:
                if np.linalg.norm(d.center - c) < 0.7:
                    k = min(d.hits, 20)
                    # совмещаем порядок концов
                    if np.linalg.norm(d.a - A) > np.linalg.norm(d.a - B):
                        A, B = B, A
                    d.a = (d.a * k + A) / (k + 1)
                    d.b = (d.b * k + B) / (k + 1)
                    d.hits += 1
                    break
            else:
                self.doors.append(Door(A, B))

    def confirmed_doors(self) -> List[Door]:
        return [d for d in self.doors if d.confirmed]

    # ───────────── форма комнаты ─────────────
    def _occ(self):
        occ = ((self.hits >= 2).astype(np.uint8)) * 255
        return cv2.dilate(occ, np.ones((3, 3), np.uint8))

    def extract_room(self, x, y) -> Optional[RoomModel]:
        occ_d = self._occ()
        walls_img = occ_d.copy()
        for d in self.confirmed_doors():
            cv2.line(walls_img, self.w2p(*d.a), self.w2p(*d.b), 255, 3)
        free = ((self.free >= 2) & (walls_img == 0)).astype(np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, lab = cv2.connectedComponents(free, connectivity=4)
        ix, iy = self.w2p(x, y)
        if not self._inside(ix, iy):
            return None
        label = lab[iy, ix]
        if label == 0:
            win = lab[max(iy - 7, 0):iy + 8, max(ix - 7, 0):ix + 8]
            vals = win[win > 0]
            if vals.size == 0:
                return None
            label = np.bincount(vals).argmax()
        room = (lab == label).astype(np.uint8)
        cnts, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        filled = np.zeros_like(room)
        cv2.drawContours(filled, [c], -1, 1, -1)
        area = cv2.contourArea(c) * self.res ** 2
        if area < 1.0:
            return None

        approx = cv2.approxPolyDP(c, 0.3 / self.res, True).reshape(-1, 2)
        poly = np.array([self.p2w(px, py) for px, py in approx], np.float64)
        dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5) * self.res

        # классификация сторон
        wall_chk = cv2.dilate(occ_d, np.ones((5, 5), np.uint8))
        doors = self.confirmed_doors()
        me = np.array([x, y])
        walls: List[Wall] = []
        m = len(poly)
        for k in range(m):
            p1, p2 = poly[k], poly[(k + 1) % m]
            L = float(np.linalg.norm(p2 - p1))
            ts = np.linspace(0.0, 1.0, max(int(L / self.res), 2))
            pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
            hit = 0
            for px, py in pts:
                jx, jy = self.w2p(px, py)
                if self._inside(jx, jy) and wall_chk[jy, jx]:
                    hit += 1
            frac = hit / len(pts)
            mid = (p1 + p2) / 2.0
            if any(np.linalg.norm(d.center - mid) < 1.0 and abs(L - d.width) < 1.0
                   for d in doors):
                kind = 'door'
            elif frac >= 0.6:
                kind = 'wall'
            else:
                kind = 'open'
            walls.append(Wall(p1, p2, kind, L, _seg_dist(me, p1, p2)))

        # углы
        s_area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                    - np.roll(poly[:, 0], -1) * poly[:, 1]))
        corners, concave = [], []
        for k in range(m):
            v, pv, nx = poly[k], poly[k - 1], poly[(k + 1) % m]
            a_, b_ = v - pv, nx - v
            z = float(a_[0] * b_[1] - a_[1] * b_[0])
            w_prev, w_next = walls[k - 1], walls[k]
            if w_prev.kind == 'open' and w_next.kind == 'open':
                continue
            e1, e2 = _unit(pv - v), _unit(nx - v)
            th = math.acos(float(np.clip(np.dot(e1, e2), -1.0, 1.0)))
            if z * s_area > 0:
                if (w_prev.kind == 'wall' and w_next.kind == 'wall'
                        and math.radians(50) < th < math.radians(130)):
                    corners.append(v)
            elif math.radians(50) < th < math.radians(130):
                concave.append(v)

        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        size = (rw * self.res, rh * self.res)
        axis = math.radians(rect[2])
        fill = area / max(size[0] * size[1], 1e-6)
        if len(concave) >= 1 and fill < 0.85:
            shape = 'Г-образная' if len(concave) == 1 else 'сложная'
        else:
            shape = 'прямоугольная'
        axis_d = [self.ray_to_wall(x, y, axis + q * math.pi / 2, walls_img)
                  for q in range(4)]
        M = cv2.moments(c)
        cen = np.array(self.p2w(M['m10'] / M['m00'], M['m01'] / M['m00']))
        self.room = RoomModel(poly, walls, corners, concave, cen, area, size,
                              axis, axis_d, shape, filled, dist)
        return self.room

    def ray_to_wall(self, x, y, ang, walls_img=None, max_d=20.0) -> float:
        img = self._occ() if walls_img is None else walls_img
        d = 0.0
        while d < max_d:
            ix, iy = self.w2p(x + d * math.cos(ang), y + d * math.sin(ang))
            if not self._inside(ix, iy) or img[iy, ix]:
                return d
            d += self.res
        return max_d

    # ───────────── точки осмотра ─────────────
    def _valid(self, p, room: RoomModel) -> bool:
        ix, iy = self.w2p(*p)
        return self._inside(ix, iy) and room.dist[iy, ix] >= SAFETY_MARGIN

    def _fit_point(self, p, room: RoomModel) -> Optional[np.ndarray]:
        """Сдвинуть точку к центру комнаты, пока она не станет безопасной."""
        if self._valid(p, room):
            return p
        d = _unit(room.centroid - p)
        for s in np.arange(0.2, 3.01, 0.2):
            q = p + d * s
            if self._valid(q, room):
                return q
        return None

    def plan_viewpoints(self, x, y, visited: List[Tuple[float, float]],
                        max_n: int = 8) -> List[Viewpoint]:
        room = self.extract_room(x, y)
        if room is None:
            return []
        cand: List[Viewpoint] = []

        # 1) углы: 2 м от обеих стен
        poly = room.polygon
        for v in room.corners:
            k = int(np.argmin(np.linalg.norm(poly - v, axis=1)))
            e1 = _unit(poly[k - 1] - v)
            e2 = _unit(poly[(k + 1) % len(poly)] - v)
            th = math.acos(float(np.clip(np.dot(e1, e2), -1, 1)))
            b = _unit(e1 + e2)
            p = None
            for d in (CORNER_OFFSET, 1.6, 1.2, 0.9):
                q = v + b * (d / math.sin(th / 2.0))
                if self._valid(q, room):
                    p = q
                    break
            if p is None:
                p = self._fit_point(v + b * (CORNER_OFFSET / math.sin(th / 2.0)), room)
            if p is None:
                continue
            h = math.atan2(b[1], b[0])          # взгляд из угла внутрь комнаты
            cand.append(Viewpoint(p[0], p[1],
                                  [h - math.radians(40), h, h + math.radians(40)],
                                  'corner'))

        # 2) открытые стороны (тень Г-комнаты / предел лидара)
        for w in room.walls:
            if w.kind != 'open' or w.length < MIN_OPEN_EDGE:
                continue
            mid = (w.p1 + w.p2) / 2.0
            t = _unit(w.p2 - w.p1)
            n_in = np.array([-t[1], t[0]])
            jx, jy = self.w2p(*(mid + 0.3 * n_in))
            if not (self._inside(jx, jy) and room.mask[jy, jx]):
                n_in = -n_in
            p = self._fit_point(mid + FRONTIER_OFFSET * n_in, room)
            if p is None:
                continue
            h = math.atan2(-n_in[1], -n_in[0])  # смотреть в неизвестную часть
            cand.append(Viewpoint(p[0], p[1], [h - math.radians(35), h,
                                               h + math.radians(35)], 'frontier'))

        # убрать повторы и уже посещённые
        out: List[Viewpoint] = []
        for c in cand:
            if any(math.hypot(c.x - vx, c.y - vy) < MERGE_DIST for vx, vy in visited):
                continue
            if any(math.hypot(c.x - o.x, c.y - o.y) < MERGE_DIST for o in out):
                continue
            out.append(c)

        # порядок облёта: ближайший сосед (углы и тени вперемешку)
        order, cur = [], np.array([x, y])
        rest = out[:]
        while rest and len(order) < max_n:
            k = int(np.argmin([math.hypot(v.x - cur[0], v.y - cur[1]) for v in rest]))
            order.append(rest.pop(k))
            cur = np.array([order[-1].x, order[-1].y])
        return order

    def door_view_heading(self, x, y) -> List[float]:
        """Направления на проёмы, видимые из точки (для фронтальной камеры)."""
        hs = []
        for d in self.confirmed_doors():
            c = d.center
            hs.append(math.atan2(c[1] - y, c[0] - x))
        hs.sort()
        return hs

    def door_along_ray(self, x, y, yaw, max_ang=math.radians(25), max_d=8.0):
        best, best_a = None, max_ang
        for d in self.confirmed_doors():
            c = d.center
            dist = math.hypot(c[0] - x, c[1] - y)
            if dist > max_d:
                continue
            a = abs(math.atan2(math.sin(math.atan2(c[1] - y, c[0] - x) - yaw),
                               math.cos(math.atan2(c[1] - y, c[0] - x) - yaw)))
            if a < best_a:
                best, best_a = d, a
        return best

    # ───────────── путь (A*) ─────────────
    def plan_path(self, start, goal, margin=SAFETY_MARGIN):
        room = self.room
        if room is None:
            return [tuple(goal)]
        ds = 2  # 0.2 м
        trav = room.dist[::ds, ::ds] >= margin
        H, W = trav.shape
        sx, sy = self.w2p(*start)
        gx, gy = self.w2p(*goal)
        s, g = (sy // ds, sx // ds), (gy // ds, gx // ds)
        if not (0 <= g[0] < H and 0 <= g[1] < W) or not (0 <= s[0] < H and 0 <= s[1] < W):
            return None
        if not trav[g]:
            return None
        trav = trav.copy()
        trav[s] = True
        nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
              (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        openh = [(0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        while openh:
            _, cur = heapq.heappop(openh)
            if cur == g:
                break
            for dy, dx, w in nb:
                nxt = (cur[0] + dy, cur[1] + dx)
                if not (0 <= nxt[0] < H and 0 <= nxt[1] < W) or not trav[nxt]:
                    continue
                nc = cost[cur] + w
                if nc < cost.get(nxt, 1e18):
                    cost[nxt] = nc
                    came[nxt] = cur
                    hh = math.hypot(nxt[0] - g[0], nxt[1] - g[1])
                    heapq.heappush(openh, (nc + hh, nxt))
        if g not in came:
            return None
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = came[c]
        cells.reverse()

        def los(a, b):
            n = int(max(abs(a[0] - b[0]), abs(a[1] - b[1]))) + 1
            for t in np.linspace(0, 1, n + 1):
                yy = int(round(a[0] + (b[0] - a[0]) * t))
                xx = int(round(a[1] + (b[1] - a[1]) * t))
                if not trav[yy, xx]:
                    return False
            return True

        simp = [cells[0]]
        k = 0
        while k < len(cells) - 1:
            j = len(cells) - 1
            while j > k + 1 and not los(cells[k], cells[j]):
                j -= 1
            simp.append(cells[j])
            k = j
        pts = [self.p2w(cx * ds + ds / 2.0 - 0.5, cy * ds + ds / 2.0 - 0.5)
               for cy, cx in simp[1:]]
        pts[-1] = (float(goal[0]), float(goal[1]))
        return pts

    # ───────────── окно лидара ─────────────
    def render(self, x, y, yaw, viewpoints=(), cur_vp=-1, path=None,
               info_lines=(), view_m=24.0, scale=3):
        half = int(view_m / 2.0 / self.res)
        cx, cy = self.w2p(x, y)
        x0, y0 = cx - half, cy - half
        S = 2 * half
        img = np.full((S, S, 3), 35, np.uint8)

        def crop(a):
            out = np.zeros((S, S), a.dtype)
            sx0, sy0 = max(x0, 0), max(y0, 0)
            sx1, sy1 = min(x0 + S, self.n), min(y0 + S, self.n)
            if sx1 > sx0 and sy1 > sy0:
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
            return out

        fr = crop(self.free) >= 2
        img[fr] = (75, 75, 75)
        if self.room is not None:
            img[crop(self.room.mask) > 0] = (60, 95, 60)
        img[crop(self.hits) >= 2] = (255, 255, 255)
        img = cv2.resize(np.flipud(img), (S * scale, S * scale),
                         interpolation=cv2.INTER_NEAREST)
        img = np.ascontiguousarray(img)

        def D(px, py):
            ix = (px - self.ox) / self.res - x0
            iy = (py - self.oy) / self.res - y0
            return int(ix * scale), int((S - iy) * scale)

        if self.last_scan_xy is not None:
            for hx, hy in self.last_scan_xy[::2]:
                cv2.circle(img, D(hx, hy), 1, (0, 0, 255), -1)
        if self.room is not None:
            col = {'wall': (0, 220, 0), 'open': (0, 220, 255), 'door': (255, 0, 255)}
            for w in self.room.walls:
                cv2.line(img, D(*w.p1), D(*w.p2), col[w.kind], 2)
                if w.kind == 'wall' and w.length > 1.0:
                    m = (w.p1 + w.p2) / 2
                    cv2.putText(img, f'{w.dist:.1f}', D(*m), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 1)
            for v in self.room.corners:
                cv2.circle(img, D(*v), 6, (0, 140, 255), 2)
            for v in self.room.concave:
                cv2.drawMarker(img, D(*v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        for d in self.doors:
            c = (255, 0, 255) if d.confirmed else (120, 0, 120)
            cv2.line(img, D(*d.a), D(*d.b), c, 3)
            cv2.putText(img, f'{d.width:.1f}m', D(*d.center), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, c, 1)
        if path:
            pts = [D(x, y)] + [D(px, py) for px, py in path]
            cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 160, 0), 2)
        for k, v in enumerate(viewpoints):
            c = (130, 130, 130) if v.visited else (255, 255, 0)
            if k == cur_vp:
                c = (0, 0, 255)
            cv2.circle(img, D(v.x, v.y), 7, c, 2)
            cv2.putText(img, str(k + 1), (D(v.x, v.y)[0] + 8, D(v.x, v.y)[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            for h in v.headings:
                e = (v.x + 0.6 * math.cos(h), v.y + 0.6 * math.sin(h))
                cv2.line(img, D(v.x, v.y), D(*e), c, 1)
        # дрон
        p0 = np.array([x, y])
        tri = [p0 + 0.5 * np.array([math.cos(yaw), math.sin(yaw)]),
               p0 + 0.3 * np.array([math.cos(yaw + 2.5), math.sin(yaw + 2.5)]),
               p0 + 0.3 * np.array([math.cos(yaw - 2.5), math.sin(yaw - 2.5)])]
        cv2.fillPoly(img, [np.array([D(*t) for t in tri], np.int32)], (0, 0, 255))

        # панель текста
        panel = np.full((img.shape[0], 330, 3), 20, np.uint8)
        lines = []
        if self.room is not None:
            r = self.room
            lines += [f'Комната: {r.shape}',
                      f'Размер: {r.size[0]:.1f} x {r.size[1]:.1f} м',
                      f'Площадь: {r.area:.1f} м2',
                      'До стен по осям:',
                      '  ' + ' / '.join(f'{d:.1f}' for d in r.axis_dists),
                      f'Стен: {sum(w.kind == "wall" for w in r.walls)}  '
                      f'углов: {len(r.corners)}  вогн.: {len(r.concave)}']
        lines.append(f'Проёмов (~{DOOR_WIDTH:.0f} м): {len(self.confirmed_doors())}')
        lines.append(f'Сканов: {self.n_scans}  довернуто/срезано: '
                     f'{self.n_rot_corrected}/{self.n_rot_dropped}')
        lines += list(info_lines)
        for k, t in enumerate(lines):
            _put_text(panel, t, (8, 22 + 20 * k))
        legend = [('стена', (0, 220, 0)), ('открыто/тень', (0, 220, 255)),
                  ('проём', (255, 0, 255)), ('угол', (0, 140, 255)),
                  ('точка осмотра', (255, 255, 0)), ('путь', (255, 160, 0))]
        for k, (t, c) in enumerate(legend):
            yy = img.shape[0] - 20 * (len(legend) - k)
            cv2.rectangle(panel, (8, yy - 10), (20, yy), c, -1)
            _put_text(panel, t, (28, yy))
        return np.hstack([img, panel])


_FONT_CACHE = {}


def _put_text(img, text, org, color=(230, 230, 230)):
    """putText с поддержкой кириллицы (Pillow, если есть; иначе транслит)."""
    try:
        from PIL import Image as PImage, ImageDraw, ImageFont
        if 'f' not in _FONT_CACHE:
            font = None
            for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                         '/usr/share/fonts/TTF/DejaVuSans.ttf'):
                try:
                    font = ImageFont.truetype(path, 14)
                    break
                except OSError:
                    pass
            _FONT_CACHE['f'] = font or ImageFont.load_default()
        x, y = org
        h = 18
        roi = img[max(y - 14, 0):y + 4, x:min(x + 320, img.shape[1])]
        pil = PImage.fromarray(roi[:, :, ::-1].copy())
        ImageDraw.Draw(pil).text((0, 0), text, font=_FONT_CACHE['f'],
                                 fill=(color[2], color[1], color[0]))
        roi[:] = np.asarray(pil)[:, :, ::-1]
    except Exception:
        tr = str.maketrans('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
                           'abvgdeejzijklmnoprstufhccss_y_euaABVGDEEJZIJKLMNOPRSTUFHCCSS_Y_EUA')
        cv2.putText(img, text.translate(tr), org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1)
