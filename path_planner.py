# path_planner.py
# Общий модуль: параметры, геометрия, граф, A*, сглаживание, планирование
import math
import heapq
from dataclasses import dataclass, replace
from typing import List, Tuple, Iterable, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed



# ---------------- Параметры по умолчанию ----------------
@dataclass(frozen=True)
class PlannerConfig:
    CRUISE_Z: float        = 2.0
    REACHED_EPS: float     = 0.15
    SAFETY_MARGIN: float   = 0.7    # инфляция зон для узлов/видимости и зазор от границы
    SIDE_OFFSET: float     = 1.0    # вынос “опорных” точек наружу от рёбер/углов
    INSIDE_WEIGHT: float   = 5.0    # штраф за 1 метр внутри зоны
    OUTSIDE_WEIGHT: float  = 1.0    # стоимость 1 метра вне зон
    ALLOW_CROSSING: bool   = True   # True: зоны = штраф (можно пересекать). False: запрет
    TAKEOFF_TIMEOUT: float = 0.5
    WAIT_POINT_TIMEOUT: float = 60.0
    LAND_TIMEOUT: float    = 20.0

    # управление скоростью
    MANUAL_HZ: float = 20.0
    V_MAX: float = 1.0  # подними до лимита симулятора, если можно
    KP_XY: float = 1.0  # чуть агрессивнее, чтобы быстрее разгонялся

    # пролёт без остановок
    FLY_THROUGH: bool = True
    PASS_RADIUS: float = 0.40  # ЧУТЬ БОЛЬШЕ, чтобы раньше переключаться
    HOLD_AT_LAST: bool = False  # на финише тоже не держим ноль (можно вернуть True)
    HOLD_TIME: float = 0.25

    # --- Ускорение пролёта (anti-stall) ---
    V_MIN_FAR: float = 0.30  # минимальная горизонтальная скорость, когда далеко от WP
    FAR_DIST: float = 1.0  # считаем «далеко от WP», если дистанция > FAR_DIST
    DYNAMIC_PASS: bool = True  # включить динамический PASS_RADIUS
    PASS_GAIN: float = 0.35  # добавка к PASS_RADIUS пропорц. текущей скорости (м на м/с)
    PASS_MIN: float = 0.35  # нижняя граница
    PASS_MAX: float = 0.80  # верхняя граница

    # --- Обогащение графа по рабочей зоне ---
    WA_PERIM_STEP: float = 5.0  # шаг точек по периметру (м). 0 = только углы и середины
    WA_GRID_STEP: float = 0.0  # шаг внутренней сетки (м). 0 = не добавлять сетку

    # --- Индекс препятствий ---
    GRID_CELL: float = 5.0  # размер ячейки индекса зон (м)


DEFAULT_CFG = PlannerConfig()
EPS = 1e-9


# ---------------- Рабочая зона (ось-ориентированный прямоугольник, м) ----------------
# Задана точками: (41, 17.5); (41, -16); (-13, -16); (-13, 17.5)
# В формате (xmin, ymin, xmax, ymax):
WORK_AREA_RECT = (-13.0, -16.0, 41.0, 17.5)



def point_in_convex_polygon(p: "Point", poly: "Poly") -> bool:
    """Проверка, находится ли точка внутри выпуклого многоугольника (CCW)."""
    poly = order_ccw(poly)
    n = len(poly)
    for i in range(n):
        v1 = poly[i]
        v2 = poly[(i + 1) % n]
        # (v2-v1) × (p-v1)
        cross = (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0])
        if cross < -1e-9:  # точка справа от ребра => вне полигона
            return False
    return True


def is_start_in_zones(start: "Point", zones: List["Poly"]) -> bool:
    """True, если стартовая точка попадает хотя бы в одну зону."""
    for zone in zones:
        if point_in_convex_polygon(start, zone):
            return True
    return False


def find_nearest_safe_point(
    start: "Point",
    zones: List["Poly"],
    work_area: Tuple[float, float, float, float],
    margin: float = 0.5
) -> "Point":
    """
    Находит ближайшую безопасную точку вне всех запретных зон.
    Стратегия:
      1) точки на границах зон с выносом наружу на 'margin'
      2) углы и центр рабочей зоны
      3) редкая сетка внутри рабочей зоны (fallback)
    """
    import math

    xmin, ymin, xmax, ymax = work_area
    candidates: List[Tuple[float, "Point"]] = []  # (distance, point)

    def add_candidate(p: "Point"):
        if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
            return
        if not is_start_in_zones(p, zones):
            dist = math.hypot(p[0] - start[0], p[1] - start[1])
            candidates.append((dist, p))

    # 1) точки вдоль рёбер зон с внешней нормалью (для CCW)
    for zone in zones:
        n = len(zone)
        for i in range(n):
            v1 = zone[i]
            v2 = zone[(i + 1) % n]
            edge = (v2[0] - v1[0], v2[1] - v1[1])
            normal = (edge[1], -edge[0])  # наружу для CCW
            length = (normal[0]**2 + normal[1]**2) ** 0.5
            if length < 1e-9:
                continue
            normal = (normal[0]/length, normal[1]/length)

            for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                edge_point = (v1[0] + t*(v2[0]-v1[0]), v1[1] + t*(v2[1]-v1[1]))
                safe_point = (edge_point[0] + margin*normal[0],
                              edge_point[1] + margin*normal[1])
                add_candidate(safe_point)

    # 2) углы и центр рабочей зоны
    corners = [
        (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
        ((xmin+xmax)/2.0, (ymin+ymax)/2.0),
    ]
    for c in corners:
        add_candidate(c)

    # 3) сетка, если кандидатов мало
    if len(candidates) < 10:
        step = min(max((xmax - xmin) / 10.0, 1e-6), max((ymax - ymin) / 10.0, 1e-6))
        x = xmin + step
        while x < xmax:
            y = ymin + step
            while y < ymax:
                add_candidate((x, y))
                y += step
            x += step

    if not candidates:
        raise RuntimeError("Не удалось найти безопасную точку! Все доступные точки находятся в запретных зонах.")

    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def _rect_norm(rect):
    xmin, ymin, xmax, ymax = rect
    if xmin > xmax: xmin, xmax = xmax, xmin
    if ymin > ymax: ymin, ymax = ymax, ymin
    return xmin, ymin, xmax, ymax

def _rect_extent(rect) -> float:
    xmin, ymin, xmax, ymax = _rect_norm(rect)
    return max(xmax - xmin, ymax - ymin, 1.0)

def point_in_rect(p: Tuple[float,float], rect, rel_tol: float = 1e-6) -> bool:
    """Точка внутри/на границе прямоугольника с масштабной толерантностью."""
    xmin, ymin, xmax, ymax = _rect_norm(rect)
    tol = rel_tol * _rect_extent(rect)
    x, y = p
    return (x >= xmin - tol) and (x <= xmax + tol) and (y >= ymin - tol) and (y <= ymax + tol)

def segment_inside_rect(p0, p1, rect, rel_tol: float = 1e-6) -> bool:
    """
    Весь отрезок внутри прямоугольника? — Liang–Barsky + допуск по длине.
    Устойчиво для граничных случаев.
    """
    xmin, ymin, xmax, ymax = _rect_norm(rect)
    x0, y0 = p0; x1, y1 = p1
    dx = x1 - x0; dy = y1 - y0
    seglen = math.hypot(dx, dy)
    if seglen < 1e-12:
        return point_in_rect(p0, rect, rel_tol)

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-15:
            if qi < 0:
                return False
            continue
        t = qi / pi
        if pi < 0:
            if t > u1: u1 = t
        else:
            if t < u2: u2 = t
        if u1 - u2 > 1e-15:
            return False

    len_tol = rel_tol * _rect_extent(rect)
    clipped_len = max(0.0, u2 - u1) * seglen
    return abs(clipped_len - seglen) <= len_tol

# ---------------- Типы ----------------
Point = Tuple[float, float]
Poly  = List[Point]

@dataclass
class Node:
    idx: int
    p: Point

# ---------------- Вспомогательная геометрия ----------------
def _dot(a: Point, b: Point) -> float: return a[0]*b[0] + a[1]*b[1]
def _sub(a: Point, b: Point) -> Point: return (a[0]-b[0], a[1]-b[1])
def _add(a: Point, b: Point) -> Point: return (a[0]+b[0], a[1]+b[1])
def _mul(a: Point, s: float) -> Point:  return (a[0]*s, a[1]*s)
def _len(a: Point) -> float:            return math.hypot(a[0], a[1])

def order_ccw(poly: List[Point]) -> List[Point]:
    """Упорядочить вершины выпуклого многоугольника против часовой (CCW)."""
    if not poly:
        return []
    cx = sum(p[0] for p in poly)/len(poly)
    cy = sum(p[1] for p in poly)/len(poly)
    pts = sorted(poly, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
    # shoelace sign: >0 => CCW, <0 => CW
    area = 0.0
    for i in range(len(pts)):
        x1,y1 = pts[i]
        x2,y2 = pts[(i+1) % len(pts)]
        area += x1*y2 - x2*y1
    if area < 0:
        pts.reverse()
    return pts

def _inward_normal_ccw(poly: Poly, i: int) -> Point:
    a = poly[i]; b = poly[(i+1) % len(poly)]
    e = _sub(b, a)
    n = (e[1], -e[0])  # inward для CCW (сохраняем как в рабочем коде)
    L = _len(n) or 1.0
    return (n[0]/L, n[1]/L)

def _edge_normal_out_ccw(poly: Poly, i: int) -> Point:
    a = poly[i]; b = poly[(i+1) % len(poly)]
    e = _sub(b, a)
    n = (e[1], -e[0])  # outward для CCW
    L = _len(n) or 1.0
    return (n[0]/L, n[1]/L)

def inflate_poly(poly: Poly, margin: float) -> Poly:
    """Офсет (инфляция) выпуклого полигона наружу на margin."""
    poly = order_ccw(poly)
    n = len(poly)
    lines = []
    for i in range(n):
        a = poly[i]
        nout = _edge_normal_out_ccw(poly, i)
        # прямая: n·x = n·a + margin
        lines.append((nout[0], nout[1], _dot(nout, a) + margin))
    def intersect(L1, L2):
        a,b,c = L1; d,e,f = L2
        det = a*e - b*d
        if abs(det) < 1e-12:
            return (float('inf'), float('inf'))
        x = (c*e - b*f)/det
        y = (a*f - c*d)/det
        return (x,y)
    out = []
    for i in range(n):
        out.append(intersect(lines[i], lines[(i+1) % n]))
    return out

def segment_clip_length_convex(p0: Point, p1: Point, poly: Poly) -> float:
    """Длина части отрезка p0->p1, лежащей строго внутри выпуклого многоугольника (граница не считается)."""
    poly = order_ccw(poly)
    d = _sub(p1, p0)
    tE, tL = 0.0, 1.0
    for i in range(len(poly)):
        vi = poly[i]
        n  = _inward_normal_ccw(poly, i)
        w  = n[0]*vi[0] + n[1]*vi[1]
        num = w - (n[0]*p0[0] + n[1]*p0[1])
        den = n[0]*d[0] + n[1]*d[1]
        if abs(den) < 1e-12:
            if num < 0:
                return 0.0  # параллельно и снаружи
            continue
        t = num / den
        if den > 0:  # leaving
            if t < tL: tL = t
        else:        # entering
            if t > tE: tE = t
        if tE - tL > 1e-12:
            return 0.0
    if tL <= tE + EPS:
        return 0.0
    return _len(d) * (tL - tE)

# ---------------- Ускорение: предвычисления для зон + грид-индекс ----------------
@dataclass
class Zone:
    poly: Poly
    normals: List[Point]                         # inward нормали по рёбрам
    aabb: Tuple[float, float, float, float]      # xmin, ymin, xmax, ymax

def _aabb_of_poly(poly: Poly) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))

def _precompute_zone(poly_ccw: Poly) -> Zone:
    m = len(poly_ccw)
    normals: List[Point] = []
    for i in range(m):
        a = poly_ccw[i]; b = poly_ccw[(i+1) % m]
        e = _sub(b, a)
        n = (e[1], -e[0])  # inward CCW в «рабочем» знаке
        L = _len(n) or 1.0
        normals.append((n[0]/L, n[1]/L))
    return Zone(poly_ccw, normals, _aabb_of_poly(poly_ccw))

class ZoneGrid:
    """Простой равномерный грид-индекс для отбора кандидатов зон по AABB отрезка."""
    def __init__(self, zones: List[Zone], cell: float = 5.0):
        self.zones = zones
        xmin, ymin, xmax, ymax = WORK_AREA_RECT
        self.cell = max(cell, 1e-3)
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        nx = max(1, int((xmax-xmin)/self.cell))
        ny = max(1, int((ymax-ymin)/self.cell))
        self.nx, self.ny = nx, ny
        self.cells: List[List[int]] = [[] for _ in range(nx*ny)]
        for zi, z in enumerate(zones):
            ax, ay, bx, by = z.aabb
            ix0 = max(0, int((ax - xmin)//self.cell))
            iy0 = max(0, int((ay - ymin)//self.cell))
            ix1 = min(nx-1, int((bx - xmin)//self.cell))
            iy1 = min(ny-1, int((by - ymin)//self.cell))
            for ix in range(ix0, ix1+1):
                for iy in range(iy0, iy1+1):
                    self.cells[iy*nx + ix].append(zi)

    def query_segment_candidates(self, p0: Point, p1: Point) -> List[int]:
        ax = min(p0[0], p1[0]); ay = min(p0[1], p1[1])
        bx = max(p0[0], p1[0]); by = max(p0[1], p1[1])
        xmin, ymin, xmax, ymax = self.xmin, self.ymin, self.xmax, self.ymax
        ix0 = max(0, int((ax - xmin)//self.cell))
        iy0 = max(0, int((ay - ymin)//self.cell))
        ix1 = min(self.nx-1, int((bx - xmin)//self.cell))
        iy1 = min(self.ny-1, int((by - ymin)//self.cell))
        seen = set()
        for ix in range(ix0, ix1+1):
            for iy in range(iy0, iy1+1):
                for zi in self.cells[iy*self.nx + ix]:
                    seen.add(zi)
        return list(seen)

def segment_clip_length_convex_fast(p0: Point, p1: Point, zone: Zone) -> float:
    """Быстрый клиппинг p0->p1 по выпуклой зоне с предвычисленными нормалями."""
    poly = zone.poly; normals = zone.normals
    d = _sub(p1, p0)
    tE, tL = 0.0, 1.0
    for i in range(len(poly)):
        vi = poly[i]; n = normals[i]
        w  = n[0]*vi[0] + n[1]*vi[1]
        num = w - (n[0]*p0[0] + n[1]*p0[1])
        den = n[0]*d[0] + n[1]*d[1]
        if abs(den) < 1e-12:
            if num < 0:
                return 0.0
            continue
        t = num / den
        if den > 0:
            if t < tL: tL = t
        else:
            if t > tE: tE = t
        if tE - tL > 1e-12:
            return 0.0
    if tL <= tE + EPS:
        return 0.0
    return _len(d) * (tL - tE)

# ---------------- Стоимость и видимость ----------------
# Кэш стоимостей рёбер (симметричный)
_EDGE_COST_CACHE: Dict[Tuple[float,float,float,float], float] = {}

def _edge_key(p1: Point, p2: Point) -> Tuple[float,float,float,float]:
    return (p1[0], p1[1], p2[0], p2[1]) if p1 <= p2 else (p2[0], p2[1], p1[0], p1[1])

def make_zone_index(zones_ccw: List[Poly], grid_cell: float = 5.0):
    zones_fast = [_precompute_zone(poly) for poly in zones_ccw]
    zgrid = ZoneGrid(zones_fast, cell=grid_cell)
    return zones_fast, zgrid

def edge_cost_indexed(p1: Point, p2: Point, zones: List[Zone], zgrid: ZoneGrid,
                      inside_w: float, outside_w: float) -> float:
    key = _edge_key(p1, p2)
    cached = _EDGE_COST_CACHE.get(key)
    if cached is not None:
        return cached
    total  = math.dist(p1, p2)
    inside = 0.0
    # берём только кандидатов из индекса + дёшевый AABB-предтест
    ax = min(p1[0], p2[0]); ay = min(p1[1], p2[1])
    bx = max(p1[0], p2[0]); by = max(p1[1], p2[1])
    for zi in zgrid.query_segment_candidates(p1, p2):
        z = zones[zi]
        zx0, zy0, zx1, zy1 = z.aabb
        if max(ax, zx0) > min(bx, zx1): continue
        if max(ay, zy0) > min(by, zy1): continue
        inside += segment_clip_length_convex_fast(p1, p2, z)
        if inside >= total:
            inside = total; break
    cost = outside_w*(total - inside) + inside_w*inside
    _EDGE_COST_CACHE[key] = cost
    return cost

def edge_cost(p1: Point, p2: Point, zones_ccw: Iterable[Poly],
              inside_w: float, outside_w: float) -> float:
    """Базовая версия (оставлена для совместимости). Оптимизированная — edge_cost_indexed."""
    total  = math.dist(p1, p2)
    inside = 0.0
    for poly in zones_ccw:
        inside += segment_clip_length_convex(p1, p2, poly)
    if inside > total:
        inside = total
    return outside_w*(total - inside) + inside_w*inside

def visible(p1: Point, p2: Point, inflated_zones: Iterable[Poly], allow_crossing: bool) -> bool:
    # --- Ограничение рабочей зоной: весь отрезок должен лежать внутри прямоугольника ---
    if WORK_AREA_RECT is not None:
        if not segment_inside_rect(p1, p2, WORK_AREA_RECT, rel_tol=1e-6):
            return False
    # --- Пересечение с инфлированными зонами (если запрещено) ---
    if allow_crossing:
        return True
    for poly in inflated_zones:
        if segment_clip_length_convex(p1, p2, poly) > EPS:
            return False
    return True

# ---------------- Узлы: добавляем рабочую зону + зону препятствий ----------------
def _work_area_nodes(perim_step: float, grid_step: float) -> List[Point]:
    """Опорные точки внутри рабочей зоны: углы, середины, точки по периметру, внутренняя сетка."""
    xmin, ymin, xmax, ymax = _rect_norm(WORK_AREA_RECT)
    nodes: List[Point] = []
    # углы и середины
    corners = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    mids = [((xmin+xmax)/2, ymin), (xmax, (ymin+ymax)/2),
            ((xmin+xmax)/2, ymax), (xmin, (ymin+ymax)/2)]
    nodes.extend(corners); nodes.extend(mids)

    # равномерно по периметру
    if perim_step and perim_step > 0:
        def linspace(a,b,step):
            n = max(0, int(math.floor((b-a)/step)))
            return [a + i*step for i in range(1, n)]  # без концов
        for x in linspace(xmin, xmax, perim_step):
            nodes.append((x, ymin))
            nodes.append((x, ymax))
        for y in linspace(ymin, ymax, perim_step):
            nodes.append((xmin, y))
            nodes.append((xmax, y))

    # внутренняя сетка (по центрам ячеек)
    if grid_step and grid_step > 0:
        nx = max(1, int((xmax - xmin) // grid_step))
        ny = max(1, int((ymax - ymin) // grid_step))
        if nx > 0 and ny > 0:
            sx = (xmax - xmin) / nx
            sy = (ymax - ymin) / ny
            for i in range(nx):
                cx = xmin + (i + 0.5)*sx
                for j in range(ny):
                    cy = ymin + (j + 0.5)*sy
                    nodes.append((cx, cy))
    return nodes

# ---------------- Построение графа ----------------
def build_nodes(start: Point, goal: Point, zones_ccw: List[Poly], cfg: PlannerConfig) -> List[Node]:
    """
    Узлы:
      - старт, финиш,
      - опорные точки по рабочей зоне (углы, середины, периметр, внутренняя сетка),
      - вершины инфлированной зоны,
      - середины рёбер, вынесенные наружу,
      - “угловые выносы” по биссектрисам внешних нормалей.
    Вспомогательные узлы, оказавшиеся вне рабочей зоны, отбрасываются.
    """
    nodes: List[Node] = [Node(0, start), Node(1, goal)]
    idx = 2

    def _try_add(pt: Point):
        nonlocal idx
        if (WORK_AREA_RECT is None) or point_in_rect(pt, WORK_AREA_RECT, rel_tol=1e-6):
            nodes.append(Node(idx, pt)); idx += 1

    # 0) опорные точки по рабочей зоне
    for p in _work_area_nodes(cfg.WA_PERIM_STEP, cfg.WA_GRID_STEP):
        _try_add(p)

    # 1) узлы от инфлированных запретных полигонов
    for poly in zones_ccw:
        infl = inflate_poly(poly, cfg.SAFETY_MARGIN)
        m = len(infl)

        # вершины инфлированного полигона
        for v in infl:
            _try_add(v)

        # середины рёбер + внешний вынос
        for i in range(m):
            a = infl[i]; b = infl[(i+1) % m]
            mid = ((a[0]+b[0]) * 0.5, (a[1]+b[1]) * 0.5)
            nout = _edge_normal_out_ccw(infl, i)
            support = _add(mid, _mul(nout, cfg.SIDE_OFFSET))
            _try_add(support)

        # угловые выносы по биссектрисе внешних нормалей
        for i in range(m):
            n_prev = _edge_normal_out_ccw(infl, (i-1) % m)
            n_i    = _edge_normal_out_ccw(infl, i)
            bis = _add(n_prev, n_i)
            L = _len(bis)
            if L > 1e-9:
                bis = (bis[0]/L, bis[1]/L)
                _try_add(_add(infl[i], _mul(bis, cfg.SIDE_OFFSET)))

    return nodes

def build_edges(nodes: List[Node], zones_ccw: List[Poly], cfg: PlannerConfig):
    # индекс и предвычисления для ускорения
    zones_fast, zgrid = make_zone_index(zones_ccw, grid_cell=cfg.GRID_CELL)

    inflated = [inflate_poly(p, cfg.SAFETY_MARGIN) for p in zones_ccw]
    edges = []
    for i in range(len(nodes)):
        p1 = nodes[i].p
        for j in range(i+1, len(nodes)):
            p2 = nodes[j].p
            if not visible(p1, p2, inflated, cfg.ALLOW_CROSSING):
                continue
            c = edge_cost_indexed(p1, p2, zones_fast, zgrid,
                                  cfg.INSIDE_WEIGHT, cfg.OUTSIDE_WEIGHT)
            edges.append((i, j, c))
            edges.append((j, i, c))
    return edges

def astar(nodes: List[Node], edges, start_idx=0, goal_idx=1) -> List[int]:
    nbrs = [[] for _ in nodes]
    for u,v,w in edges:
        nbrs[u].append((v,w))
    h = lambda i: math.dist(nodes[i].p, nodes[goal_idx].p)

    g = [math.inf]*len(nodes)
    par = [-1]*len(nodes)
    g[start_idx] = 0.0
    pq = [(h(start_idx), start_idx)]
    while pq:
        _, u = heapq.heappop(pq)
        if u == goal_idx: break
        for v, w in nbrs[u]:
            ng = g[u] + w
            if ng < g[v]:
                g[v] = ng
                par[v] = u
                heapq.heappush(pq, (ng + h(v), v))
    if par[goal_idx] == -1 and start_idx != goal_idx:
        raise RuntimeError("Путь не найден.")
    path = []
    cur = goal_idx
    while cur != -1:
        path.append(cur); cur = par[cur]
    path.reverse()
    return path

def smooth_path(pts: List[Point], zones_ccw: List[Poly], cfg: PlannerConfig) -> List[Point]:
    """
    Кост-осведомлённое сглаживание: заменяем подпуть i..j на прямую i->j
    только если cost(i->j) <= cost(i..j) - EPS, и при этом прямой шорткат:
      — остаётся внутри рабочей зоны;
      — (если ALLOW_CROSSING=False) не заходит в инфлированные полигоны.
    """
    if len(pts) <= 2:
        return pts
    # индекс для быстрых стоимостей
    zones_fast, zgrid = make_zone_index(zones_ccw, grid_cell=cfg.GRID_CELL)

    EPS_COST = 1e-9
    n = len(pts)
    seg_cost = [edge_cost_indexed(pts[k], pts[k+1], zones_fast, zgrid,
                                  cfg.INSIDE_WEIGHT, cfg.OUTSIDE_WEIGHT)
                for k in range(n-1)]
    out = [pts[0]]
    i = 0

    inflated = [inflate_poly(p, cfg.SAFETY_MARGIN) for p in zones_ccw] if not cfg.ALLOW_CROSSING else []

    while i < n-1:
        best_j = i+1
        j = n-1
        while j > i+1:
            # шорткат должен быть допустим внутри рабочей зоны
            if not segment_inside_rect(pts[i], pts[j], WORK_AREA_RECT, rel_tol=1e-6):
                j -= 1; continue
            # и не пересекать инфлированные зоны, если crossing запрещён
            if not cfg.ALLOW_CROSSING:
                if any(segment_clip_length_convex(pts[i], pts[j], poly) > EPS for poly in inflated):
                    j -= 1; continue

            direct = edge_cost_indexed(pts[i], pts[j], zones_fast, zgrid,
                                       cfg.INSIDE_WEIGHT, cfg.OUTSIDE_WEIGHT)
            chain  = sum(seg_cost[k] for k in range(i, j))
            if direct <= chain - EPS_COST:
                best_j = j
                break
            j -= 1
        out.append(pts[best_j])
        i = best_j
    return out

# ---------------- Загрузка файла зон ----------------
def load_zones_file(path: str) -> Tuple[Point, Point, List[Poly]]:
    """
    Формат:
        x_start y_start
        x_finish y_finish
        N
        x1 y1 x2 y2 x3 y3 x4 y4   # 4 вершины прямоугольника в любом порядке (можно повернутый)
        ...
    """
    def next_line(it):
        for line in it:
            s = line.strip()
            if not s or s.startswith("#"): continue
            return s
        raise ValueError("Неожиданный конец файла zones.txt")

    with open(path, "r", encoding="utf-8") as f:
        it = iter(f.readlines())

    sx, sy = map(float, next_line(it).split())
    fx, fy = map(float, next_line(it).split())
    n = int(next_line(it).split()[0])

    zones: List[Poly] = []
    for k in range(n):
        vals = list(map(float, next_line(it).split()))
        if len(vals) != 8:
            raise ValueError(f"Зона {k+1}: ожидалось 8 чисел, получено {len(vals)}")
        pts = [(vals[i], vals[i+1]) for i in range(0, 8, 2)]
        zones.append(order_ccw(pts))
    return (sx, sy), (fx, fy), zones

# ---------------- Высокоуровневые функции планирования ----------------
def plan_path_from_struct(start_m: Point, goal_m: Point, zones_polys_m: List[Poly],
                          cfg: PlannerConfig = DEFAULT_CFG) -> Tuple[List[Point], List[Point]]:
    """Построить сырой и сглаженный путь из уже подготовленных структур (метры).
       Рабочая зона ограничивает узлы/рёбра изнутри прямоугольника WORK_AREA_RECT."""
    zones_ccw = [order_ccw(poly) for poly in zones_polys_m]

    # Динамическая густота внутренней сетки:
    # если зон <= 10 — используем более плотную сетку (WA_GRID_STEP=8.0),
    # если зон > 10 — оставляем значения из cfg как есть.
    effective_cfg = cfg if len(zones_ccw) > 30 else replace(cfg, WA_GRID_STEP=5.0)

    nodes = build_nodes(start_m, goal_m, zones_ccw, effective_cfg)
    edges = build_edges(nodes, zones_ccw, effective_cfg)
    idx_path = astar(nodes, edges, 0, 1)
    raw = [nodes[i].p for i in idx_path]
    path = smooth_path(raw, zones_ccw, effective_cfg)
    return raw, path

def plan_path_from_file(zones_path: str,
                        cfg: PlannerConfig = DEFAULT_CFG) -> Tuple[List[Point], List[Point]]:
    """Прочитать файл зон, построить сырой и сглаженный путь (в метрах)."""
    start, goal, zones = load_zones_file(zones_path)
    return plan_path_from_struct(start, goal, zones, cfg)

def reset_edge_cost_cache() -> None:
    """Сброс кэша стоимостей рёбер (используется визуальным редактором)."""
    _EDGE_COST_CACHE.clear()
