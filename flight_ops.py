# flight_ops.py
# Обёртки над pioneer_sdk: подключение, взлёт, полёт по точкам, посадка.

import time, math
from typing import List, Tuple, Optional

from pioneer_sdk import Pioneer

from path_planner import PlannerConfig, DEFAULT_CFG

import threading

Point = Tuple[float, float]


def connect_pioneer(ip: str, port: int):
    """Создать экземпляр Pioneer и вернуть его."""
    from pioneer_sdk import Pioneer
    return Pioneer(ip=ip, mavlink_port=port, simulator=True)



def connect_and_takeoff_after_delay(
    ip: str,
    port: int,
    cfg: PlannerConfig,
    ready_event: threading.Event,
    need_relocation: bool,
    safe_start: Point,
    start_original: Point
) -> None:
    """Через 8 секунд выполняем подключение, взлёт и немедленную релокацию (если нужно)."""
    time.sleep(8)
    print("[INFO] Прошло 8 секунд — подключаюсь к дрону и выполняю взлёт...")
    pio = connect_pioneer(ip, port)
    arm_and_takeoff(pio, cfg)

    if need_relocation:
        print(f"[INFO] Немедленно перемещаюсь в безопасную точку {safe_start}")
        relocation_path = [start_original, safe_start]
        try:
            fly_waypoints(pio, relocation_path, z=cfg.CRUISE_Z, cfg=cfg, yaw=0.0)
            print("[INFO] Перемещение в безопасную точку завершено")
        except Exception as e:
            print(f"[ERROR] Ошибка при перемещении в безопасную точку: {e}")
            land_and_wait(pio, cfg)
            raise

    setattr(ready_event, "pio", pio)
    ready_event.set()


def arm_and_takeoff(pio, cfg: PlannerConfig = DEFAULT_CFG):
    """Арминг и взлёт; ожидаем фиксированное время (или тут можно добавить ожидание высоты, если доступно)."""
    print("[SDK] arm()")
    pio.arm()
    print("[SDK] takeoff()")
    pio.takeoff()

    t0 = time.time()
    while time.time() - t0 < cfg.TAKEOFF_TIMEOUT:
        time.sleep(0.2)


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def _norm2(x, y):
    return math.hypot(x, y)

def fly_waypoints(pio,
                  waypoints_xy: List[Point],
                  z: float,  # не используется
                  cfg: PlannerConfig = PlannerConfig(),
                  yaw: Optional[float] = 0.0):

    dt = 1.0 / cfg.MANUAL_HZ

    def _get_xy():
        pos = pio.get_local_position_lps(True)
        return (pos[0], pos[1]) if pos and len(pos) >= 2 else None

    n = len(waypoints_xy)
    i = 0
    t_segment_start = time.time()
    vx_last = vy_last = 0.0  # для оценки текущей скорости

    while i < n:
        xy = _get_xy()
        if xy is None:
            pio.set_manual_speed(0.0, 0.0, 0.0, 0.0)
            if time.time() - t_segment_start > cfg.WAIT_POINT_TIMEOUT:
                print("[WARN] нет координат слишком долго — пропускаю точку")
                i += 1
                t_segment_start = time.time()
            time.sleep(dt)
            continue

        cx, cy = xy
        tx, ty = waypoints_xy[i]

        ex, ey = (tx - cx), (ty - cy)
        dist_to_wp = _norm2(ex, ey)

        # --- динамический PASS_RADIUS (раньше «перескакиваем», если летим быстрее) ---
        pass_r = cfg.PASS_RADIUS
        if cfg.DYNAMIC_PASS:
            v_now = _norm2(vx_last, vy_last)
            pass_r = _clip(cfg.PASS_RADIUS + cfg.PASS_GAIN * v_now,
                           cfg.PASS_MIN, cfg.PASS_MAX)

        # «пролёт» промежуточных точек
        if cfg.FLY_THROUGH and i < n - 1 and dist_to_wp <= pass_r:
            i += 1
            t_segment_start = time.time()
            time.sleep(dt)
            continue

        # П-регулятор по горизонтали
        vx = cfg.KP_XY * ex
        vy = cfg.KP_XY * ey

        # ограничение по максимальной скорости
        vxy = _norm2(vx, vy)
        if vxy > cfg.V_MAX > 0.0:
            s = cfg.V_MAX / vxy
            vx *= s; vy *= s
            vxy = cfg.V_MAX

        # анти-залипание: если до WP далеко — не даём скорости падать ниже порога
        if dist_to_wp > cfg.FAR_DIST and vxy < cfg.V_MIN_FAR and dist_to_wp > 1e-3:
            ux, uy = (ex / dist_to_wp, ey / dist_to_wp)
            vx, vy = ux * cfg.V_MIN_FAR, uy * cfg.V_MIN_FAR
            vxy = cfg.V_MIN_FAR

        pio.set_manual_speed(vx, vy, 0.0, 0.0)
        vx_last, vy_last = vx, vy  # для динамического pass_r

        # финиш: без фиксации, если HOLD_AT_LAST=False
        if i == n - 1 and dist_to_wp <= cfg.REACHED_EPS:
            if cfg.HOLD_AT_LAST:
                t_hold = time.time() + cfg.HOLD_TIME
                while time.time() < t_hold:
                    pio.set_manual_speed(0.0, 0.0, 0.0, 0.0)
                    time.sleep(dt)
            break

        # защита от зависания на сегменте
        if time.time() - t_segment_start > cfg.WAIT_POINT_TIMEOUT:
            print("[WARN] timeout сегмента — иду дальше")
            i += 1
            t_segment_start = time.time()

        time.sleep(dt)

    # страховочный стоп
    for _ in range(2):
        pio.set_manual_speed(0.0, 0.0, 0.0, 0.0)
        time.sleep(0.03)

def fly_waypoints_old(pio, waypoints_xy: List[Point], z: float,
                      cfg: PlannerConfig = DEFAULT_CFG,
                      yaw: Optional[float] = 0.0):
    """Проход по маршруту: go_to_local_point -> ожидание point_reached() с таймаутом."""
    for k, (x, y) in enumerate(waypoints_xy):
        print(f"[SDK] go_to_local_point({x:.2f}, {y:.2f}, {z:.2f}, yaw={yaw}) [{k + 1}/{len(waypoints_xy)}]")
        pio.go_to_local_point(x, y, z, yaw or 0.0)

        t1 = time.time()
        while True:
            if pio.point_reached():
                break
            if time.time() - t1 > cfg.WAIT_POINT_TIMEOUT:
                print("[WARN] timeout ожидания точки — продолжаю")
                break
            time.sleep(0.1)


def land_and_wait(pio, cfg: PlannerConfig = DEFAULT_CFG):
    """Команда посадки и короткое ожидание завершения."""
    print("[SDK] land()")
    pio.land()
    t2 = time.time()
    while time.time() - t2 < cfg.LAND_TIMEOUT:
        time.sleep(0.2)
    print("[DONE] Посадка завершена.")


def fly_mission(pio: Pioneer, waypoints_xy: List[Point], z: float,
                cfg: PlannerConfig = DEFAULT_CFG, yaw: Optional[float] = 0.0):
    """
    Удобный «one-call»: подключиться -> взлететь -> пройти точки -> сесть.
    Если требуется более тонкий контроль, используйте функции по отдельности.
    """
    pio = pio
    fly_waypoints(pio, waypoints_xy, z, cfg, yaw=yaw)
    land_and_wait(pio, cfg)
