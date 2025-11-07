import sys
from time import sleep
import threading
import math
from typing import Tuple, List
from path_planner import (
    plan_path_from_file, plan_path_from_struct,
    PlannerConfig, DEFAULT_CFG, load_zones_file,
    order_ccw, Point, Poly, WORK_AREA_RECT,
    point_in_convex_polygon, is_start_in_zones, find_nearest_safe_point,
)
from flight_ops import (
    fly_mission, connect_pioneer, arm_and_takeoff,
    fly_waypoints, land_and_wait, connect_and_takeoff_after_delay
)


def go(x, y, z):
    pio.go_to_local_point(x, y, z, 0)
    while not pio.point_reached():
        sleep(0.1)

def main():
    global pio
    if len(sys.argv) < 3:
        print("Использование: python solution_safe_start.py <ip> <mavlink_port> <zones_file>")
        sys.exit(1)

    ip = sys.argv[1]
    port = int(sys.argv[2])

    cfg: PlannerConfig = DEFAULT_CFG


    work_area: Tuple[float, float, float, float] = WORK_AREA_RECT

    pio = connect_pioneer(ip, port)
    arm_and_takeoff(pio, cfg)
    coords = pio.get_local_position_lps()
    go(coords[0], coords[1], 2,)
    pio.cargo_grab()
    go(0, 0, 2)
    pio.land()
    pio.takeoff()
    go(0, 0, 2)



if __name__ == "__main__":
    main()