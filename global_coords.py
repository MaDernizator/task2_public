import math





def pixel_to_lps(u, v,
                 drone_x,  # Текущая координата дрона X в LPS
                 drone_y,  # Текущая координата дрона Y в LPS
                 drone_alt,  # Высота дрона над землёй (м)
                 drone_yaw,  # Текущий рыскательный угол дрона (рад)
                 image_width=640,  # Ширина кадра (пиксели)
                 image_height=480,  # Высота кадра (пиксели)
                 fov_x_deg=90,  # Угол обзора камеры по горизонтали (градусы)
                 fov_y_deg=67.5  # Угол обзора камеры по вертикали (градусы)
                 ):
    """
    Проецирует пиксель (u,v) из кадра в глобальные координаты (X, Y) LPS на земле.
    Предполагается, что камера смотрит строго вниз, а дрон находится на высоте drone_alt.
    """

    # Переводим FOV из градусов в радианы
    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)

    # Горизонтальный/вертикальный "размах" (ширина/высота) участка земли, попадающего в кадр
    coverage_x = 2 * drone_alt * math.tan(fov_x / 2)
    coverage_y = 2 * drone_alt * math.tan(fov_y / 2)

    # Нормализуем пиксель (u,v) в диапазон [-0.5 .. +0.5] по обеим осям
    # (0,0) пиксель -> (-0.5, -0.5)
    # (image_width, image_height) пиксель -> (+0.5, +0.5)
    nx = (u / (image_width - 1)) - 0.5
    ny = (v / (image_height - 1)) - 0.5

    # Определяем локальные координаты на плоскости (X' в теле дрона, Y' в теле дрона)
    # Центр изображения считается над точкой (0,0) в LPS
    # Если надо "перевернуть" ось Y — это можно учесть знаком при вычислениях
    local_x = nx * coverage_x
    local_y = -ny * coverage_y

    # Переходим из системы дрона в глобальную LPS с учётом yaw
    # В LPS:
    #   x_new = x_drone +  (x_local*cos(yaw) - y_local*sin(yaw))
    #   y_new = y_drone +  (x_local*sin(yaw) + y_local*cos(yaw))
    global_x = drone_x + (local_x * math.cos(drone_yaw) - local_y * math.sin(drone_yaw))
    global_y = drone_y + (local_x * math.sin(drone_yaw) + local_y * math.cos(drone_yaw))

    return global_x, global_y


def lps_to_pixel(global_x, global_y,
                 drone_x,          # Координата дрона X в LPS
                 drone_y,          # Координата дрона Y в LPS
                 drone_alt,        # Высота дрона над землёй (м)
                 drone_yaw,        # Рыскательный угол дрона (рад)
                 image_width=640,  # Ширина кадра (пиксели)
                 image_height=480, # Высота кадра (пиксели)
                 fov_x_deg=90,     # Угол обзора камеры по горизонтали (°)
                 fov_y_deg=67.5    # Угол обзора камеры по вертикали (°)
                 ):
    """
    Преобразует точку на земле в LPS‑координатах (global_x, global_y) в координаты пикселя (u, v)
    на изображении, снятом отвесной (nadir) камерой дрона.

    Возвращаемые u и v ‑ вещественные; если нужны целые координаты, просто округлите.
    Если точка лежит вне поля зрения камеры (у < 0 или v < 0, либо у > image_width‑1 /
    v > image_height‑1), она физически не попадает в кадр.
    """

    # 1. Размер покрытия земли (м) по каждой оси
    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)
    coverage_x = 2 * drone_alt * math.tan(fov_x / 2)
    coverage_y = 2 * drone_alt * math.tan(fov_y / 2)

    # 2. Вектор от дрона до точки в глобальных координатах
    dx = global_x - drone_x
    dy = global_y - drone_y

    # 3. Переход в систему дрона (поворот на –yaw)
    local_x =  dx * math.cos(drone_yaw) + dy * math.sin(drone_yaw)
    local_y = -dx * math.sin(drone_yaw) + dy * math.cos(drone_yaw)

    # 4. Нормализованные координаты в диапазоне [-0.5 … +0.5]
    nx =  local_x / coverage_x
    ny = -local_y / coverage_y        # «–» — потому что ось v растёт вниз

    # 5. Перевод в пиксели
    u = (nx + 0.5) * (image_width  - 1)
    v = (ny + 0.5) * (image_height - 1)

    return u, v
