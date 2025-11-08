from pioneer_sdk import Pioneer, Camera
import cv2
import numpy as np
import time
import sys
import math
from typing import List, Tuple, Optional
import threading
from global_coords import *

class CargoData:
    """Данные о грузе"""
    def __init__(self, world_x: float, world_y: float, area: float):
        self.world_x = world_x
        self.world_y = world_y
        self.area = area
        self.captured = False
    
    def distance_from(self, x: float, y: float) -> float:
        return math.sqrt((self.world_x - x)**2 + (self.world_y - y)**2)


class CargoHunter:
    """Охотник за грузами с двойной визуализацией"""

    def __init__(self, drone: Pioneer, camera: Camera):
        self.drone = drone
        self.camera = camera

        # Параметры полета
        self.SCAN_HEIGHT = 6.0
        self.APPROACH_HEIGHT = 3.0  # Высота для стабилизации перед посадкой
        self.TRANSPORT_HEIGHT = 2.0
        self.RECON_RADIUS = 5.0

        # Корзина
        self.basket_x = None
        self.basket_y = None
        self.BASKET_EXCLUSION_RADIUS = 2.0

        # Детекция - АГРЕССИВНЫЕ параметры для маленьких колец
        self.GREEN_LOWER = np.array([35, 60, 40])
        self.GREEN_UPPER = np.array([95, 255, 255])
        self.MIN_AREA = 50  # Минимум для маленьких колец
        self.MAX_AREA = 20000

        # Камера
        self.IMG_WIDTH = None
        self.IMG_HEIGHT = None
        self.CENTER_X = None
        self.CENTER_Y = None
        self.camera_calibrated = False

        # Карта грузов
        self.cargo_map: List[CargoData] = []
        self.MIN_CARGO_DISTANCE = 0.1

        # Статистика
        self.delivered = 0
        self.start_time = None

        self.last_position = []

        # Визуализация - ДВА ОКНА
        self.latest_frame = None
        self.latest_processed = None
        self.frame_lock = threading.Lock()
        self.visualization_active = True
        

    def _visualization_loop(self):
        """ПОСТОЯННЫЙ поток визуализации - ДВА ОКНА"""
        cv2.namedWindow('RAW Camera', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RAW Camera', 640, 480)
        cv2.resizeWindow('Detection', 640, 480)
        
        while self.visualization_active:
            frame = self.camera.get_cv_frame()
            self.latest_frame = frame
            processed = self.latest_processed
            
            if frame is not None:
                cv2.imshow('RAW Camera', frame)
            
            if processed is not None:
                cv2.imshow('Detection', processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.03)  # ~30 FPS

    def log(self, message: str):
        """Лог с таймером"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"[{elapsed:6.1f}s] {message}")
        else:
            print(f"[INIT] {message}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Получение кадра"""
        try:
            frame = self.latest_frame
            # Обновляем RAW окно
            if frame is not None:
                with self.frame_lock:
                    self.latest_frame=frame.copy()
                return frame
            return None
        except:
            return None

    def get_position(self) -> Tuple[float, float, float]:
        """Текущая позиция"""
        try:
            pos = self.drone.get_local_position_lps()
            if pos is None:
                return (self.last_position[0], self.last_position[1], self.last_position[2],)
            print("[POSITION]", pos)
            self.last_position = [pos[0], pos[1], pos[2]]
            return (pos[0], pos[1], pos[2])
        except:
            return (0.0, 0.0, 0.0)

    def calibrate_camera(self) -> bool:
        """Калибровка камеры"""
        self.log("Калибровка камеры...")

        for attempt in range(10):
            frame = self.get_frame()

            if frame is not None:
                height, width = frame.shape[:2]

                if height > 0 and width > 0:
                    self.IMG_HEIGHT = height
                    self.IMG_WIDTH = width
                    self.CENTER_X = width // 2
                    self.CENTER_Y = height // 2
                    self.camera_calibrated = True

                    self.log(f"✓ Камера: {width}x{height}, центр: ({self.CENTER_X}, {self.CENTER_Y})")
                    time.sleep(0.1)
                    return True

        self.log("!!! ОШИБКА калибровки")
        return False

    def quick_goto(self, x: float, y: float, z: float, timeout: float = 15.0) -> bool:
        """Полет к точке"""
        self.drone.go_to_local_point(x=x, y=y, z=z, yaw=0)
        while not self.drone.point_reached():
            time.sleep(0.01)
        return True

    def detect_rings_advanced(self, frame: np.ndarray, debug: bool = False) -> List[Tuple[int, int, float]]:
        """
        ПРОДВИНУТАЯ детекция маленьких колец с визуализацией
        """
        if frame is None or not self.camera_calibrated:
            return []

        # HSV конверсия
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создаём маску
        mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)

        # Морфология - убираем шум, но сохраняем маленькие объекты
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Визуализация для окна Detection
        vis_frame = frame.copy()
        
        # Рисуем центр
        cv2.line(vis_frame, (self.CENTER_X, 0), (self.CENTER_X, self.IMG_HEIGHT), (0, 255, 255), 2)
        cv2.line(vis_frame, (0, self.CENTER_Y), (self.IMG_WIDTH, self.CENTER_Y), (0, 255, 255), 2)
        cv2.circle(vis_frame, (self.CENTER_X, self.CENTER_Y), 15, (0, 255, 255), 2)

        if debug:
            self.log(f"  Всего контуров: {len(contours)}")

        pos = self.get_position()
        near_basket = self.basket_x is not None and math.sqrt((pos[0] - self.basket_x)**2 + (pos[1] - self.basket_y)**2) < self.BASKET_EXCLUSION_RADIUS

        rings = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            # Базовая фильтрация по площади
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue

            # Рисуем ВСЕ контуры для отладки (серым)
            #cv2.drawContours(vis_frame, [cnt], -1, (128, 128, 128), 1)

            # Проверка на круглость (для колец)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * area / (perimeter ** 2)
            
            # МЯГКАЯ фильтрация по круглости
            if circularity < 0.25:
                continue

            
            
            # Центр масс
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])


            rings.append((cx, cy, area))

            if debug:
                self.log(f"    ✓ Кольцо #{len(rings)}: центр=({cx},{cy}), S={area:.0f}px, circ={circularity:.2f}")

            # Рисуем ПРИНЯТЫЕ кольца (зелёным)
            radius = int(math.sqrt(area / math.pi))
            cv2.circle(vis_frame, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(vis_frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Линия к центру
            cv2.line(vis_frame, (self.CENTER_X, self.CENTER_Y), (cx, cy), (0, 255, 0), 1)
            
            # Информация
            dx = cx - self.CENTER_X
            dy = cy - self.CENTER_Y
            label = f"dx:{dx:+d} dy:{dy:+d}"
            cv2.putText(vis_frame, label, (cx-40, cy-radius-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            try:
                world_x, world_y = pixel_to_lps(u=cx, v=cy, drone_x=pos[0], drone_y=pos[1], drone_alt=pos[2],
                                                        drone_yaw=0, image_height=self.IMG_HEIGHT, image_width=self.IMG_WIDTH)
                if self.is_unique_cargo(world_x=world_x, world_y=world_y):
                    cargo = CargoData(world_x, world_y,area)
                    self.cargo_map.append(cargo)
            except:
                pass

        # Сортировка по площади
        rings.sort(key=lambda r: r[2], reverse=True)

        # Информация на кадре
        info_y = 20
        cv2.putText(vis_frame, f"Height: {pos[2]:.2f}m", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Rings: {len(rings)}", (10, info_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Delivered: {self.delivered}", (10, info_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Показываем маску рядом
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis_frame, mask_colored])

        # Обновляем окно Detection
        with self.frame_lock:
            self.latest_processed = combined

        return rings

    def is_unique_cargo(self, world_x: float, world_y: float) -> bool:
        """Проверка уникальности груза"""
        # Проверка корзины
        if self.basket_x is not None:
            if math.sqrt((world_x - self.basket_x)**2 + (world_y - self.basket_y)**2) < self.BASKET_EXCLUSION_RADIUS:
                return False

        # Проверка дубликатов
        for cargo in self.cargo_map:
            if cargo.distance_from(world_x, world_y) < self.MIN_CARGO_DISTANCE:
                return False

        return True

    def scan_corners(self, corner):
        """Сканирование углов"""
        self.log("\n" + "="*60)
        self.log("СКАНИРОВАНИЕ УГЛОВ")
        self.log("="*60)

        corners = [ corner
        ]

        for i, (tx, ty, name) in enumerate(corners, 1):
            self.log(f"\nУгол {i}/4: {name} ({tx:.1f}, {ty:.1f})")

            if not self.quick_goto(tx, ty, self.SCAN_HEIGHT, timeout=15.0):
                self.log("  ⚠️ Таймаут")
                continue

            # Стабилизация
            self.log("  Стабилизация...")
            time.sleep(0.2)

            # Несколько снимков
            for scan_num in range(5):
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.2)
                    continue

                rings = self.detect_rings_advanced(frame, debug=(scan_num == 0))
                
                if rings:
                    pos = self.get_position()
                    
                    for cx, cy, area in rings[:3]:
                        world_x, world_y = pixel_to_lps(u=cx, v=cy, drone_x=pos[0], drone_y=pos[1], drone_alt=pos[2],
                                                        drone_yaw=0, image_height=self.IMG_HEIGHT, image_width=self.IMG_WIDTH)
                        
                        if self.is_unique_cargo(world_x, world_y):
                            cargo = CargoData(world_x, world_y, area)
                            self.cargo_map.append(cargo)
                            dist = cargo.distance_from(self.basket_x, self.basket_y)
                            self.log(f"    ✓ Груз #{len(self.cargo_map)}: ({world_x:.2f}, {world_y:.2f}), dist={dist:.2f}м")
                
                time.sleep(0.3)

        self.log(f"\n✓ Найдено {len(self.cargo_map)} грузов")

        if self.cargo_map:
            for i, cargo in enumerate(self.cargo_map, 1):
                dist = cargo.distance_from(self.basket_x, self.basket_y)
                self.log(f"  #{i}: ({cargo.world_x:.2f}, {cargo.world_y:.2f}), dist={dist:.2f}м")

    def stabilize_over_cargo(self, target_x: float, target_y: float, target_height: float,
                     max_attempts: int = 5, pixel_tolerance: int = 30) -> bool:
        """
        Улучшенная стабилизация над грузом с отладкой
        """
        self.log(f"Стабилизация над грузом на {target_height:.1f} м (до {max_attempts} итераций)...")

        for attempt in range(1, max_attempts + 1):
            # Получаем текущую позицию
            pos = self.get_position()
            self.log(f"  [{attempt}] Текущая позиция дрона: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            # Получаем кадр
            frame = self.get_frame()
            if frame is None:
                self.log(f"  [{attempt}] Нет кадра, жду...")
                time.sleep(0.2)
                continue

            # Детектируем кольца
            rings = self.detect_rings_advanced(frame, debug=(attempt == 1))
            if not rings:
                self.log(f"  [{attempt}] Кольца не найдены на кадре")
                time.sleep(0.2)
                continue

            # Находим кольцо, ближайшее к центру кадра
            best = min(rings, key=lambda r: math.hypot(r[0] - self.CENTER_X, r[1] - self.CENTER_Y))
            cx, cy, area = best
            dx_px = cx - self.CENTER_X
            dy_px = cy - self.CENTER_Y
            dist_px = math.hypot(dx_px, dy_px)

            self.log(f"  [{attempt}] Лучшее кольцо: pixel=({cx},{cy}), смещение от центра = {dist_px:.1f}px")

            # Если уже в допуске — успех
            if dist_px <= pixel_tolerance:
                self.log(f"  ✓ По центру (±{pixel_tolerance}px).")
                return True

            # КРИТИЧНО: проверяем параметры для конверсии
            self.log(f"  [{attempt}] Параметры конверсии:")
            self.log(f"    - Высота дрона: {pos[2]:.2f} м")
            self.log(f"    - Размер кадра: {self.IMG_WIDTH}x{self.IMG_HEIGHT}")
            self.log(f"    - Пиксель кольца: ({cx}, {cy})")
            
            # Конвертируем пиксель в мировые координаты
            try:
                new_x, new_y = pixel_to_lps(
                    u=cx, 
                    v=cy,
                    drone_x=pos[0], 
                    drone_y=pos[1],
                    drone_alt=pos[2],  # Текущая высота
                    drone_yaw=0.0,     # Yaw в радианах (0 = север)
                    image_width=self.IMG_WIDTH, 
                    image_height=self.IMG_HEIGHT,
                    fov_x_deg=90.0,    # FOV камеры Pioneer
                    fov_y_deg=67.5
                )
                
                self.log(f"  [{attempt}] Результат конверсии: ({new_x:.2f}, {new_y:.2f})")
                
                # Проверка на разумность результата
                delta_x = new_x - pos[0]
                delta_y = new_y - pos[1]
                correction_distance = math.hypot(delta_x, delta_y)
                
                self.log(f"  [{attempt}] Коррекция: dx={delta_x:.2f}м, dy={delta_y:.2f}м, dist={correction_distance:.2f}м")
                
                    
            except Exception as e:
                self.log(f"  [{attempt}] ❌ Ошибка конверсии: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.3)
                continue

            # Выполняем коррекцию позиции (БЕЗ изменения высоты!)
            self.log(f"  [{attempt}] Корректирую позицию: переход к ({new_x:.2f}, {new_y:.2f}) на высоте {pos[2]:.2f}м")
            
            self.quick_goto(new_x, new_y, pos[2])
            time.sleep(0.05)

        self.log("!!! Стабилизация не достигнута (превышено max_attempts)")
        return False

    def capture_cargo(self, cargo: CargoData) -> bool:
        """Улучшенная логика захвата с отладкой"""
        self.log(f"\n{'='*60}")
        self.log(f"ЗАХВАТ ГРУЗА: ({cargo.world_x:.2f}, {cargo.world_y:.2f})")
        self.log('='*60)

        # --- 1. Подлёт к грузу на высоте сканирования ---
        self.log(f"Подлёт к грузу на высоте {self.APPROACH_HEIGHT:.1f} м...")
        if not self.quick_goto(cargo.world_x, cargo.world_y, self.APPROACH_HEIGHT, timeout=15.0):
            self.log("!!! Не удалось подлететь к грузу")
            return False

        
        # --- 4. Повторное центрирование на меньшей высоте ---
        pos = self.get_position()
        self.log("ЭТАП 3: Точное центрирование на малой высоте...")
        if not self.stabilize_over_cargo(pos[0], pos[1], self.APPROACH_HEIGHT, max_attempts=10, pixel_tolerance=10):
            return False

        self.log("ЭТАП 6: Активация магнита для захвата груза...")
        self.drone.cargo_grab()
        # --- 5. Первая посадка ---
        self.log("ЭТАП 4: Первая посадка на груз...")
        self.drone.land()


        # --- 9. Взлёт с грузом ---
        self.log("ЭТАП 8: Взлёт с грузом...")
        self.drone.takeoff()
        cur_pos = self.get_position()
        if not self.quick_goto(cur_pos[0], cur_pos[1], self.TRANSPORT_HEIGHT, timeout=6.0):
            self.log("⚠️ Не удалось выйти на транспортную высоту")

        # --- 10. Полёт к корзине ---
        self.log("ЭТАП 9: Перелёт к корзине для выгрузки...")
        self.quick_goto(self.basket_x, self.basket_y, self.TRANSPORT_HEIGHT, timeout=10.0)

        # --- 11. Выгрузка ---
        self.log("ЭТАП 10: Отпускание груза...")
        self.drone.cargo_release()

        self.delivered += 1
        self.log(f"✓ Груз #{self.delivered} доставлен успешно!")

        return True
    
    def capture_all(self):
        """Захват всех грузов"""
        if not self.cargo_map:
            self.log("!!! Нет грузов")
            return

        # Сортировка по расстоянию
        self.cargo_map.sort(key=lambda c: c.distance_from(self.basket_x, self.basket_y))

        self.log(f"\nПорядок захвата ({len(self.cargo_map)} грузов):")
        for i, cargo in enumerate(self.cargo_map, 1):
            dist = cargo.distance_from(self.basket_x, self.basket_y)
            self.log(f"  #{i}: ({cargo.world_x:.2f}, {cargo.world_y:.2f}), dist={dist:.2f}м")

        for i, cargo in enumerate(self.cargo_map, 1):
            self.log(f"\n>>> ГРУЗ {i}/{len(self.cargo_map)}")

            if self.capture_cargo(cargo):
                cargo.captured = True
            else:
                self.log("⚠️ Пропускаем груз")

        self.log(f"\n✓ Обработано: {self.delivered}/{len(self.cargo_map)}")

    def run(self):
        """ГЛАВНАЯ ФУНКЦИЯ"""
        self.start_time = time.time()

        print("\n" + "="*70)
        print("ДВОЙНАЯ ВИЗУАЛИЗАЦИЯ + ТОЧНАЯ ДЕТЕКЦИЯ")
        print("="*70 + "\n")

        viz_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        viz_thread.start()

        try:
            self.log("Взлёт... ")
            self.drone.arm()
            time.sleep(1.0)
            self.drone.takeoff()

            pos = self.get_position()
            self.basket_x = pos[0]
            self.basket_y = pos[1]

            self.log(f"Корзина: ({self.basket_x:.2f}, {self.basket_y:.2f})")

            if not self.calibrate_camera():
                self.log("!!! Ошибка камеры")
                self.drone.land()
                return

            self.log(f"Подъём на {self.SCAN_HEIGHT}м...")
            self.quick_goto(self.basket_x, self.basket_y, self.SCAN_HEIGHT, timeout=6.0)
            time.sleep(1.0)

            
            corners = [
                (self.basket_x, self.basket_y, "CUR"),
                (self.basket_x - self.RECON_RADIUS, self.basket_y - self.RECON_RADIUS, "ЮЗ"),
                (self.basket_x + self.RECON_RADIUS, self.basket_y - self.RECON_RADIUS, "ЮВ"),
                (self.basket_x + self.RECON_RADIUS, self.basket_y + self.RECON_RADIUS, "СВ"),
                (self.basket_x - self.RECON_RADIUS, self.basket_y + self.RECON_RADIUS, "СЗ"),
            ]
            for i in corners:
                # Сканирование
                self.scan_corners(i)

                # Захват
                self.capture_all()

            elapsed = time.time() - self.start_time
            print(f"\n{'='*70}")
            print(f"ЗАВЕРШЕНО за {elapsed:.1f}с | Доставлено: {self.delivered}/{len(self.cargo_map)}")
            print("="*70 + "\n")

            # Возврат
            self.quick_goto(self.basket_x, self.basket_y, self.SCAN_HEIGHT, timeout=10.0)
            self.drone.land()
            time.sleep(4.0)
            self.drone.disarm()

        except KeyboardInterrupt:
            self.log("\n!!! СТОП")
            try:
                self.drone.cargo_release()
                self.drone.land()
            except:
                pass

        except Exception as e:
            self.log(f"\n!!! ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.drone.cargo_release()
                self.drone.land()
            except:
                pass

        finally:
            self.visualization_active = False
            time.sleep(0.5)
            cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 4:
        print("Использование: python main.py <IP> <drone_port> <camera_port>")
        print("Пример: python main.py 127.0.0.1 8000 18000")
        sys.exit(1)

    ip = sys.argv[1]
    drone_port = int(sys.argv[2])
    camera_port = int(sys.argv[3])

    print(f"[ПОДКЛЮЧЕНИЕ] IP={ip}, Drone={drone_port}, Camera={camera_port}")

    drone = Pioneer(ip=ip, mavlink_port=drone_port, simulator=True)
    camera = Camera(ip=ip, port=camera_port)

    hunter = CargoHunter(drone, camera)
    hunter.run()


if __name__ == "__main__":
    main()