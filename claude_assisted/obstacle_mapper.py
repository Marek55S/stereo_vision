import math
import numpy as np
from pymavlink import mavutil
import time

class ObstacleMapper:
    def __init__(self, mavlink_comm, image_width=640, image_height=480):
        """
        Klasa do tworzenia mapy przeszkód z depth map

        Args:
            mavlink_comm: Instancja MAVLinkCommunication
            image_width: Szerokość obrazu
            image_height: Wysokość obrazu
        """
        self.mavlink_comm = mavlink_comm
        self.image_width = image_width
        self.image_height = image_height

        # Parametry dla wiadomości OBSTACLE_DISTANCE (72 sektory po 5°)
        self.num_sectors = 72
        self.sector_angle = 360.0 / self.num_sectors  # 5 stopni na sektor

        # Parametry dla DISTANCE_SENSOR (8 kierunków)
        self.distance_orientations = [
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_0,  # Przód
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_45,  # Przód-prawo
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_90,  # Prawo
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_135,  # Tył-prawo
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_180,  # Tył
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_225,  # Tył-lewo
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_270,  # Lewo
            mavutil.mavlink.MAV_SENSOR_ROTATION_YAW_315,  # Przód-lewo
        ]

        self.last_heartbeat = time.time()

    def depth_to_obstacle_map(self, depth_map, max_range_m=8.0):
        """
        Konwertuj depth map na mapę przeszkód 360°

        Args:
            depth_map: Mapa głębi w metrach
            max_range_m: Maksymalny zasięg w metrach

        Returns:
            distances_array: Tablica 72 elementów z odległościami w cm
        """
        height, width = depth_map.shape
        center_x, center_y = width // 2, height // 2

        # Inicjalizuj tablicę odległości
        distances_cm = np.full(self.num_sectors, 65535, dtype=np.uint16)  # 65535 = brak przeszkody

        # Dla każdego sektora znajdź minimalną odległość
        for sector in range(self.num_sectors):
            angle_start = sector * self.sector_angle
            angle_end = (sector + 1) * self.sector_angle

            # Pobierz piksele w tym sektorze
            sector_distances = self._get_sector_distances(
                depth_map, center_x, center_y, angle_start, angle_end, max_range_m
            )

            if len(sector_distances) > 0:
                # Weź minimalną odległość w sektorze (najbliższa przeszkoda)
                min_distance_m = np.min(sector_distances)
                if min_distance_m < max_range_m and min_distance_m > 0.1:  # Ignoruj bardzo bliskie pomiary
                    distances_cm[sector] = int(min_distance_m * 100)  # Konwersja na cm

        return distances_cm

    def _get_sector_distances(self, depth_map, center_x, center_y, angle_start, angle_end, max_range_m):
        """Pobierz odległości z określonego sektora kątowego"""
        height, width = depth_map.shape
        distances = []

        # Promień do skanowania (od centrum do krawędzi obrazu)
        max_radius = min(center_x, center_y, width - center_x, height - center_y)

        # Skanuj linie w sektorze
        num_rays = 5  # Liczba promieni na sektor
        for i in range(num_rays):
            angle = math.radians(angle_start + i * (angle_end - angle_start) / num_rays)

            # Skanuj wzdłuż promienia
            for r in range(10, max_radius, 5):  # Co 5 pikseli
                x = int(center_x + r * math.cos(angle))
                y = int(center_y + r * math.sin(angle))

                if 0 <= x < width and 0 <= y < height:
                    depth = depth_map[y, x]
                    if 0.1 < depth < max_range_m:  # Filtruj nieprawidłowe wartości
                        distances.append(depth)

        return distances

    def get_directional_distances(self, depth_map, max_range_m=8.0):
        """
        Pobierz odległości w 8 głównych kierunkach dla DISTANCE_SENSOR

        Returns:
            distances_8dir: Lista odległości w 8 kierunkach (cm)
        """
        height, width = depth_map.shape
        center_x, center_y = width // 2, height // 2

        distances_8dir = []
        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

        for angle_deg in angles_deg:
            angle_rad = math.radians(angle_deg)

            # Pobierz średnią odległość w kierunku (±10°)
            sector_distances = self._get_sector_distances(
                depth_map, center_x, center_y, angle_deg - 10, angle_deg + 10, max_range_m
            )

            if len(sector_distances) > 0:
                # Użyj mediany dla stabilności
                distance_m = np.median(sector_distances)
                distances_8dir.append(int(distance_m * 100))  # cm
            else:
                distances_8dir.append(800)  # Brak przeszkody - maksymalna odległość

        return distances_8dir

    def send_obstacle_data(self, depth_map):
        """
        Wyślij dane o przeszkodach do ArduPilot

        Args:
            depth_map: Mapa głębi w metrach
        """
        # Metoda 1: Wyślij pełną mapę 360° (OBSTACLE_DISTANCE)
        obstacle_distances = self.depth_to_obstacle_map(depth_map)
        self.mavlink_comm.send_obstacle_distance(obstacle_distances)

        # Metoda 2: Wyślij dane kierunkowe (DISTANCE_SENSOR) - alternatywnie
        # directional_distances = self.get_directional_distances(depth_map)
        # for i, distance_cm in enumerate(directional_distances):
        #     self.mavlink_comm.send_distance_sensor(
        #         distance_cm,
        #         self.distance_orientations[i]
        #     )

        # Wyślij heartbeat co sekundę
        current_time = time.time()
        if current_time - self.last_heartbeat > 1.0:
            self.mavlink_comm.send_heartbeat()
            self.last_heartbeat = current_time
