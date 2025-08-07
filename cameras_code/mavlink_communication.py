import time
import math
import numpy as np
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink2


class MAVLinkCommunication:
    def __init__(self, connection_string='/dev/ttyACM0', baud=57600):
        """
        Inicjalizacja połączenia MAVLink z ArduPilot

        Args:
            connection_string: Ścieżka do urządzenia (dla Pi zwykle /dev/ttyACM0 lub /dev/serial0)
            baud: Prędkość transmisji
        """
        self.connection = mavutil.mavlink_connection(connection_string, baud=baud)
        self.system_id = 1  # ID systemu (komputer towarzyszący)
        self.component_id = mavutil.mavlink.MAV_COMP_ID_PATHPLANNER  # 195

        # Czekaj na heartbeat z ArduPilot
        print("Oczekiwanie na heartbeat z ArduPilot...")
        self.connection.wait_heartbeat()
        print(
            f"Połączono z systemem ID: {self.connection.target_system}, komponent: {self.connection.target_component}")

        # Wyślij własny heartbeat
        self.send_heartbeat()

    def send_heartbeat(self):
        """Wyślij heartbeat do ArduPilot"""
        self.connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,  # typ
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,  # autopilot type
            0,  # base mode
            0,  # custom mode
            mavutil.mavlink.MAV_STATE_ACTIVE  # system status
        )

    def send_distance_sensor(self, distance_cm, orientation, min_distance=10, max_distance=800):
        """
        Wyślij wiadomość DISTANCE_SENSOR

        Args:
            distance_cm: Odległość w cm
            orientation: Orientacja sensora (MAV_SENSOR_ROTATION)
            min_distance: Minimalna odległość sensora w cm
            max_distance: Maksymalna odległość sensora w cm
        """
        timestamp = int(time.time() * 1000) % (2 ** 32)  # timestamp w ms

        self.connection.mav.distance_sensor_send(
            timestamp,
            min_distance,
            max_distance,
            distance_cm,
            mavutil.mavlink.MAV_DISTANCE_SENSOR_UNKNOWN,  # typ sensora
            0,  # ID sensora
            orientation,  # orientacja
            255,  # covariance (255 = nieznana)
            0,  # horizontal_fov (0 = nieznane)
            0  # vertical_fov (0 = nieznane)
        )

    def send_obstacle_distance(self, distances_array, increment=5):
        """
        Wyślij wiadomość OBSTACLE_DISTANCE (360° obstacle data)

        Args:
            distances_array: Tablica 72 elementów z odległościami w cm
            increment: Krok kątowy między pomiarami w stopniach
        """
        timestamp = int(time.time() * 1000) % (2 ** 32)

        # Konwertuj float array na uint16 array (wymagane przez MAVLink)
        distances_uint16 = np.array(distances_array, dtype=np.uint16)

        # Utwórz wiadomość MAVLink
        msg = mavlink2.MAVLink_obstacle_distance_message(
            timestamp,
            mavutil.mavlink.MAV_DISTANCE_SENSOR_UNKNOWN,  # sensor_type
            distances_uint16.tolist(),  # distances (72 elementy)
            increment,  # angular_width_deg
            10,  # min_distance cm
            65535,  # max_distance cm (65535 = unknown)
            0,  # angle_offset degrees
            0  # frame (MAV_FRAME_GLOBAL = 0)
        )

        self.connection.send(msg)