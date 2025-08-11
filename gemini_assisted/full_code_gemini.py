# Wymagane biblioteki:
# pip install picamera2 opencv-python numpy pymavlink

import threading
import time
import cv2
import numpy as np
import sys
import os

# Sprawdź, czy pymavlink jest dostępny
try:
    from pymavlink import mavutil
except ImportError:
    print("Błąd: Biblioteka pymavlink nie jest zainstalowana. Uruchom 'pip install pymavlink'")
    sys.exit(1)

# Sprawdź, czy picamera2 jest dostępna
try:
    from picamera2 import Picamera2
except ImportError:
    print("Błąd: Biblioteka picamera2 nie jest zainstalowana. Uruchom 'pip install picamera2'")
    sys.exit(1)


# --- KLASY DO OBLICZEŃ STEREO ---

class StereoCamera:
    """
    Klasa do pobierania klatek z dwóch kamer Picamera2 w osobnych wątkach.
    """

    def __init__(self, cam_id_left=0, cam_id_right=1, resolution=(640, 480)):
        self.running = False
        self.left_frame = None
        self.right_frame = None
        self.resolution = resolution

        # Inicjalizacja dwóch instancji Picamera2
        print(f"Inicjalizacja kamery lewej (ID: {cam_id_left})...")
        self.cam_left = Picamera2(cam_id_left)
        config_left = self.cam_left.create_still_configuration(main={"size": resolution, "format": "RGB888"})
        self.cam_left.configure(config_left)

        print(f"Inicjalizacja kamery prawej (ID: {cam_id_right})...")
        self.cam_right = Picamera2(cam_id_right)
        config_right = self.cam_right.create_still_configuration(main={"size": resolution, "format": "RGB888"})
        self.cam_right.configure(config_right)

    def start(self):
        self.running = True
        self.cam_left.start()
        self.cam_right.start()

        self.thread_left = threading.Thread(target=self._update_left, daemon=True)
        self.thread_right = threading.Thread(target=self._update_right, daemon=True)

        self.thread_left.start()
        self.thread_right.start()
        print("Wątki kamer uruchomione.")

    def _update_left(self):
        while self.running:
            self.left_frame = self.cam_left.capture_array("main")
            # Dodano konwersję kolorów, aby pasowały do OpenCV
            self.left_frame = cv2.cvtColor(self.left_frame, cv2.COLOR_RGB2BGR)
            time.sleep(0.01)

    def _update_right(self):
        while self.running:
            self.right_frame = self.cam_right.capture_array("main")
            self.right_frame = cv2.cvtColor(self.right_frame, cv2.COLOR_RGB2BGR)
            time.sleep(0.01)

    def get_frames(self):
        return self.left_frame, self.right_frame

    def stop(self):
        self.running = False
        if self.thread_left.is_alive():
            self.thread_left.join()
        if self.thread_right.is_alive():
            self.thread_right.join()
        self.cam_left.stop()
        self.cam_right.stop()
        print("Wątki kamer zatrzymane.")


class DisparityComputer:
    """
    Klasa do obliczania mapy dysparycji z dwóch obrazów stereo.
    """

    def __init__(self):
        # Parametry StereoSGBM. Wymagają dostrojenia dla najlepszych rezultatów.
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # Musi być wielokrotnością 16
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def compute(self, left_frame, right_frame):
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Oblicz mapę dysparycji
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        return disparity


class DepthComputer:
    """
    Klasa do obliczania mapy głębi z mapy dysparycji.
    """

    def __init__(self, focal_length_px, baseline_m):
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m

    def compute_depth(self, disparity_map):
        # Unikaj dzielenia przez 0 i bardzo małe wartości.
        # Wartości dysparycji <= 0 są nieprawidłowe.
        disparity_map[disparity_map <= 0.0] = np.nan
        depth_map = (self.focal_length_px * self.baseline_m) / disparity_map
        return depth_map


# --- GŁÓWNA KLASA APLIKACJI ---

class StereoObstacleAvoidance:
    """
    Główna klasa aplikacji, która łączy wszystkie funkcje.
    """

    def __init__(self, mavlink_connection_string, calibration_file="stereo_calibration.npz"):
        self.mavlink_connection_string = mavlink_connection_string
        self.calibration_file = calibration_file

        # --- USTAWIENIA PARAMETRÓW KAMERY I MAVLINKA ---
        self.FOCAL_LENGTH_PX = 2714.29  # Wartość PRZYKŁADOWA - zostanie zastąpiona kalibracją
        self.BASELINE_M = 0.06  # Wartość PRZYKŁADOWA - zostanie zastąpiona kalibracją
        self.HORIZONTAL_FOV_DEG = 62.2  # Horyzontalne pole widzenia w stopniach
        self.IMAGE_WIDTH = 640
        self.IMAGE_HEIGHT = 480

        # Bezpieczna odległość w metrach, powyżej której uznajemy przeszkodę za nieistotną
        self.MAX_DISTANCE_M = 15.0
        # Minimalna odległość w metrach, poniżej której uznajemy przeszkodę za bardzo bliską
        self.MIN_DISTANCE_M = 0.5

        # Inicjalizacja zmiennych kalibracyjnych
        self.map1_left, self.map2_left = None, None
        self.map1_right, self.map2_right = None, None
        self.Q_matrix = None

        # Inicjalizacja komponentów
        self.stereo_cam = StereoCamera(resolution=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        self.disparity_comp = DisparityComputer()
        self.depth_comp = DepthComputer(self.FOCAL_LENGTH_PX,
                                        self.BASELINE_M)  # Tymczasowo, zostanie zaktualizowana po wczytaniu kalibracji

        self.mav_connection = None

        self.is_running = False

    def load_calibration_data(self):
        """
        Wczytuje macierze kalibracyjne z pliku.
        """
        if not os.path.exists(self.calibration_file):
            print(
                f"Błąd: Nie znaleziono pliku kalibracji '{self.calibration_file}'. Używane będą wartości domyślne, co może prowadzić do niedokładnych pomiarów.")
            # Używamy domyślnych wartości
            return

        try:
            calib_data = np.load(self.calibration_file)
            self.map1_left = calib_data['map1_left']
            self.map2_left = calib_data['map2_left']
            self.map1_right = calib_data['map1_right']
            self.map2_right = calib_data['map2_right']
            self.Q_matrix = calib_data['Q']

            # Ekstrakcja danych z macierzy Q dla obliczeń głębi
            focal_length = self.Q_matrix[2, 3]
            baseline_focal = -1 / self.Q_matrix[3, 2]  # (1 / Q[3,2])

            # Aktualizacja obiektu DepthComputer o skalibrowane wartości
            self.depth_comp.focal_length_px = focal_length
            self.depth_comp.baseline_m = baseline_focal / focal_length

            print(f"Dane kalibracyjne wczytane pomyślnie z '{self.calibration_file}'.")
            print(f"Skalibrowana ogniskowa (px): {self.depth_comp.focal_length_px:.2f}")
            print(f"Skalibrowana linia bazowa (m): {self.depth_comp.baseline_m:.4f}")

        except Exception as e:
            print(f"Błąd podczas wczytywania pliku kalibracji: {e}")
            print("Używane będą wartości domyślne.")

    def connect_mavlink(self):
        """
        Inicjalizacja połączenia MAVLink.
        """
        print(f"Łączenie z ArduPilot przez: {self.mavlink_connection_string}...")
        self.mav_connection = mavutil.mavlink_connection(self.mavlink_connection_string)
        self.mav_connection.wait_heartbeat()
        print("Połączono z ArduPilot! Otrzymano Heartbeat.")

    def send_obstacle_distance(self, depth_map):
        """
        Przetwarza mapę głębi i wysyła ją jako wiadomość MAVLink OBSTACLE_DISTANCE.
        """
        # Sprawdź, czy połączenie MAVLink istnieje
        if not self.mav_connection:
            print("Błąd: Brak połączenia MAVLink. Pomiń wysyłanie danych.")
            return

        # Przygotowanie danych do wysłania: 72 odległości (0-359 stopni)
        distances = np.full(72, 30000, dtype=np.uint16)  # Domyślnie bezpieczna odległość 30m w mm

        # Obliczenie kąta środkowego każdego słupka pikseli (kolumny)
        # 72 sektory pokrywają 360 stopni. Obliczymy odległości tylko dla obszaru widzenia kamery.
        # Przykładowo, HFOV 62.2 stopnia.
        # 62.2 stopnia / 360 stopni * 72 sektory = ~12.44 sektora
        # Użyjemy np. 12 sektorów centralnie.

        # Obliczenie szerokości jednego sektora w pikselach
        sector_angle = 360 / 72
        fov_start_sector = int((360 - self.HORIZONTAL_FOV_DEG) / 2 / sector_angle)
        fov_end_sector = int((360 + self.HORIZONTAL_FOV_DEG) / 2 / sector_angle)

        # Wyrównanie do rozmiaru mapy głębi
        step = int(self.IMAGE_WIDTH / (fov_end_sector - fov_start_sector))

        for i in range(fov_start_sector, fov_end_sector):
            col_start = int((i - fov_start_sector) * step)
            col_end = int(col_start + step)

            # Wycinek mapy głębi dla danego sektora
            sector_depths = depth_map[:, col_start:col_end]

            # Usuń nieprawidłowe wartości (nan, inf)
            valid_depths = sector_depths[~np.isnan(sector_depths) & ~np.isinf(sector_depths) & (sector_depths > 0)]

            if valid_depths.size > 0:
                min_depth = np.min(valid_depths)
                # Ogranicz odległość do zakresu MAVLink (0-30000mm)
                min_depth = np.clip(min_depth * 1000, 0, 30000)
                distances[i] = int(min_depth)

        # Wysłanie wiadomości MAVLink
        time_usec = int(time.time() * 1000000)
        self.mav_connection.mav.obstacle_distance_send(
            time_usec,
            0,  # sensor_type: 0-ultrasound, 1-lidar, 2-infrared, 3-stereo
            distances,
            0,  # increment: kąt w stopniach (nieużywane dla 72 sektorów)
            0,  # min_distance: 0mm
            30000,  # max_distance: 30m
            0,  # sensor_id: 0
            mavutil.mavlink.MAV_DISTANCE_SENSOR_GENERIC  # type: 16
        )

    def run(self):
        """
        Główna pętla aplikacji.
        """
        self.is_running = True
        try:
            self.load_calibration_data()
            self.connect_mavlink()
            self.stereo_cam.start()

            # Czas na rozgrzanie kamery i wątków
            time.sleep(2)

            print("Rozpoczynanie pętli przetwarzania...")
            while self.is_running:
                left, right = self.stereo_cam.get_frames()
                if left is None or right is None:
                    continue

                # Rektyfikacja obrazów (wyrównanie po kalibracji)
                if self.map1_left is not None:
                    left_rectified = cv2.remap(left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
                    right_rectified = cv2.remap(right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
                else:
                    # Jeśli kalibracja nie została wczytana, używamy obrazów bez rektyfikacji
                    left_rectified = left
                    right_rectified = right

                # Obliczenie mapy dysparycji
                disparity = self.disparity_comp.compute(left_rectified, right_rectified)

                # Obliczenie mapy głębi
                depth = self.depth_comp.compute_depth(disparity)

                # Wysłanie danych do ArduPilota
                self.send_obstacle_distance(depth)

                # Normalizacja do obrazu 8-bit do wizualizacji
                disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Wyświetlanie na ekranie (opcjonalne, może obciążać Raspberry Pi)
                cv2.imshow("Disparity Map", disp_vis)
                cv2.imshow("Depth Map", depth_vis)
                cv2.imshow("Left Camera", left_rectified)

                # Zakończenie działania po naciśnięciu 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

        except Exception as e:
            print(f"Wystąpił błąd: {e}")
        finally:
            self.stop()

    def stop(self):
        """
        Zatrzymuje aplikację i zwalnia zasoby.
        """
        self.is_running = False
        self.stereo_cam.stop()
        cv2.destroyAllWindows()
        print("Aplikacja zatrzymana.")


if __name__ == "__main__":
    # Parametry połączenia MAVLink
    # 'udp:127.0.0.1:14550' dla SITL lub 'serial:/dev/ttyACM0:57600' dla fizycznego
    mavlink_connection_string = 'udp:127.0.0.1:14550'

    avoidance_app = StereoObstacleAvoidance(mavlink_connection_string)
    avoidance_app.run()
