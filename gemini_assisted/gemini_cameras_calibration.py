# Wymagane biblioteki:
# pip install opencv-python numpy picamera2

import numpy as np
import cv2
import glob
import os
import time

# Sprawdź, czy picamera2 jest dostępna
try:
    from picamera2 import Picamera2
except ImportError:
    print("Błąd: Biblioteka picamera2 nie jest zainstalowana. Uruchom 'pip install picamera2'")
    exit()

# --- USTAWIENIA KALIBRACJI ---
# Rozmiar szachownicy
CHECKERBOARD = (6, 9)
# Rozmiar kwadratu na szachownicy w metrach (np. 0.025 m dla 2.5 cm)
SQUARE_SIZE = 0.025

# Rozdzielczość kamer
RESOLUTION = (640, 480)

# Nazwy folderów do zapisu obrazów
IMAGE_DIR = "calibration_images"
LEFT_DIR = os.path.join(IMAGE_DIR, "left")
RIGHT_DIR = os.path.join(IMAGE_DIR, "right")

# Nazwy plików, w których zostaną zapisane wyniki kalibracji
CALIBRATION_FILE = "stereo_calibration.npz"

# Minimalna liczba obrazów do kalibracji
MIN_IMAGES = 15

# Kryteria zakończenia algorytmu kalibracji (używane przez OpenCV)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def initialize_cameras():
    """
    Inicjalizuje obie kamery Picamera2.
    """
    print("Inicjalizacja kamer...")
    cam_left = Picamera2(0)
    config_left = cam_left.create_still_configuration(main={"size": RESOLUTION, "format": "RGB888"})
    cam_left.configure(config_left)

    cam_right = Picamera2(1)
    config_right = cam_right.create_still_configuration(main={"size": RESOLUTION, "format": "RGB888"})
    cam_right.configure(config_right)

    cam_left.start()
    cam_right.start()
    print("Kamery gotowe.")

    # Czas na rozgrzewkę
    time.sleep(2)
    return cam_left, cam_right

def capture_images(cam_left, cam_right):
    """
    Zapisuje obrazy kalibracyjne.
    """
    os.makedirs(LEFT_DIR, exist_ok=True)
    os.makedirs(RIGHT_DIR, exist_ok=True)

    img_count = 0
    print(f"Zapisywanie obrazów kalibracyjnych. Naciśnij 's' aby zrobić zdjęcie, 'q' aby zakończyć.")
    print("Pamiętaj, aby zrobić zdjęcia planszy z różnych kątów i odległości!")

    while True:
        frame_left = cam_left.capture_array("main")
        frame_right = cam_right.capture_array("main")

        # Konwersja do BGR dla OpenCV
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

        # Wyświetlanie na ekranie
        cv2.imshow("Left Camera - Press 's' to save", frame_left)
        cv2.imshow("Right Camera", frame_right)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print(f"Zapisywanie obrazu {img_count+1}...")
            left_path = os.path.join(LEFT_DIR, f"left_{img_count:02}.png")
            right_path = os.path.join(RIGHT_DIR, f"right_{img_count:02}.png")
            cv2.imwrite(left_path, frame_left)
            cv2.imwrite(right_path, frame_right)
            img_count += 1
            time.sleep(1) # Unikamy zapisywania identycznych klatek

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cam_left.stop()
    cam_right.stop()
    print(f"Zapisano {img_count} par obrazów kalibracyjnych.")
    return img_count

def calibrate(img_dir_left, img_dir_right):
    """
    Główna funkcja kalibracji.
    """
    # Tablice do przechowywania punktów
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    # Przygotowanie punktów 3D szachownicy
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    images_left = sorted(glob.glob(os.path.join(img_dir_left, '*.png')))
    images_right = sorted(glob.glob(os.path.join(img_dir_right, '*.png')))

    if len(images_left) < MIN_IMAGES or len(images_right) < MIN_IMAGES:
        print(f"Błąd: Zbyt mało obrazów do kalibracji. Potrzebne minimum {MIN_IMAGES}, znaleziono {len(images_left)}.")
        return

    print("Rozpoczęcie detekcji punktów...")
    for i in range(len(images_left)):
        img_left = cv2.imread(images_left[i])
        img_right = cv2.imread(images_right[i])

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Znajdź rogi szachownicy
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

        if ret_left and ret_right:
            print(f"Punkty znalezione na parze obrazów {i+1}/{len(images_left)}")

            objpoints.append(objp)

            # Subpikselowa dokładność dla punktów
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), CRITERIA)
            imgpoints_left.append(corners2_left)

            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), CRITERIA)
            imgpoints_right.append(corners2_right)

            # Opcjonalnie: wizualizacja detekcji
            cv2.drawChessboardCorners(img_left, CHECKERBOARD, corners2_left, ret_left)
            cv2.drawChessboardCorners(img_right, CHECKERBOARD, corners2_right, ret_right)
            cv2.imshow('Left', img_left)
            cv2.imshow('Right', img_right)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # --- Kalibracja monokularowa dla każdej kamery ---
    print("Kalibracja monokularowa...")
    ret_left, K_left, D_left, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    ret_right, K_right, D_right, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

    print("Kalibracja stereo...")
    # Kalibracja stereo
    ret_stereo, K_left_rect, D_left_rect, K_right_rect, D_right_rect, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, K_left, D_left, K_right, D_right, gray_left.shape[::-1],
        criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC
    )

    print("Rectyfikacja stereo...")
    R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
        K_left_rect, D_left_rect, K_right_rect, D_right_rect, gray_left.shape[::-1], R, T, alpha=0
    )

    # Obliczenie map rektyfikacji
    map1_left, map2_left = cv2.initUndistortRectifyMap(K_left_rect, D_left_rect, R_left, P_left, gray_left.shape[::-1], cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K_right_rect, D_right_rect, R_right, P_right, gray_right.shape[::-1], cv2.CV_16SC2)

    # Zapisz wyniki kalibracji
    np.savez(CALIBRATION_FILE,
             K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right,
             R=R, T=T, E=E, F=F, Q=Q,
             P_left=P_left, P_right=P_right,
             map1_left=map1_left, map2_left=map2_left,
             map1_right=map1_right, map2_right=map2_right)

    print(f"Kalibracja zakończona. Wyniki zapisano w pliku '{CALIBRATION_FILE}'.")
    print(f"Błąd reprojekcji: {ret_stereo}")

    return Q

if __name__ == '__main__':
    # 1. Zapisywanie obrazów
    cam_left, cam_right = initialize_cameras()
    img_count = capture_images(cam_left, cam_right)

    if img_count >= MIN_IMAGES:
        # 2. Kalibracja i zapis danych
        Q_matrix = calibrate(LEFT_DIR, RIGHT_DIR)

        # 3. Dodatkowa weryfikacja
        print("\nPrzeprowadzanie weryfikacji kalibracji...")
        calib_data = np.load(CALIBRATION_FILE)
        map1_left = calib_data['map1_left']
        map2_left = calib_data['map2_left']
        map1_right = calib_data['map1_right']
        map2_right = calib_data['map2_right']

        # Wczytaj przykładowy obraz i go wyprostuj
        img_left_orig = cv2.imread(os.path.join(LEFT_DIR, f"left_00.png"))
        img_right_orig = cv2.imread(os.path.join(RIGHT_DIR, f"right_00.png"))

        img_left_rectified = cv2.remap(img_left_orig, map1_left, map2_left, cv2.INTER_LINEAR)
        img_right_rectified = cv2.remap(img_right_orig, map1_right, map2_right, cv2.INTER_LINEAR)

        # Wyświetl obrazki obok siebie
        h, w = img_left_rectified.shape[:2]
        vis = np.zeros((h, w*2, 3), np.uint8)
        vis[:, :w] = img_left_rectified
        vis[:, w:] = img_right_rectified

        for i in range(20):
            cv2.line(vis, (0, i * h // 20), (w*2, i * h // 20), (0, 255, 0), 1)

        cv2.imshow('Rectified and Aligned Images (Green lines should be straight)', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nie zebrano wystarczającej liczby obrazów. Kalibracja nie została przeprowadzona.")
