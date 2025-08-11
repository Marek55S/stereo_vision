import threading
import time
import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    print("Błąd: Biblioteka picamera2 nie jest zainstalowana. Uruchom 'pip install picamera2'")
    exit()

class StereoCamera:
    def __init__(self):
        self.running = False
        self.left_frame = None
        self.right_frame = None

        # Inicjalizacja dwóch instancji Picamera2
        self.cam_left = Picamera2(0)  # Kamera 0
        self.cam_right = Picamera2(1)  # Kamera 1

        # Konfiguracja trybu podglądu (preview)
        config_left = self.cam_left.create_still_configuration(main={"size": (640, 480)})
        config_right = self.cam_right.create_still_configuration(main={"size": (640, 480)})

        self.cam_left.configure(config_left)
        self.cam_right.configure(config_right)

    def start(self):
        self.running = True

        self.cam_left.start()
        self.cam_right.start()

        # Wątki do pobierania obrazu
        self.thread_left = threading.Thread(target=self._update_left, daemon=True)
        self.thread_right = threading.Thread(target=self._update_right, daemon=True)

        self.thread_left.start()
        self.thread_right.start()

    def _update_left(self):
        while self.running:
            self.left_frame = self.cam_left.capture_array("main")
            time.sleep(0.01)

    def _update_right(self):
        while self.running:
            self.right_frame = self.cam_right.capture_array("main")
            time.sleep(0.01)

    def get_frames(self):
        return self.left_frame, self.right_frame

    def stop(self):
        self.running = False
        self.thread_left.join()
        self.thread_right.join()
        self.cam_left.stop()
        self.cam_right.stop()

    def show_frames(self):
        while self.running:
            left, right = self.get_frames()
            if left is not None and right is not None:
                combined = np.hstack((left, right))
                cv2.imshow("Stereo Camera", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stereo = StereoCamera()
    stereo.start()
    stereo.show_frames()
