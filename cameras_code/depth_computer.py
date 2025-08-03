import cv2
import numpy as np
from time import sleep
from stereo_pair import StereoCamera
from disparity_computer import DisparityComputer



# focal length 3.04mm
# px size 1.12um = 0.00112mm
# focal length in pixels = 3040 / 1.12 = 2714.285714285714

class DepthComputer:
    def __init__(self,focal_length_px,baseline_m):
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m

    def compute_depth(self, disparity_map):
        # Unikaj dzielenia przez 0 i bardzo małe wartości
        disparity_map[disparity_map <= 0.0] = 0.1
        depth_map = (self.focal_length_px * self.baseline_m) / disparity_map
        return depth_map

    def normalize(self, depth_map):
        """
        Przeskaluj mapę głębi do obrazu 8-bitowego do wyświetlenia (0-255)
        :param max_depth: maksymalna głębokość w metrach (do skalowania)
        """
        # depth_map_clip = np.clip(depth_map, 0, max_depth)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        return depth_vis.astype(np.uint8)

    def show_depth(self, depth_map):
        depth_vis = self.normalize(depth_map)
        cv2.imshow("Depth Map", depth_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parametry kamery
    focal_length_px = 2714.29  # przeliczone z mm / pixel_size_mm
    baseline_m = 0.06  # np. 6 cm, podaj zgodnie z rzeczywistym setupem

    # Inicjalizacja klas
    disparity_computer = DisparityComputer()
    depth_computer = DepthComputer(focal_length_px, baseline_m)
    stereo = StereoCamera()

    # Uruchom stereo kamerę
    stereo.start()
    sleep(1)  # czas na rozgrzanie kamery

    try:
        while True:
            left, right = stereo.get_frames()
            if left is None or right is None:
                continue

            disparity = disparity_computer.compute(left, right)
            depth = depth_computer.compute_depth(disparity)

            # Normalizacja do obrazu 8-bit
            disp_vis = disparity_computer.normalize(disparity)
            depth_vis = depth_computer.normalize(depth)

            # Pokazanie obu map
            cv2.imshow("Disparity Map", disp_vis)
            cv2.imshow("Depth Map", depth_vis)
            cv2.imshow("Left Camera", left)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Zatrzymano przez użytkownika.")
    finally:
        stereo.stop()
        cv2.destroyAllWindows()