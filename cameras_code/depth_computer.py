import cv2
import numpy as np


class DepthComputer:
    def __init__(self,focal_length_px,baseline_m):
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m

    def compute_depth(self, disparity_map):
        # Unikaj dzielenia przez 0 i bardzo małe wartości
        disparity_map[disparity_map <= 0.0] = 0.1
        depth_map = (self.f * self.B) / disparity_map
        return depth_map

    def normalize(self, depth_map, max_depth=2.0):
        """
        Przeskaluj mapę głębi do obrazu 8-bitowego do wyświetlenia (0-255)
        :param max_depth: maksymalna głębokość w metrach (do skalowania)
        """
        depth_map_clip = np.clip(depth_map, 0, max_depth)
        depth_vis = cv2.normalize(depth_map_clip, None, 0, 255, cv2.NORM_MINMAX)
        return depth_vis.astype(np.uint8)

    def show_depth(self, depth_map):
        depth_vis = self.normalize(depth_map)
        cv2.imshow("Depth Map", depth_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
