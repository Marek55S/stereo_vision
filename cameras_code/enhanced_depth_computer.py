import cv2
import numpy as np
from time import sleep
from stereo_pair import StereoCamera
from disparity_computer import DisparityComputer


class EnhancedDepthComputer:
    def __init__(self, focal_length_px, baseline_m):
        self.focal_length_px = focal_length_px
        self.baseline_m = baseline_m

    def compute_depth(self, disparity_map):
        """Oblicz mapę głębi z mapy dysparycji"""
        # Unikaj dzielenia przez 0
        disparity_map[disparity_map <= 0.0] = 0.1
        depth_map = (self.focal_length_px * self.baseline_m) / disparity_map

        # Filtruj nieprawidłowe wartości
        depth_map[depth_map < 0.1] = 0.0  # Za blisko
        depth_map[depth_map > 20.0] = 0.0  # Za daleko

        return depth_map

    def apply_filters(self, depth_map):
        """Zastosuj filtry do wygładzenia mapy głębi"""
        # Filtr medianowy do usunięcia szumu
        filtered = cv2.medianBlur(depth_map.astype(np.float32), 5)

        # Filtr Gaussowski do wygładzenia
        filtered = cv2.GaussianBlur(filtered, (3, 3), 0)

        return filtered

    def normalize_for_display(self, depth_map, max_depth=10.0):
        """Normalizuj mapę głębi do wyświetlenia"""
        depth_clip = np.clip(depth_map, 0, max_depth)
        depth_vis = cv2.normalize(depth_clip, None, 0, 255, cv2.NORM_MINMAX)
        return depth_vis.astype(np.uint8)