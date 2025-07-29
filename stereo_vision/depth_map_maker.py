import cv2
import numpy as np
import matplotlib.pyplot as plt

class DepthMapMaker:
    def __init__(self, disparity_map_maker, focal_length, baseline):
        self.disparity_map_maker = disparity_map_maker
        self.focal_length = focal_length
        self.baseline = baseline

    def compute_depth_map(self, disparity_map):
        if np.any(disparity_map == 0):
            raise ValueError("Disparity map contains zero values, cannot compute depth.")

        disparity_map[disparity_map <= 0] = 0.01
        depth_map = (self.focal_length * self.baseline) / disparity_map

        return depth_map