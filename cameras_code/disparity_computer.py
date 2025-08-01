import cv2
import numpy as np


class DisparityComputer:
    def __init__(self):
        self.stereo = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def compute(self, left_frame, right_frame):
        # Konwertuj do skali szarości
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Oblicz mapę dysparycji
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        return disparity

    def normalize(self, disparity):
        # Przeskaluj do widocznego zakresu 0-255
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        return disp_vis.astype(np.uint8)
