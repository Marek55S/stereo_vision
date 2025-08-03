import cv2
import numpy as np
from time import sleep
from stereo_pair import StereoCamera

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

    def compute_disparity_live(self):
        stereo = StereoCamera()
        stereo.start()

        sleep(1)
        while True:
            left, right = stereo.get_frames()
            if left is None or right is None:
                continue

            disparity = self.compute(left, right)
            disp_vis = self.normalize(disparity)

            cv2.imshow("Disparity Map", disp_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        stereo.stop()
        cv2.destroyAllWindows()

    def show_disparity(self, disparity):
        disp_vis = self.normalize(disparity)
        cv2.imshow("Disparity Map", disp_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    disparity_computer = DisparityComputer()
    disparity_computer.compute_disparity_live()
    # Można również użyć:
    # disparity_computer.show_disparity(disparity)
    # gdzie 'disparity' to wcześniej obliczona mapa dysparycji.