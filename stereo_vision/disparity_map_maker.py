import cv2
import numpy as np
import matplotlib.pyplot as plt

class DisparityMapMaker:
    def __init__(self, min_disp=0, num_disp=16*5, block_size=5):
        self.stereoSGBM = cv2.StereoSGBM.create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
        )

    def compute_disparity_from_images(self,left_image_path,right_image_path):
        img_left, img_right = self._read_images_in_grayscale(left_image_path, right_image_path)
        return self.stereoSGBM.compute(img_left, img_right).astype(np.float32) / 16.0

    def _read_images_in_grayscale(self, left_image_path, right_image_path):
        img_left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

        if img_left is None or img_right is None:
            raise ValueError("Could not load images. Please check the paths.")

        return img_left, img_right

    def visualize_disparity_from_images(self, left_image_path, right_image_path):
        disparity = self.compute_disparity_from_images(left_image_path, right_image_path)

        plt.figure(figsize=(10, 5))
        plt.imshow(disparity, 'gray')
        plt.colorbar(label='Disparity')
        plt.title('Disparity Map (SGM)')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Example usage
    left_image_path = '../test_data/teddy1.png'
    right_image_path = '../test_data/teddy2.png'

    disparity_maker = DisparityMapMaker()
    disparity_maker.visualize_disparity_from_images(left_image_path, right_image_path)