import cv2
import numpy as np
import matplotlib.pyplot as plt
from stereo_vision.disparity_map_maker import DisparityMapMaker

class DepthMapMaker:
    def __init__(self,  focal_length, baseline):
        self.disparity_map_maker = DisparityMapMaker()
        self.focal_length = focal_length
        self.baseline = baseline

    def compute_depth_map_from_images(self, left_image_path, right_image_path):
        try:
            disparity_map = self.disparity_map_maker.compute_disparity_from_images(left_image_path, right_image_path)
        except ValueError as e:
            print(f"Error during disparity computation: {e}")
            return None

        disparity_map[disparity_map <= 0] = 0.01
        depth_map = (self.focal_length * self.baseline) / disparity_map

        return depth_map

    def visualize_depth_map_from_images(self, left_image_path, right_image_path):
        try:
            depth_map = self.compute_depth_map_from_images(left_image_path, right_image_path)
            if depth_map is None:
                return

            plt.figure(figsize=(10, 5))
            plt.imshow(depth_map, cmap='plasma')
            plt.colorbar(label='Depth')
            plt.title('Depth Map')
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error visualizing depth map: {e}")


if __name__ == "__main__":
    # Example usage
    left_image_path = '../test_data/teddy1.png'
    right_image_path = '../test_data/teddy2.png'

    focal_length = 30  # Example focal length in pixels
    baseline = 0.1      # Example baseline in meters

    depth_map_maker = DepthMapMaker( focal_length, baseline)
    depth_map_maker.visualize_depth_map_from_images(left_image_path, right_image_path)