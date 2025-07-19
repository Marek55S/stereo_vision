import cv2
import numpy as np
import matplotlib.pyplot as plt

class DisparityMapMaker:
    def __init__(self):
        pass

# Załaduj obrazy stereo (lewą i prawą kamerę)
img_left = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# Sprawdzenie, czy obrazy się załadowały
if img_left is None or img_right is None:
    raise ValueError("Nie udało się załadować obrazów.")

# Parametry SGBM
min_disp = 0
num_disp = 16 * 5  # musi być wielokrotnością 16
block_size = 5

stereo = cv2.StereoSGBM.create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 1 * block_size ** 2,
    P2=32 * 1 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Obliczenie mapy dysparytetu
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# Wizualizacja
plt.figure(figsize=(10, 5))
plt.imshow(disparity, 'gray')
plt.colorbar(label='Disparity')
plt.title('Disparity Map (SGM)')
plt.axis('off')
plt.show()
