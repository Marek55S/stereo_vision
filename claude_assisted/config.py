class Config:
    # Parametry kamery
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FOCAL_LENGTH_PX = 2714.29
    BASELINE_M = 0.06

    # Parametry MAVLink
    MAVLINK_CONNECTION = '/dev/ttyACM0'  # lub '/dev/serial0' dla RPi UART
    MAVLINK_BAUD = 57600

    # Parametry obstacle avoidance
    MAX_RANGE_M = 8.0
    MIN_DISTANCE_M = 0.1
    UPDATE_RATE_HZ = 10  # Częstotliwość wysyłania danych

    # Parametry dysparycji (dostosuj do Twojego setupu)
    DISPARITY_PARAMS = {
        'minDisparity': 0,
        'numDisparities': 64,
        'blockSize': 5,
        'P1': 8 * 3 * 5 ** 2,
        'P2': 32 * 3 * 5 ** 2,
        'disp12MaxDiff': 1,
        'uniquenessRatio': 10,
        'speckleWindowSize': 100,
        'speckleRange': 32
    }