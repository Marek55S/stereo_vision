import cv2


cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

if not cam0.isOpened():
    print("Camera 0 failed to open")
if not cam1.isOpened():
    print("Camera 1 failed to open")

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if ret0:
        cv2.imshow("Camera 0", frame0)
    else:
        print("No image from camera 0")

    if ret1:
        cv2.imshow("Camera 1", frame1)
    else:
        print("No image from camera 1")

    if cv2.waitKey(1) == 27:
        break

cam0.release()
cam1.release()
cv2.destroyAllWindows()
