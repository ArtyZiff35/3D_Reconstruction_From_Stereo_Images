import time
import cv2


MAX_FRAMES_CHECKBOARD = 20

chessboardCaptured = 0

left_camera_port = 0
right_camera_port = 1

left_camera = cv2.VideoCapture(left_camera_port)
time.sleep(0.5)
right_camera = cv2.VideoCapture(right_camera_port)
time.sleep(0.5)

# chessboard image capturing cycle
while chessboardCaptured != MAX_FRAMES_CHECKBOARD:

    cv2.waitKey(2000)
    print('STAND STILL')
    cv2.waitKey(1000)

    # Capturing image from camera
    left_return_value, leftImage = left_camera.read()
    right_return_value, rightImage = right_camera.read()

    cv2.imshow('left', leftImage)
    cv2.imshow('right', rightImage)

    # Finding chessboard corners
    left_gray = cv2.cvtColor(leftImage, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(rightImage, cv2.COLOR_BGR2GRAY)

    left_ret, leftCorners = cv2.findChessboardCorners(left_gray, (9, 6), None)
    rightRet, rightCorners = cv2.findChessboardCorners(right_gray, (9, 6), None)

    # only save images where the chessboard was found
    if left_ret and rightRet:

        cv2.imwrite("./calibration/left/" + str(chessboardCaptured) + ".png", leftImage)
        cv2.imwrite("./calibration/right/" + str(chessboardCaptured) + ".png", rightImage)

        cv2.waitKey(1000)
        chessboardCaptured = chessboardCaptured + 1
        print("OK, " + str(MAX_FRAMES_CHECKBOARD - chessboardCaptured) + " more images needed")
    else:
        print("Could not find chessboard")

del left_camera
del right_camera
cv2.destroyAllWindows()
