import cv2
import numpy as np

# chessboard sizing
chessboard_size = (9, 6)  # מספר פינות 
square_size = 1.0

objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# load video
cap = cv2.VideoCapture('videoss/calibration.mov') 
frame_interval = 10 

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret_cb:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            # display of corners found
            cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret_cb)
            cv2.imshow('Chessboard', frame)
            cv2.waitKey(100)

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# results
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# save to file
np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)