import numpy as np
import cv2

# Camera calibration results
camera_matrix = np.array([
    [3.38821744e+03, 0.00000000e+00, 9.01478839e+02],
    [0.00000000e+00, 3.35612041e+03, 5.95209942e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([1.37363472e+00, -6.84275187e+01, 1.46233947e-02, 6.68579741e-02, 1.36646007e+03], dtype=np.float32)


# 3D real world coordinates from digitizer
#משנה את זה כל פעם בהתאם למספר הנבדק 
# 2 normal1
object_points = np.array([
    [-28.965, -37.577, -31.688],  # LPA
    [-11.194, -63.726, -35.089],  # left eye
    [-5.019, -65.552, -37.677],   # nose bridge
    [-5.206, -69.480, -32.350],   # nose tip
    [1.591, -64.934, -36.399],    # right eye
    [24.617, -36.662, -36.605],   # RPA
    [22.829, -46.234, -44.207],   # F8
    [6.667, -66.554, -44.802],    # FP2
    [-5.299, -65.962, -55.662],   # middle green
    [-13.587, -66.716, -44.609],  # FP1
    [-29.162, -47.334, -39.386],  # F7
    [-7.090, -41.420, -73.224],   # CZ
    [-12.623, -8.888, -45.240],   # O1
    [-5.030, -7.877, -45.794],    # OZ
    [3.197, -8.927, -45.968],     # O2
    [-7.090, -41.420, -73.224]    # Real CZ 
], dtype=np.float32)

# 2D image pixel coordinates from the labeled image
#משנה את זה כל פעם בהתאם למספר הנבדק 
# 2 normal1
image_points = np.array([
    [550, 500],  # LPA
    [605, 450],  # left eye
    [560, 470],  # nose bridge
    [570, 520],  # nose tip
    [505, 455],  # right eye
    [440, 460],  # RPA
    [325, 375],  # F8
    [510, 415],  # FP2
    [380, 375],  # middle green
    [410, 440],  # FP1
    [503, 450],  # F7
    [500, 270],  # CZ
    [360, 355],  # O1
    [395, 375],  # OZ
    [440, 395],  # O2
    [500, 270]   # Real CZ
], dtype=np.float32)

# Reshape for solvePnP
object_points = object_points.reshape(-1, 1, 3)
image_points = image_points.reshape(-1, 1, 2)

# Solve PnP
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
)

# Reproject points
projected_points, _ = cv2.projectPoints(object_points, rvec,tvec, camera_matrix, dist_coeffs)

# Compute reprojection errors
errors = np.linalg.norm(image_points - projected_points, axis=2)
mean_error = np.mean(errors)

# Print results
print("Reprojection errors per point (pixels):", errors.flatten())
print("Mean reprojection error: {:.2f} pixels".format(mean_error))


#results:
#Reprojection errors per point (pixels): [ 55.438168  80.76618   54.011852  71.10866   17.886925  83.24023
  73.7062    35.883583 136.06458  131.31345   50.643074  41.39217
  74.632164  12.114842  61.9873    41.39217 ]
#Mean reprojection error: 63.85 pixels
