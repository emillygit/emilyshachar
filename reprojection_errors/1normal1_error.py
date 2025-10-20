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
# 1 normal1

object_points = np.array([
    [-27.594, -31.162, -25.456],   # LPA
    [-9.489, -56.840, -34.442],    # left eye
    [-2.422, -59.028, -37.022],    # nose bridge
    [-2.272, -63.173, -32.666],    # nose tip
    [ 4.452, -57.491, -34.021],    # right eye
    [24.799, -31.421, -24.841],    # RPA
    [22.910, -44.290, -37.904],    # F8
    [12.479, -55.797, -38.989],    # FP2
    [-1.495, -57.797, -45.368],    # middle green
    [-5.628, -58.775, -39.348],    # FP1
    [-24.243, -45.593, -35.012],   # F7
    [-0.185, -27.108, -66.815],    # CZ
    [-0.185, -27.108, -66.815],    # Real CZ (same location)
], dtype=np.float32)


# 2D image pixel coordinates from the labeled image
#משנה את זה כל פעם בהתאם למספר הנבדק 
# 1 normal1

image_points = np.array([
    [525, 380],  # LPA
    [730, 290],  # left eye
    [685, 300],  # nose bridge
    [680, 330],  # nose tip
    [650, 285],  # right eye
    [610, 290],  # RPA
    [545, 255],  # F8
    [645, 245],  # FP2
    [680, 205],  # middle green
    [725, 250],  # FP1
    [480, 280],  # F7
    [705, 130],  # CZ
    [705, 130],  # Real CZ
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
#Reprojection errors per point (pixels): [ 74.422066   79.02345    10.331881    5.5189137  21.491535   73.93534
  99.92662    39.337738   57.205692   64.39111   120.50366    91.24312
  91.24312  ]
#Mean reprojection error: 63.74 pixels