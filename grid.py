import os
import cv2
import glob
import numpy as np
import gdown

#פרמטרי כיול מצלמה 
camera_matrix = np.array([
    [3388.21744, 0, 901.478839],
    [0, 3356.12041, 595.209942],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([
    1.37363472, -68.4275187, 0.0146233947, 0.0668579741, 1366.46007
], dtype=np.float32)

# drive
folder_url = "https://drive.google.com/drive/folders/1EWOXetis6L35BEXHZ7Wm6qIPs5wHVsf2?usp=share_link"
out_folder = "input_images"
os.makedirs(out_folder, exist_ok=True)
gdown.download_folder(folder_url, output=out_folder, quiet=False)

# עיבוד התמונות 
output_folder = "output_grid_images"
os.makedirs(output_folder, exist_ok=True)
grid_spacing = 50  # כל כמה פיקסלים קו ברשת

def draw_pixel_grid(img, spacing=50):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h), (128, 128, 128), 1)
        cv2.putText(img, str(x), (x+2, 15), font, 0.4, (0, 255, 0), 1)
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w, y), (128, 128, 128), 1)
        cv2.putText(img, str(y), (5, y - 5), font, 0.4, (0, 255, 0), 1)
    return img

image_paths = glob.glob(os.path.join(out_folder, "*.*"))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print("download failed", img_path)
        continue

    h, w = img.shape[:2]
    new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mtx)
    with_grid = draw_pixel_grid(undistorted, spacing=grid_spacing)

    out_name = os.path.join(output_folder, f"grid_{os.path.basename(img_path)}")
    cv2.imwrite(out_name, with_grid)
    print("saved:", out_name)

print("Finish")
