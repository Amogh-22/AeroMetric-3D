import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pycolmap
import numpy as np
from PIL import Image
from transformers import pipeline

def transform_to_camera_coords(image, p3d):
    """
    Safely transform world 3D coordinates to camera coordinates
    handling pycolmap 3.x and 4.x API variants.
    """
    # Check if cam_from_world is callable or a Rigid3d property
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        rigid3d = cam_from_world()
    else:
        rigid3d = cam_from_world

    # If pycolmap supports direct matrix/vector multiplication on Rigid3d:
    if hasattr(rigid3d, "rotation") and hasattr(rigid3d, "translation"):
        R = rigid3d.rotation.matrix()
        t = rigid3d.translation
        p_cam = np.dot(R, p3d) + t
        return p_cam
    elif hasattr(rigid3d, "matrix"):
        T = rigid3d.matrix()
        p_homog = np.append(p3d, 1.0)
        return np.dot(T, p_homog)[:3]
    else:
        # Fallback to apply method
        return rigid3d * p3d

def generate_metric_depths(dataset_dir):
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"1. Loading Depth Anything V2 on device: {device.upper()}...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

    print("2. Loading PyCOLMAP Sparse Reconstruction...")
    sparse_path = os.path.join(dataset_dir, "sparse", "0")
    if not os.path.exists(sparse_path):
        print(f"Error: Could not find COLMAP model at {sparse_path}")
        return
        
    reconstruction = pycolmap.Reconstruction(sparse_path)
    output_dir = os.path.join(dataset_dir, "metric_depths")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded model with {len(reconstruction.images)} registered frames. Starting Fusion...")

    for image_id, image in reconstruction.images.items():
        image_name = image.name
        img_path = os.path.join(dataset_dir, "images", image_name)
        
        if not os.path.exists(img_path):
            continue

        raw_image = Image.open(img_path).convert('RGB')
        depth_output = pipe(raw_image)
        rel_depth_map = np.array(depth_output["depth"], dtype=np.float32)
        
        metric_depths = []
        rel_depths = []
        
        for p2d in image.points2D:
            if p2d.has_point3D():
                p3d = reconstruction.points3D[p2d.point3D_id].xyz
                p_cam = transform_to_camera_coords(image, p3d)
                metric_z = p_cam[2]
                
                u, v = int(p2d.xy[0]), int(p2d.xy[1])
                if 0 <= v < rel_depth_map.shape[0] and 0 <= u < rel_depth_map.shape[1]:
                    rel_val = rel_depth_map[v, u]
                    rel_depths.append(rel_val)
                    metric_depths.append(metric_z)
                    
        if len(metric_depths) >= 4:
            inv_rel_depths = 1.0 / (np.array(rel_depths) + 1e-5)
            
            A = np.vstack([inv_rel_depths, np.ones(len(inv_rel_depths))]).T
            sol, residuals, rank, s_vals = np.linalg.lstsq(A, metric_depths, rcond=None)
            s, t = sol[0], sol[1]
            
            inv_full_depth = 1.0 / (rel_depth_map + 1e-5)
            metric_depth_map = s * inv_full_depth + t
            metric_depth_map = np.clip(metric_depth_map, a_min=0.1, a_max=None)
            
            save_path = os.path.join(output_dir, f"{image_name}.npy")
            np.save(save_path, metric_depth_map)
            print(f"Scaled {image_name} | Used {len(metric_depths)} points | Scale: {s:.3f}, Shift: {t:.3f}")
        else:
            # Fallback estimation for frames with insufficient 3D sparse points
            norm_depth = (rel_depth_map - rel_depth_map.min()) / (np.ptp(rel_depth_map) + 1e-5)
            metric_depth_map = 5.0 + norm_depth * 25.0  # reasonable drone flight elevation span (5m - 30m)
            save_path = os.path.join(output_dir, f"{image_name}.npy")
            np.save(save_path, metric_depth_map)
            print(f"Fallback depth applied for {image_name} ({len(metric_depths)} points).")

if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    generate_metric_depths(dataset_dir)