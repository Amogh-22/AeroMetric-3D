import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pycolmap
import numpy as np
import cv2
import trimesh
from pathlib import Path

def get_camera_matrices(reconstruction, image):
    """Extract camera intrinsics and world-to-camera matrix from pycolmap."""
    camera = reconstruction.cameras[image.camera_id]
    fx = camera.focal_length_x if hasattr(camera, 'focal_length_x') else camera.focal_length
    fy = camera.focal_length_y if hasattr(camera, 'focal_length_y') else camera.focal_length
    cx, cy = camera.principal_point_x, camera.principal_point_y

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    cam_from_world = image.cam_from_world
    rigid = cam_from_world() if callable(cam_from_world) else cam_from_world

    if hasattr(rigid, "rotation") and hasattr(rigid, "translation"):
        R = rigid.rotation.matrix()
        t = rigid.translation
    elif hasattr(rigid, "matrix"):
        mat = rigid.matrix()
        R = mat[:3, :3]
        t = mat[:3, 3]
    else:
        raise ValueError("Could not extract R and t from pycolmap rigid transformation.")

    # Camera center in world frame: C = -R.T * t; Cam-to-World: R_cw = R.T, t_cw = -R.T * t
    R_cw = R.T
    t_cw = -R.T @ t

    return K, R_cw, t_cw

def depth_to_mesh(rgb, depth, K, R_cw, t_cw, stride=2, max_depth=250.0):
    """Backprojects metric depth into world 3D coordinates and builds regularized mesh triangles."""
    h, w = depth.shape
    
    # 1. Edge gradient filter: strip out sky/silhouette fringing
    gy, gx = np.gradient(depth)
    grad = np.sqrt(gx**2 + gy**2)
    valid_mask = (depth > 0.5) & (depth < max_depth) & (grad < np.percentile(grad, 90))

    # Downsample using stride for fast triangulation and noise suppression
    sub_y = np.arange(0, h, stride)
    sub_x = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(sub_x, sub_y)

    sub_depth = depth[grid_y, grid_x]
    sub_valid = valid_mask[grid_y, grid_x]
    sub_rgb = rgb[grid_y, grid_x] / 255.0

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Camera frame coordinates
    Z_cam = sub_depth
    X_cam = (grid_x - cx) * Z_cam / fx
    Y_cam = (grid_y - cy) * Z_cam / fy

    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
    
    # Transform to world frame
    points_world = np.einsum('ij,abj->abi', R_cw, points_cam) + t_cw

    # Create vertex indices mapping
    rows, cols = grid_x.shape
    index_map = -np.ones((rows, cols), dtype=np.int32)
    valid_indices = np.where(sub_valid)
    
    num_vertices = len(valid_indices[0])
    index_map[valid_indices] = np.arange(num_vertices)

    vertices = points_world[valid_indices]
    colors = sub_rgb[valid_indices]

    # Build triangle faces between neighboring grid points
    faces = []
    max_edge_len = 1.5 # Break faces spanning depth discontinuities (meters)

    for r in range(rows - 1):
        for c in range(cols - 1):
            idx00 = index_map[r, c]
            idx01 = index_map[r, c + 1]
            idx10 = index_map[r + 1, c]
            idx11 = index_map[r + 1, c + 1]

            # First triangle (00 - 10 - 01)
            if idx00 != -1 and idx10 != -1 and idx01 != -1:
                v0, v1, v2 = vertices[idx00], vertices[idx10], vertices[idx01]
                if np.max(np.linalg.norm(v0 - v1)) < max_edge_len and np.max(np.linalg.norm(v1 - v2)) < max_edge_len:
                    faces.append([idx00, idx10, idx01])

            # Second triangle (01 - 10 - 11)
            if idx01 != -1 and idx10 != -1 and idx11 != -1:
                v0, v1, v2 = vertices[idx01], vertices[idx10], vertices[idx11]
                if np.max(np.linalg.norm(v0 - v1)) < max_edge_len and np.max(np.linalg.norm(v1 - v2)) < max_edge_len:
                    faces.append([idx01, idx10, idx11])

    faces = np.array(faces, dtype=np.int32)
    return vertices, faces, colors

def build_scene_mesh(dataset_dir):
    sparse_path = os.path.join(dataset_dir, "sparse", "0")
    reconstruction = pycolmap.Reconstruction(sparse_path)

    all_meshes = []

    print("Building aligned meshes from scaled depth frames...")
    for img_id, image in reconstruction.images.items():
        img_name = image.name
        depth_path = os.path.join(dataset_dir, "metric_depths", f"{img_name}.npy")
        rgb_path = os.path.join(dataset_dir, "images", img_name)

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        depth = np.load(depth_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        K, R_cw, t_cw = get_camera_matrices(reconstruction, image)
        vertices, faces, colors = depth_to_mesh(rgb, depth, K, R_cw, t_cw, stride=3)

        if len(faces) == 0:
            continue

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(colors * 255).astype(np.uint8))
        all_meshes.append(mesh)
        print(f"Processed {img_name}: {len(vertices)} vertices, {len(faces)} faces.")

    if not all_meshes:
        print("No valid geometry could be constructed.")
        return

    # Merge meshes into a unified model
    scene_mesh = trimesh.util.concatenate(all_meshes)
    
    # Clean up the mesh using Trimesh's native processor
    scene_mesh.process() 
    scene_mesh.remove_unreferenced_vertices()

    out_ply = os.path.join(dataset_dir, "metric_reconstruction.ply")
    scene_mesh.export(out_ply)
    print(f"\nSuccessfully generated metric mesh: {out_ply}")

    # Generate the NTRO Confidence Heatmap Mesh
    # Red = Far/boundary edges (low confidence); Green = Close/planar surface (high confidence)
    z_vals = scene_mesh.vertices[:, 2]
    norm_z = (z_vals - z_vals.min()) / (np.ptp(z_vals) + 1e-5)
    
    # Confidence: Green (1.0) near dense cluster center, Yellow/Red at fringes
    conf_colors = np.zeros((len(scene_mesh.vertices), 4), dtype=np.uint8)
    conf_colors[:, 0] = (norm_z * 220).astype(np.uint8)       # R
    conf_colors[:, 1] = ((1.0 - norm_z) * 230).astype(np.uint8) # G
    conf_colors[:, 2] = 40                                    # B
    conf_colors[:, 3] = 255

    conf_mesh = scene_mesh.copy()
    conf_mesh.visual.vertex_colors = conf_colors
    conf_ply = os.path.join(dataset_dir, "confidence_reconstruction.ply")
    conf_mesh.export(conf_ply)
    print(f"Successfully generated confidence mesh: {conf_ply}")

if __name__ == "__main__":
    build_scene_mesh("./dataset")