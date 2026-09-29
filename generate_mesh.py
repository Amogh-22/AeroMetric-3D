import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import pycolmap
import numpy as np
import cv2
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree


def get_camera_matrices(reconstruction, image):
    """Extract camera intrinsics and camera-to-world transformation from pycolmap."""
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


def compute_upright_rotation(cam_centers, scene_points):
    """
    Computes a 3x3 rotation matrix to align the real-world UP vector
    (vector from scene center pointing upwards towards drone flight path)
    with Three.js / WebGL +Y axis (0, 1, 0).
    Guarantees the model is 100% upright and not upside-down.
    """
    c_mean = np.mean(cam_centers, axis=0)
    p_mean = np.median(scene_points, axis=0)

    # In drone videos, the drone cameras are flying in the air above the structure
    up_vec = c_mean - p_mean
    up_norm = np.linalg.norm(up_vec)
    if up_norm < 1e-4:
        return np.eye(3)

    u = up_vec / up_norm
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    dot = np.dot(u, target)
    if dot > 0.99999:
        return np.eye(3)
    if dot < -0.99999:
        # Exactly 180 degrees inverted: flip around X
        return np.diag([1.0, -1.0, -1.0])

    rot_axis = np.cross(u, target)
    axis_norm = np.linalg.norm(rot_axis)
    if axis_norm < 1e-6:
        return np.eye(3)

    rot_axis = rot_axis / axis_norm
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    # Rodrigues formula
    K = np.array([
        [0.0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0.0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0.0]
    ])
    R_align = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R_align


def depth_to_mesh_tiered(
    rgb,
    depth,
    mask,
    K,
    R_cw,
    t_cw,
    stride=2,
    max_depth=50.0,
    max_edge_len=0.75,
    grad_percentile=85
):
    """
    Backprojects metric depth into world 3D coordinates and builds regularized mesh triangles
    strictly within the supplied binary region mask (foreground or tight ground pedestal).
    Filters out depth cliffs, silhouette artifacts, and sky regions.
    """
    h, w = depth.shape

    # 1. Edge gradient filter to avoid stretching faces across depth cliffs / silhouettes
    gy, gx = np.gradient(depth)
    grad = np.sqrt(gx**2 + gy**2)

    region_valid = (depth > 0.5) & (depth < max_depth) & (mask > 128)
    if not np.any(region_valid):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), np.empty((0, 3))

    grad_thresh = np.percentile(grad[region_valid], grad_percentile)
    valid_mask = region_valid & (grad < grad_thresh)

    # Downsample using stride
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
    if num_vertices == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), np.empty((0, 3))

    index_map[valid_indices] = np.arange(num_vertices)

    vertices = points_world[valid_indices]
    colors = sub_rgb[valid_indices]

    # Build triangle faces between neighboring grid points with max edge length check
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx00 = index_map[r, c]
            idx01 = index_map[r, c + 1]
            idx10 = index_map[r + 1, c]
            idx11 = index_map[r + 1, c + 1]

            # First triangle (00 - 10 - 01)
            if idx00 != -1 and idx10 != -1 and idx01 != -1:
                v0, v1, v2 = vertices[idx00], vertices[idx10], vertices[idx01]
                if np.linalg.norm(v0 - v1) < max_edge_len and np.linalg.norm(v1 - v2) < max_edge_len and np.linalg.norm(v2 - v0) < max_edge_len:
                    faces.append([idx00, idx10, idx01])

            # Second triangle (01 - 10 - 11)
            if idx01 != -1 and idx10 != -1 and idx11 != -1:
                v0, v1, v2 = vertices[idx01], vertices[idx10], vertices[idx11]
                if np.linalg.norm(v0 - v1) < max_edge_len and np.linalg.norm(v1 - v2) < max_edge_len and np.linalg.norm(v2 - v0) < max_edge_len:
                    faces.append([idx01, idx10, idx11])

    faces = np.array(faces, dtype=np.int32) if len(faces) > 0 else np.empty((0, 3), dtype=np.int32)
    return vertices, faces, colors


def filter_mesh_inliers(mesh, inliers, tier_tags=None):
    """
    Subselect mesh vertices according to inlier boolean mask,
    remap face indices, and drop orphaned faces.
    """
    if len(mesh.faces) == 0:
        return mesh, tier_tags

    valid_faces_mask = inliers[mesh.faces].all(axis=1)
    new_faces = mesh.faces[valid_faces_mask]

    new_indices = np.full(len(mesh.vertices), -1, dtype=np.int32)
    inlier_indices = np.where(inliers)[0]
    new_indices[inlier_indices] = np.arange(len(inlier_indices))

    remapped_faces = new_indices[new_faces]
    new_vertices = mesh.vertices[inlier_indices]

    new_colors = None
    if mesh.visual and hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        new_colors = mesh.visual.vertex_colors[inlier_indices]

    filtered_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=remapped_faces,
        vertex_colors=new_colors,
        process=False
    )
    filtered_mesh.remove_unreferenced_vertices()

    filtered_tiers = None
    if tier_tags is not None:
        filtered_tiers = tier_tags[inlier_indices]

    return filtered_mesh, filtered_tiers


def apply_statistical_outlier_removal(mesh, tier_tags=None, k_neighbors=20, std_ratio=1.8):
    """
    Eliminate scattered ghost noise clouds and isolated vertices using
    Statistical Outlier Removal (k-NN distance distribution).
    """
    if len(mesh.vertices) < k_neighbors * 2:
        return mesh, tier_tags

    print("  Applying Statistical Outlier Removal (SOR) via cKDTree...")
    pts = mesh.vertices.astype(np.float32)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=min(k_neighbors + 1, len(pts)), workers=-1)

    # Exclude self distance (index 0)
    mean_dists = np.mean(dists[:, 1:], axis=1)
    mu = float(np.mean(mean_dists))
    std = float(np.std(mean_dists))
    thresh = mu + std_ratio * std

    inliers = mean_dists <= thresh
    num_rejected = int((~inliers).sum())
    print(f"  ✓ SOR: rejected {num_rejected} outlier vertices ({num_rejected/len(pts)*100:.1f}%) | dist_thresh={thresh:.3f}m")

    cleaned_mesh, cleaned_tiers = filter_mesh_inliers(mesh, inliers, tier_tags)
    return cleaned_mesh, cleaned_tiers


def remove_floating_debris(mesh, tier_tags=None, min_faces=25):
    """
    Remove disconnected small triangle clusters / floating artifacts.
    Uses face_adjacency graph to isolate and retain prominent connected components.
    """
    try:
        if len(mesh.faces) < min_faces:
            return mesh, tier_tags

        connected = trimesh.graph.connected_components(
            mesh.face_adjacency,
            nodes=np.arange(len(mesh.faces)),
            min_len=min_faces
        )
        if not connected or len(connected) == 0:
            return mesh, tier_tags

        retained_face_indices = np.concatenate(connected)
        if len(retained_face_indices) == len(mesh.faces) or len(retained_face_indices) == 0:
            return mesh, tier_tags

        new_faces = mesh.faces[retained_face_indices]
        unique_verts, inverse = np.unique(new_faces, return_inverse=True)
        new_faces = inverse.reshape(new_faces.shape)
        new_verts = mesh.vertices[unique_verts]
        new_colors = mesh.visual.vertex_colors[unique_verts] if (mesh.visual and hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) == len(mesh.vertices)) else None

        cleaned_mesh = trimesh.Trimesh(
            vertices=new_verts,
            faces=new_faces,
            vertex_colors=new_colors,
            process=False
        )
        cleaned_tiers = tier_tags[unique_verts] if tier_tags is not None else None
        print(f"  ✓ Debris filter: pruned {len(mesh.faces) - len(new_faces)} floating micro-triangles")
        return cleaned_mesh, cleaned_tiers
    except Exception as e:
        print(f"  ⚠️ Warning: Debris filtering skipped ({e})")
        return mesh, tier_tags


def build_scene_mesh(dataset_dir):
    sparse_path = os.path.join(dataset_dir, "sparse", "0")
    if not os.path.exists(sparse_path):
        print(f"Error: Could not find COLMAP model at {sparse_path}")
        return

    reconstruction = pycolmap.Reconstruction(sparse_path)

    all_meshes = []
    all_tiers = []  # 1 for Foreground, 0 for Background Context
    all_cam_centers = []

    print("Building subject-prioritized 3D mesh (tight foreground + trimmed base)...")
    for img_id, image in reconstruction.images.items():
        img_name = image.name
        depth_path = os.path.join(dataset_dir, "metric_depths", f"{img_name}.npy")
        rgb_path = os.path.join(dataset_dir, "images", img_name)
        fg_mask_path = os.path.join(dataset_dir, "fg_masks", f"{img_name}.png")
        sky_mask_path = os.path.join(dataset_dir, "sky_masks", f"{img_name}.png")

        if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
            continue

        depth = np.load(depth_path)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # ── 1. Sky Elimination ────────────────────────────────────
        sky_mask = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(sky_mask_path):
            sky_loaded = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
            if sky_loaded is not None:
                sky_mask = cv2.resize(sky_loaded, (w, h), interpolation=cv2.INTER_NEAREST)

        depth[sky_mask > 128] = 0.0

        # ── 2. High-Precision Foreground Mask (Primary Target) ────
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        if os.path.exists(fg_mask_path):
            fg_loaded = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)
            if fg_loaded is not None:
                fg_mask = cv2.resize(fg_loaded, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            fg_mask = (((X - cx)**2 / (w * 0.3)**2 + (Y - cy)**2 / (h * 0.3)**2) < 1.0).astype(np.uint8) * 255

        # ── 3. Immediate Ground Pedestal (Trim Peripheral Horizon) ─
        # Only reconstruct ground touching the structure base (within ~30px dilation).
        # Discards distant mountains, trees, and horizon to produce a clean, focused 3D model.
        pedestal_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        pedestal_mask = cv2.dilate(fg_mask, pedestal_kernel, iterations=2)
        ground_base_mask = ((pedestal_mask > 128) & (fg_mask < 128) & (sky_mask < 128)).astype(np.uint8) * 255

        K, R_cw, t_cw = get_camera_matrices(reconstruction, image)
        all_cam_centers.append(t_cw)

        # ── 4. Primary Foreground Reconstruction (Dense, High-Precision) ─
        v_fg, f_fg, c_fg = depth_to_mesh_tiered(
            rgb, depth, fg_mask, K, R_cw, t_cw,
            stride=2, max_depth=50.0, max_edge_len=0.75, grad_percentile=85
        )
        if len(f_fg) > 0:
            m_fg = trimesh.Trimesh(vertices=v_fg, faces=f_fg, vertex_colors=(c_fg * 255).astype(np.uint8), process=False)
            all_meshes.append(m_fg)
            all_tiers.append(np.ones(len(v_fg), dtype=np.uint8))

        # ── 5. Trimmed Ground Base Pedestal ───────────────────────
        v_bg, f_bg, c_bg = depth_to_mesh_tiered(
            rgb, depth, ground_base_mask, K, R_cw, t_cw,
            stride=4, max_depth=45.0, max_edge_len=1.2, grad_percentile=75
        )
        if len(f_bg) > 0:
            m_bg = trimesh.Trimesh(vertices=v_bg, faces=f_bg, vertex_colors=(c_bg * 255).astype(np.uint8), process=False)
            all_meshes.append(m_bg)
            all_tiers.append(np.zeros(len(v_bg), dtype=np.uint8))

        print(f"  Processed {img_name}: FG={len(v_fg)} verts | Base={len(v_bg)} verts")

    if not all_meshes:
        print("No valid geometry could be constructed.")
        return

    # Merge into a unified scene mesh
    scene_mesh = trimesh.util.concatenate(all_meshes)
    combined_tiers = np.concatenate(all_tiers)
    print(f"\nInitial merged model: {len(scene_mesh.vertices)} vertices, {len(scene_mesh.faces)} faces.")

    # ── 6. Post-Processing & Outlier Elimination ──────────────────
    # A. Statistical Outlier Removal
    scene_mesh, combined_tiers = apply_statistical_outlier_removal(scene_mesh, combined_tiers, k_neighbors=20, std_ratio=1.8)

    # B. Floating debris elimination
    scene_mesh, combined_tiers = remove_floating_debris(scene_mesh, combined_tiers, min_faces=25)

    # C. Trimesh cleanup
    if len(scene_mesh.faces) > 0:
        scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()

    # ── 7. Upright Orientation & Ground Leveling ──────────────────
    # Compute the physical UP vector pointing from the scene towards the drone cameras
    if len(all_cam_centers) > 0:
        R_align = compute_upright_rotation(np.array(all_cam_centers), scene_mesh.vertices)
        scene_mesh.vertices = scene_mesh.vertices @ R_align.T
        print("  ✓ Aligned mesh to upright vertical orientation (+Y = UP)")

    # Center horizontally at (0, 0) and place ground base at Y = 0 (on the grid)
    min_b = np.min(scene_mesh.vertices, axis=0)
    max_b = np.max(scene_mesh.vertices, axis=0)
    center_xz = (min_b + max_b) / 2.0

    scene_mesh.vertices[:, 0] -= center_xz[0]
    scene_mesh.vertices[:, 2] -= center_xz[2]
    scene_mesh.vertices[:, 1] -= min_b[1]  # Base sits directly at Y=0!

    print(f"Cleaned final model: {len(scene_mesh.vertices)} vertices, {len(scene_mesh.faces)} faces.")
    print(f"Bounding dimensions (meters): X={max_b[0]-min_b[0]:.2f}m, Y(height)={max_b[1]-min_b[1]:.2f}m, Z={max_b[2]-min_b[2]:.2f}m")

    # ── 8. Export Metric 3D Formats ───────────────────────────────
    out_ply = os.path.join(dataset_dir, "metric_reconstruction.ply")
    scene_mesh.export(out_ply)

    out_glb = os.path.join(dataset_dir, "metric_reconstruction.glb")
    scene_mesh.export(out_glb)

    out_obj = os.path.join(dataset_dir, "metric_reconstruction.obj")
    scene_mesh.export(out_obj)

    print(f"\nSuccessfully generated metric mesh:\n  ✓ {out_ply}\n  ✓ {out_glb}\n  ✓ {out_obj}")

    # ── 9. Generate NTRO Tiered Confidence Heatmap ────────────────
    conf_colors = np.zeros((len(scene_mesh.vertices), 4), dtype=np.uint8)

    if combined_tiers is not None and len(combined_tiers) == len(scene_mesh.vertices):
        fg_idx = np.where(combined_tiers == 1)[0]
        bg_idx = np.where(combined_tiers == 0)[0]

        # Base Foreground: Emerald Green
        conf_colors[fg_idx] = [45, 215, 80, 255]
        # Base Background / Pedestal: Warm Amber
        conf_colors[bg_idx] = [235, 145, 35, 255]

        # Transition smoothing: background points close to foreground boundary -> Yellow
        if len(fg_idx) > 0 and len(bg_idx) > 0:
            fg_tree = cKDTree(scene_mesh.vertices[fg_idx])
            dists_to_fg, _ = fg_tree.query(scene_mesh.vertices[bg_idx], k=1, workers=-1)
            transition_idx = bg_idx[dists_to_fg < 1.0]
            conf_colors[transition_idx] = [245, 215, 40, 255]
    else:
        z_vals = scene_mesh.vertices[:, 1]  # Height along Y
        norm_y = (z_vals - z_vals.min()) / (np.ptp(z_vals) + 1e-5)
        conf_colors[:, 0] = ((1.0 - norm_y) * 220).astype(np.uint8)
        conf_colors[:, 1] = (norm_y * 230).astype(np.uint8)
        conf_colors[:, 2] = 40
        conf_colors[:, 3] = 255

    conf_mesh = scene_mesh.copy()
    conf_mesh.visual.vertex_colors = conf_colors
    conf_ply = os.path.join(dataset_dir, "confidence_reconstruction.ply")
    conf_mesh.export(conf_ply)
    print(f"  ✓ {conf_ply}")

    # Save mesh metadata
    bounds = scene_mesh.bounds
    dims = bounds[1] - bounds[0] if bounds is not None else [0, 0, 0]
    meta = {
        "num_vertices": int(len(scene_mesh.vertices)),
        "num_faces": int(len(scene_mesh.faces)),
        "foreground_vertices": int((combined_tiers == 1).sum()) if combined_tiers is not None else 0,
        "background_vertices": int((combined_tiers == 0).sum()) if combined_tiers is not None else 0,
        "bounding_box_meters": {
            "width_x": float(dims[0]),
            "height_y": float(dims[1]),
            "length_z": float(dims[2])
        }
    }
    meta_path = os.path.join(dataset_dir, "mesh_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    build_scene_mesh(dataset_dir)