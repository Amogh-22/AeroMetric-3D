import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import pycolmap
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline


def transform_to_camera_coords(image, p3d):
    """
    Safely transform world 3D coordinates to camera coordinates
    handling pycolmap 3.x and 4.x API variants.
    """
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        rigid3d = cam_from_world()
    else:
        rigid3d = cam_from_world

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
        return rigid3d * p3d


# =====================================================================
#  Foreground / Sky Segmentation (Disparity + Color + Spatial Priors)
# =====================================================================

def generate_segmentation_masks(rel_disparity_map, frame_shape, frame_bgr=None):
    """
    Generate high-precision foreground and sky masks from relative disparity
    (Depth Anything V2 output: higher disparity = closer; lower disparity = farther/sky)
    and optional BGR color features.

    Returns:
        fg_mask  (uint8, 0 or 255)
        sky_mask (uint8, 0 or 255)
    """
    h, w = frame_shape[:2]

    if rel_disparity_map.shape != (h, w):
        disp = cv2.resize(rel_disparity_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        disp = rel_disparity_map.copy()

    # Robust normalization to [0, 1] where 1 = closest (max disparity), 0 = farthest/infinity
    d_lo, d_hi = np.percentile(disp, 1), np.percentile(disp, 99)
    norm_disp = np.clip((disp - d_lo) / (d_hi - d_lo + 1e-6), 0.0, 1.0)

    # ── 1. Sky Detection ──────────────────────────────────────────
    # Depth Anything assigns lowest disparity (norm_disp near 0) to sky & horizon
    sky_depth_candidate = norm_disp < 0.18

    # Color confirmation if frame_bgr is provided
    if frame_bgr is not None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        blue_sky = (H >= 90) & (H <= 135) & (S >= 15) & (V >= 70)
        white_sky = (S <= 50) & (V >= 170)
        sky_color = blue_sky | white_sky

        # Combine depth candidate with color consistency
        sky_combined = (sky_depth_candidate & sky_color).astype(np.uint8) * 255

        # Check gradient in potential sky
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
        sky_combined[np.abs(grad) > 30] = 0

        # Floodfill from top border to enforce upper-atmosphere connectivity
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        for x in range(0, w, max(1, w // 40)):
            if sky_combined[0, x] == 255 and flood_mask[1, x + 1] == 0:
                cv2.floodFill(sky_combined, flood_mask, (x, 0), 128)

        # Region with low depth or connected top sky
        sky_raw = ((sky_combined == 128) | (norm_disp < 0.10)).astype(np.uint8) * 255
    else:
        sky_raw = (norm_disp < 0.15).astype(np.uint8) * 255

    kernel_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    sky_mask = cv2.morphologyEx(sky_raw, cv2.MORPH_CLOSE, kernel_lg)

    # ── 2. Foreground Detection ───────────────────────────────────
    non_sky = sky_mask < 128

    # Center-weighted 2D Gaussian prior (drone footage frames ROI in center)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    sig_y, sig_x = h * 0.38, w * 0.38
    center_w = np.exp(-((Y - cy) ** 2 / (2 * sig_y ** 2) + (X - cx) ** 2 / (2 * sig_x ** 2)))

    # High disparity = closest subject; combined with centrality
    fg_score = norm_disp * 0.65 + center_w * 0.35
    fg_score[~non_sky] = 0.0

    if non_sky.any():
        fg_thresh = np.percentile(fg_score[non_sky], 50)
    else:
        fg_thresh = 0.50

    fg_raw = ((fg_score > fg_thresh) & (norm_disp > 0.25) & non_sky).astype(np.uint8) * 255
    kernel_md = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_raw, cv2.MORPH_CLOSE, kernel_md)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_md)

    return fg_mask, sky_mask


# =====================================================================
#  RANSAC Depth Alignment (Robust Inlier Regression)
# =====================================================================

def ransac_depth_fit(inv_rel, metric, n_iter=300, inlier_thresh=0.8):
    """
    Robustly fit  metric_z = s · (1/rel) + t  via RANSAC.
    Rejects outlier sparse points from foliage, terrain edges, or sky fringe.
    Returns (s, t, inlier_mask).
    """
    n = len(inv_rel)
    if n < 4:
        A = np.vstack([inv_rel, np.ones(n)]).T
        sol, *_ = np.linalg.lstsq(A, metric, rcond=None)
        return float(sol[0]), float(sol[1]), np.ones(n, dtype=bool)

    best_n = 0
    best_s, best_t = 1.0, 0.0
    best_mask = np.ones(n, dtype=bool)

    for _ in range(n_iter):
        idx = np.random.choice(n, 2, replace=False)
        # Avoid degenerate anchor pairs with almost identical disparity
        if abs(inv_rel[idx[0]] - inv_rel[idx[1]]) < 1e-4:
            continue
        A_s = np.vstack([inv_rel[idx], np.ones(2)]).T
        try:
            sol = np.linalg.solve(A_s, metric[idx])
        except np.linalg.LinAlgError:
            continue
        s, t = sol
        if s <= 0 or s > 500:
            continue

        predicted = s * inv_rel + t
        inliers = np.abs(predicted - metric) < inlier_thresh
        cnt = int(inliers.sum())
        if cnt > best_n:
            best_n = cnt
            best_s, best_t = float(s), float(t)
            best_mask = inliers

    # Re-fit on all RANSAC inliers
    if best_n >= 4:
        A_in = np.vstack([inv_rel[best_mask], np.ones(best_n)]).T
        sol, *_ = np.linalg.lstsq(A_in, metric[best_mask], rcond=None)
        best_s, best_t = float(sol[0]), float(sol[1])

    return best_s, best_t, best_mask


# =====================================================================
#  Main Entry Point
# =====================================================================

def generate_metric_depths(dataset_dir):
    # ── Device selection ──────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"1. Loading Depth Anything V2 on device: {device.upper()}...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device
    )

    # ── Load SfM reconstruction ───────────────────────────────────
    print("2. Loading PyCOLMAP Sparse Reconstruction...")
    sparse_path = os.path.join(dataset_dir, "sparse", "0")
    if not os.path.exists(sparse_path):
        print(f"Error: Could not find COLMAP model at {sparse_path}")
        return

    reconstruction = pycolmap.Reconstruction(sparse_path)

    # Output directories
    depth_dir = os.path.join(dataset_dir, "metric_depths")
    fg_dir    = os.path.join(dataset_dir, "fg_masks")
    sky_dir   = os.path.join(dataset_dir, "sky_masks")
    for d in [depth_dir, fg_dir, sky_dir]:
        os.makedirs(d, exist_ok=True)

    n_images = len(reconstruction.images)
    print(f"   Loaded {n_images} registered frames.")
    print("3. Running Depth Estimation + Foreground Prioritization + RANSAC Fusion...\n")

    stats = {}

    for image_id, image in reconstruction.images.items():
        image_name = image.name
        img_path = os.path.join(dataset_dir, "images", image_name)
        if not os.path.exists(img_path):
            continue

        raw_image = Image.open(img_path).convert("RGB")
        frame_bgr = cv2.imread(img_path)

        # ── A. Relative disparity estimation ──────────────────────
        depth_out = pipe(raw_image)
        rel_disp_map = np.array(depth_out["depth"], dtype=np.float32)

        # ── B. High-precision Foreground / sky segmentation ───────
        fg_mask, sky_mask = generate_segmentation_masks(
            rel_disp_map,
            frame_bgr.shape,
            frame_bgr=frame_bgr
        )
        cv2.imwrite(os.path.join(fg_dir,  f"{image_name}.png"), fg_mask)
        cv2.imwrite(os.path.join(sky_dir, f"{image_name}.png"), sky_mask)

        # ── C. Collect SfM anchor correspondences ────────────────
        fg_resized = cv2.resize(
            fg_mask,
            (rel_disp_map.shape[1], rel_disp_map.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        sky_resized = cv2.resize(
            sky_mask,
            (rel_disp_map.shape[1], rel_disp_map.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        metric_vals, rel_vals, fg_flags = [], [], []

        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            p3d = reconstruction.points3D[p2d.point3D_id].xyz
            p_cam = transform_to_camera_coords(image, p3d)
            metric_z = p_cam[2]
            if metric_z <= 0.1:  # Behind or directly on camera plane
                continue

            u, v = int(p2d.xy[0]), int(p2d.xy[1])
            if 0 <= v < rel_disp_map.shape[0] and 0 <= u < rel_disp_map.shape[1]:
                # Exclude anchors that erroneously fall in sky regions
                if sky_resized[v, u] > 128:
                    continue

                rel_vals.append(rel_disp_map[v, u])
                metric_vals.append(metric_z)
                fg_flags.append(fg_resized[v, u] > 128)

        # ── D. Foreground-Prioritized RANSAC depth alignment ──────
        aligned = False
        if len(metric_vals) >= 4:
            inv_rel = 1.0 / (np.array(rel_vals) + 1e-4)
            metric_arr = np.array(metric_vals)
            fg_arr = np.array(fg_flags)

            s, t = None, None
            # Stage 1: Prioritize foreground anchors to avoid scale contamination from background
            if fg_arr.sum() >= 4:
                s, t, inlier_mask = ransac_depth_fit(inv_rel[fg_arr], metric_arr[fg_arr])
                num_fg_inliers = int(inlier_mask.sum())
                # If foreground fit succeeds with good inliers
                if s is not None and s > 0 and num_fg_inliers >= 3:
                    aligned = True
                    align_type = f"FG-Prioritized ({num_fg_inliers}/{int(fg_arr.sum())} inliers)"

            # Stage 2: Fall back to all non-sky inliers if foreground alone was insufficient
            if not aligned:
                s, t, inlier_mask = ransac_depth_fit(inv_rel, metric_arr)
                if s is not None and s > 0:
                    aligned = True
                    align_type = f"All-Inlier RANSAC ({int(inlier_mask.sum())}/{len(metric_vals)} inliers)"

            if aligned and s is not None and s > 0:
                metric_depth_map = s / (rel_disp_map + 1e-4) + t
                metric_depth_map = np.clip(metric_depth_map, 0.1, 250.0)

                # CRITICAL: Eliminate sky geometry at source by zeroing out sky depth
                metric_depth_map[sky_resized > 128] = 0.0

                np.save(os.path.join(depth_dir, f"{image_name}.npy"), metric_depth_map)
                print(f"  ✓ {image_name} | {align_type} | s={s:.3f} t={t:.3f}")
                stats[image_name] = {"anchors": len(metric_vals), "s": s, "t": t, "type": align_type}
                continue

        # ── E. Fallback (too few anchors / RANSAC failure) ────────
        norm_d = (rel_disp_map - rel_disp_map.min()) / (np.ptp(rel_disp_map) + 1e-5)
        # Invert disparity for metric depth fallback: closer = 5m, farther = 35m
        metric_depth_map = 5.0 + (1.0 - norm_d) * 30.0
        metric_depth_map[sky_resized > 128] = 0.0
        np.save(os.path.join(depth_dir, f"{image_name}.npy"), metric_depth_map)
        print(f"  ⚠ {image_name} | {len(metric_vals)} anchors — using normalized fallback depth")
        stats[image_name] = {"anchors": len(metric_vals), "fallback": True}

    # Save depth fusion metadata
    meta_path = os.path.join(dataset_dir, "depth_fusion_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Segmentation masks → {fg_dir}, {sky_dir}")
    print(f"✓ Metric depths      → {depth_dir}")


if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    generate_metric_depths(dataset_dir)