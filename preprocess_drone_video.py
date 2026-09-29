import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import argparse
import sys
from ultralytics import YOLO

def detect_sky_hsv(frame):
    """
    Detect sky regions in aerial/drone imagery using HSV color space,
    gradient uniformity, and top-boundary connectivity.
    Returns uint8 mask (255 = sky, 0 = non-sky).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 1. Color heuristics:
    # Blue sky: Hue 95-135, Saturation >= 20, Value >= 80
    blue_sky = (H >= 95) & (H <= 135) & (S >= 20) & (V >= 80)
    # Overcast / white / pale sky: low saturation, high brightness
    white_sky = (S <= 45) & (V >= 175)

    candidate = (blue_sky | white_sky).astype(np.uint8) * 255

    # 2. Gradient uniformity: sky has minimal texture/gradient
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1)
    candidate[np.abs(sobel) >= 25] = 0

    # 3. Top-edge connectivity: sky starts at the top frame boundary
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, max(1, w // 40)):
        if candidate[0, x] == 255 and flood_mask[1, x + 1] == 0:
            cv2.floodFill(candidate, flood_mask, (x, 0), 128)

    sky_mask = (candidate == 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    return sky_mask


def generate_initial_fg_mask(frame, sky_mask, dynamic_mask, roi_prompt=None):
    """
    Generate initial foreground guidance mask prioritizing the primary subject.
    Centered Gaussian prior combined with non-sky, non-dynamic region.
    """
    h, w = frame.shape[:2]
    non_sky = (sky_mask < 128) & (dynamic_mask > 128)

    # Subject in drone footage is predominantly centered
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    sig_y, sig_x = h * 0.35, w * 0.35
    center_w = np.exp(-((Y - cy) ** 2 / (2 * sig_y ** 2) + (X - cx) ** 2 / (2 * sig_x ** 2)))

    # Foreground candidate is center-weighted within non-sky static areas
    fg_score = center_w * non_sky.astype(np.float32)
    fg_mask = (fg_score > 0.35).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def process_drone_video(
    video_path,
    output_images_dir,
    output_masks_dir,
    target_fps=3,
    blur_thresh=10.0,
    output_fg_masks_dir=None,
    output_sky_masks_dir=None,
    roi_prompt=None
):
    """
    Preprocess drone video:
      - Sharpness filtering (blur detection via Laplacian variance)
      - YOLO dynamic object masking (vehicles, people)
      - Sky detection & elimination
      - Foreground ROI candidate mask extraction
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    if output_fg_masks_dir:
        os.makedirs(output_fg_masks_dir, exist_ok=True)
    if output_sky_masks_dir:
        os.makedirs(output_sky_masks_dir, exist_ok=True)

    # Load YOLO11 for dynamic object segmentation
    model = YOLO("yolo11n-seg.pt")

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(video_fps // target_fps))

    frame_idx = 0
    saved_idx = 0

    print(f"Ingesting video {video_path} (FPS: {video_fps:.1f}, sampling every {interval} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            # 1. Blur Detection using Variance of the Laplacian
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 2. Reject if below sharpness threshold
            if variance > blur_thresh:
                h, w = frame.shape[:2]

                # 3. Sky Detection
                sky_mask = detect_sky_hsv(frame)

                # 4. Dynamic Object Masking (Vehicles, People)
                # Classes: 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
                results = model(frame, classes=[0, 2, 3, 5, 7], verbose=False)

                # Dynamic mask: 255 = static geometry, 0 = dynamic object
                dyn_mask = np.ones((h, w), dtype=np.uint8) * 255
                if results and results[0].masks is not None:
                    for seg in results[0].masks.data:
                        seg_resized = cv2.resize(seg.cpu().numpy(), (w, h))
                        dyn_mask[seg_resized > 0.5] = 0

                # 5. Combined SfM mask:
                # 255 = valid feature region (static structure + ground)
                # 0 = excluded from SfM feature extraction (sky + dynamic objects)
                sfm_mask = dyn_mask.copy()
                sfm_mask[sky_mask > 128] = 0

                # 6. Foreground ROI guidance mask
                fg_mask = generate_initial_fg_mask(frame, sky_mask, dyn_mask, roi_prompt)

                # Save artifacts
                base_name = f"frame_{saved_idx:05d}"
                cv2.imwrite(os.path.join(output_images_dir, f"{base_name}.jpg"), frame)
                cv2.imwrite(os.path.join(output_masks_dir, f"{base_name}.png"), sfm_mask)

                if output_sky_masks_dir:
                    cv2.imwrite(os.path.join(output_sky_masks_dir, f"{base_name}.png"), sky_mask)
                if output_fg_masks_dir:
                    cv2.imwrite(os.path.join(output_fg_masks_dir, f"{base_name}.png"), fg_mask)

                saved_idx += 1

        frame_idx += 1

    cap.release()

    # Safety fallback: if too few frames passed blur threshold, re-extract frames cleanly
    if saved_idx < 5:
        print(f"  ⚠️ Warning: Only {saved_idx} frames passed blur filter. Extracting sampled frames directly...")
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                h, w = frame.shape[:2]
                sky_mask = detect_sky_hsv(frame)
                sfm_mask = np.ones((h, w), dtype=np.uint8) * 255
                sfm_mask[sky_mask > 128] = 0
                fg_mask = generate_initial_fg_mask(frame, sky_mask, sfm_mask, roi_prompt)

                base_name = f"frame_{saved_idx:05d}"
                cv2.imwrite(os.path.join(output_images_dir, f"{base_name}.jpg"), frame)
                cv2.imwrite(os.path.join(output_masks_dir, f"{base_name}.png"), sfm_mask)
                if output_sky_masks_dir:
                    cv2.imwrite(os.path.join(output_sky_masks_dir, f"{base_name}.png"), sky_mask)
                if output_fg_masks_dir:
                    cv2.imwrite(os.path.join(output_fg_masks_dir, f"{base_name}.png"), fg_mask)
                saved_idx += 1
            frame_idx += 1
        cap.release()

    print(f"Extraction complete. Retained {saved_idx} crisp frames with dynamic & sky masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess drone video for 3D reconstruction")
    parser.add_argument("video", nargs="?", default="flight_pass.mp4", help="Input video path")
    parser.add_argument("images_dir", nargs="?", default="./frames", help="Output images directory")
    parser.add_argument("masks_dir", nargs="?", default="./masks", help="Output masks directory")
    parser.add_argument("--fg_dir", default=None, help="Output foreground masks directory")
    parser.add_argument("--sky_dir", default=None, help="Output sky masks directory")
    parser.add_argument("--roi_prompt", default=None, help="ROI text prompt (e.g. monument, statue)")
    parser.add_argument("--fps", type=float, default=3.0, help="Target FPS sampling rate")
    parser.add_argument("--blur_thresh", type=float, default=10.0, help="Blur variance threshold")

    args = parser.parse_args()

    # Automatic inference of fg_dir / sky_dir if dataset parent directory detected
    fg_dir = args.fg_dir
    sky_dir = args.sky_dir
    if not fg_dir and args.masks_dir:
        parent = os.path.dirname(os.path.abspath(args.masks_dir))
        fg_candidate = os.path.join(parent, "fg_masks")
        sky_candidate = os.path.join(parent, "sky_masks")
        if os.path.basename(args.masks_dir) == "masks":
            fg_dir = fg_candidate
            sky_dir = sky_candidate

    process_drone_video(
        video_path=args.video,
        output_images_dir=args.images_dir,
        output_masks_dir=args.masks_dir,
        target_fps=args.fps,
        blur_thresh=args.blur_thresh,
        output_fg_masks_dir=fg_dir,
        output_sky_masks_dir=sky_dir,
        roi_prompt=args.roi_prompt
    )