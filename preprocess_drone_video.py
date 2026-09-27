import cv2
import numpy as np
import os
from ultralytics import YOLO

def process_drone_video(video_path, output_images_dir, output_masks_dir, target_fps=3, blur_thresh=10.0):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Load YOLO11 (runs efficiently on Apple MPS or CPU)
    model = YOLO("yolo11n-seg.pt")
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps // target_fps))
    
    frame_idx = 0
    saved_idx = 0
    
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
                # 3. Dynamic Object Masking (Vehicles, People)
                # Classes: 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
                results = model(frame, classes=[0, 2, 3, 5, 7], verbose=False)
                
                # Initialize a pure white mask (255 = static geometry)
                h, w = frame.shape[:2]
                mask = np.ones((h, w), dtype=np.uint8) * 255
                
                # If dynamic objects are found, paint them black (0 = ignore)
                if results[0].masks is not None:
                    for seg in results[0].masks.data:
                        # Resize YOLO mask back to original frame dimensions
                        seg_resized = cv2.resize(seg.cpu().numpy(), (w, h))
                        mask[seg_resized > 0.5] = 0
                
                # Save the validated frame and its corresponding SfM mask
                base_name = f"frame_{saved_idx:05d}"
                cv2.imwrite(f"{output_images_dir}/{base_name}.jpg", frame)
                cv2.imwrite(f"{output_masks_dir}/{base_name}.png", mask)
                
                saved_idx += 1
                
        frame_idx += 1
        
    cap.release()

    # Safety fallback: if too few frames passed blur threshold, re-extract frames cleanly
    if saved_idx < 5:
        print(f"  ⚠️ Warning: Only {saved_idx} frames passed blur filter. Extracting sampled frames directly...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                h, w = frame.shape[:2]
                mask = np.ones((h, w), dtype=np.uint8) * 255
                base_name = f"frame_{saved_idx:05d}"
                cv2.imwrite(f"{output_images_dir}/{base_name}.jpg", frame)
                cv2.imwrite(f"{output_masks_dir}/{base_name}.png", mask)
                saved_idx += 1
            frame_idx += 1
        cap.release()

    print(f"Extraction complete. Retained {saved_idx} crisp frames with dynamic masks.")

# Execution
if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "flight_pass.mp4"
    frames_dir = sys.argv[2] if len(sys.argv) > 2 else "./frames"
    masks_dir = sys.argv[3] if len(sys.argv) > 3 else "./masks"
    process_drone_video(video, frames_dir, masks_dir)