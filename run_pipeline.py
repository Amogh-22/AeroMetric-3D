"""
AeroMetric-3D — Full Pipeline Orchestrator
==========================================
Runs the complete reconstruction pipeline in sequence:
  1. Video Preprocessing (frame extraction + YOLO masking)
  2. Structure from Motion (SfM)
  3. Metric Depth Fusion (AI depth + SfM alignment)
  4. Dense Mesh Generation (back-projection + meshing)

Usage:
  python run_pipeline.py [video_path] [dataset_dir]

  Defaults:
    video_path  = ./flight_pass.mp4
    dataset_dir = ./dataset
"""

import os
# CRITICAL: Prevent pipe deadlocks with \r progress bars when piped to Colab's Flask server
os.environ["TQDM_DISABLE"] = "1" 
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime


def run_pipeline(video_path="./flight_pass.mp4", dataset_dir="./dataset"):
    """Execute the full AeroMetric-3D reconstruction pipeline."""

    dataset_path = Path(dataset_dir).resolve()
    video_path = Path(video_path).resolve()
    script_dir = Path(__file__).parent.resolve()

    print("=" * 70)
    print("  AeroMetric-3D — Full Reconstruction Pipeline")
    print("=" * 70)
    print(f"  Video:   {video_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    def update_progress(msg):
        try:
            with open(dataset_path / "geospatial_metadata.json", "w") as f:
                json.dump({"progress": msg}, f)
        except Exception:
            pass
            
    update_progress("Initializing pipeline...")

    pipeline_start = time.time()
    stage_times = {}

    # ==================================================================
    # STAGE 1: Video Preprocessing
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STAGE 1/4: VIDEO PREPROCESSING")
    print("▓" * 70)
    update_progress("Stage 1/5: Extracting frames & YOLO dynamic masking...")

    if not video_path.exists():
        # Check if video exists with spaces, or find video file in parent/workspace directory
        resolved = False
        search_dirs = [video_path.parent, dataset_path.parent, dataset_path, script_dir, Path(".")]
        for sdir in search_dirs:
            if sdir and sdir.exists():
                vids = list(sdir.glob("*.mp4")) + list(sdir.glob("*.mov")) + list(sdir.glob("*.avi"))
                if vids:
                    video_path = vids[0].resolve()
                    print(f"  ✓ Auto-resolved video file from workspace: {video_path}")
                    resolved = True
                    break
        if not resolved:
            error_msg = f"ERROR: Video file not found: {video_path}"
            print(error_msg)
            with open(dataset_path / "geospatial_metadata.json", "w") as f:
                json.dump({"error": error_msg}, f)
            return False

    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"

    # Check if frames already exist
    existing_frames = list(images_dir.glob("*.jpg")) if images_dir.exists() else []
    if existing_frames:
        print(f"Found {len(existing_frames)} existing frames — skipping extraction.")
        print("(Delete dataset/images/ to re-extract)")
    else:
        import subprocess
        print("  Running preprocess_drone_video.py in isolated process...")
        t0 = time.time()
        result = subprocess.run([sys.executable, str(script_dir / "preprocess_drone_video.py"), str(video_path), str(images_dir), str(masks_dir)], capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = "ERROR: Preprocessing failed:\n" + result.stderr
            print(error_msg)
            with open(dataset_path / "geospatial_metadata.json", "w") as f:
                json.dump({"error": error_msg}, f)
            return False
        stage_times["preprocessing"] = time.time() - t0
        print(f"Preprocessing completed in {stage_times['preprocessing']:.1f}s")

    # Verify output
    n_frames = len(list(images_dir.glob("*.jpg")))
    n_masks = len(list(masks_dir.glob("*.png")))
    print(f"\n  Frames: {n_frames}")
    print(f"  Masks:  {n_masks}")

    if n_frames == 0:
        print("ERROR: No frames extracted!")
        return False

    # ==================================================================
    # STAGE 2: Structure from Motion
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STAGE 2/4: STRUCTURE FROM MOTION")
    print("▓" * 70)
    update_progress("Stage 2/5: Structure from Motion (pycolmap)... this takes 1-2 minutes.")

    t0 = time.time()
    import subprocess
    print("  Running run_sfm.py in isolated process...")
    result = subprocess.run([sys.executable, str(script_dir / "run_sfm.py"), dataset_dir], capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = "ERROR: SfM failed — cannot continue pipeline:\n" + result.stderr
        print(error_msg)
        with open(dataset_path / "geospatial_metadata.json", "w") as f:
            json.dump({"error": error_msg}, f)
        return False
    stage_times["sfm"] = time.time() - t0

    import pycolmap
    sparse_path = dataset_path / "sparse" / "0"
    if not sparse_path.exists():
        print("ERROR: No SfM model found at output directory.")
        return False

    reconstruction = pycolmap.Reconstruction(str(sparse_path))
    n_registered = len(reconstruction.images)
    n_points = len(reconstruction.points3D)
    reg_ratio = n_registered / n_frames * 100
    print(f"\n  Registered: {n_registered}/{n_frames} frames ({reg_ratio:.0f}%)")
    print(f"  3D Points:  {n_points}")

    if n_registered < 2:
        print("ERROR: Fewer than 2 frames registered — cannot produce depth")
        return False

    # ==================================================================
    # STAGE 3: Metric Depth Fusion
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STAGE 3/4: METRIC DEPTH FUSION")
    print("▓" * 70)
    update_progress("Stage 3/5: AI Metric Depth Fusion (Downloading/Running Depth Anything V2)...")

    t0 = time.time()
    print("  Running metric_depth_fusion.py in isolated process...")
    result = subprocess.run([sys.executable, str(script_dir / "metric_depth_fusion.py"), dataset_dir], capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = "ERROR: Metric Depth Fusion failed:\n" + result.stderr
        print(error_msg)
        with open(dataset_path / "geospatial_metadata.json", "w") as f:
            json.dump({"error": error_msg}, f)
        return False
    stage_times["depth_fusion"] = time.time() - t0

    # ==================================================================
    # STAGE 4: Dense Mesh Generation
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STAGE 4/4: DENSE MESH GENERATION")
    print("▓" * 70)
    update_progress("Stage 4/5: Dense Mesh Generation (Back-projecting and Meshing)...")

    t0 = time.time()
    print("  Running generate_mesh.py in isolated process...")
    result = subprocess.run([sys.executable, str(script_dir / "generate_mesh.py"), dataset_dir], capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = "ERROR: Mesh Generation failed:\n" + result.stderr
        print(error_msg)
        with open(dataset_path / "geospatial_metadata.json", "w") as f:
            json.dump({"error": error_msg}, f)
        return False
    stage_times["mesh_generation"] = time.time() - t0

    # ==================================================================
    # STAGE 5: Geospatial Export
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STAGE 5/5: GEOSPATIAL PROJECTION & EXPORT")
    print("▓" * 70)
    update_progress("Stage 5/5: Geospatial Export (Applying UTM Projection)...")

    t0 = time.time()
    print("  Running geospatial_export.py in isolated process...")
    result = subprocess.run([sys.executable, str(script_dir / "geospatial_export.py"), dataset_dir, str(video_path)], capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = "ERROR: Geospatial Export failed:\n" + result.stderr
        print(error_msg)
        with open(dataset_path / "geospatial_metadata.json", "w") as f:
            json.dump({"error": error_msg}, f)
        return False
    stage_times["geospatial_export"] = time.time() - t0

    # ==================================================================
    # SUMMARY
    # ==================================================================
    total_time = time.time() - pipeline_start
    stage_times["total"] = total_time

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
    for stage, t in stage_times.items():
        if stage != "total":
            print(f"    {stage:20s} {t:.1f}s")
    print()

    # List outputs
    output_files = [
        dataset_path / "metric_reconstruction.ply",
        dataset_path / "metric_reconstruction.glb",
        dataset_path / "metric_reconstruction.obj",
        dataset_path / "confidence_reconstruction.ply",
        dataset_path / "metric_reconstruction.las",
        dataset_path / "orthomosaic_dsm.tif",
        dataset_path / "sfm_metadata.json",
        dataset_path / "depth_fusion_metadata.json",
        dataset_path / "mesh_metadata.json",
        dataset_path / "geospatial_metadata.json",
    ]

    print("  Outputs:")
    for f in output_files:
        if f.exists():
            size_mb = f.stat().st_size / 1e6
            print(f"    ✓ {f.name:40s} ({size_mb:.1f} MB)")
        else:
            print(f"    ✗ {f.name:40s} (not generated)")

    print("=" * 70)

    # Save pipeline metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "video": str(video_path),
        "dataset": str(dataset_path),
        "frames_extracted": n_frames,
        "frames_registered": n_registered,
        "sparse_points": n_points,
        "stage_times": stage_times,
    }
    meta_path = dataset_path / "pipeline_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return True

def run_pipeline_with_error_reporting(video_path, dataset_dir):
    dataset_path = Path(dataset_dir).resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)
    try:
        success = run_pipeline(video_path, dataset_dir)
        if not success:
            meta_file = dataset_path / "geospatial_metadata.json"
            has_err = False
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        if "error" in json.load(f):
                            has_err = True
                except:
                    pass
            if not has_err:
                with open(meta_file, "w") as f:
                    json.dump({"error": "Pipeline returned False"}, f)
        return success
    except Exception as e:
        import traceback
        with open(dataset_path / "geospatial_metadata.json", "w") as f:
            json.dump({"error": "Pipeline exception", "traceback": traceback.format_exc()}, f)
        return False


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        video = "./flight_pass.mp4"
        dataset = "./dataset"
    elif len(sys.argv) == 2:
        video = sys.argv[1]
        dataset = "./dataset"
    elif len(sys.argv) == 3:
        video = sys.argv[1]
        dataset = sys.argv[2]
    else:
        # If shell split unquoted video path containing spaces:
        # Last argument is dataset_dir, all prior arguments form the video path
        dataset = sys.argv[-1]
        video = " ".join(sys.argv[1:-1])

    success = run_pipeline_with_error_reporting(video, dataset)
    sys.exit(0 if success else 1)
