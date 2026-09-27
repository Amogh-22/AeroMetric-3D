# AeroMetric-3D

**Single-Pass Drone Video to Accurate 3D Model Generation System**

> SIH 2026 — Problem Statement SIH26158 — NTRO — Robotics and Drones

## Overview

AeroMetric-3D reconstructs a **metrically accurate, confidence-aware 3D digital twin** from a **single forward-moving UAV video pass**.

The core challenge: a forward UAV flight provides many frames along the flight direction but very little cross-track viewpoint diversity, making conventional multi-view stereo unreliable. AeroMetric-3D overcomes this by fusing:

- **Sparse metric geometry** from Structure-from-Motion (COLMAP)
- **Dense learned depth** from Depth Anything V2
- **Metric scale alignment** via SfM↔AI depth fusion
- **Confidence-aware classification** of reconstructed geometry

```
UAV Video → Frame Selection → SfM → AI Depth → Metric Fusion → 3D Digital Twin
```

## Technical USP

AeroMetric-3D combines sparse metric geometric constraints from SfM and UAV telemetry with dense learned depth to overcome the limited-view geometry of a single forward UAV pass.

**Use classical geometry where geometric evidence exists, and AI to fill where classical reconstruction is weak.**

## Architecture

```
VIDEO + TELEMETRY
       ↓
PREPROCESSING (OpenCV + YOLO11)
  Frame extraction, blur rejection, dynamic object masking
       ↓
METRIC SKELETON (PyCOLMAP)
  Feature extraction, matching, SfM, camera poses, sparse 3D
       ↓
AI DENSE DEPTH (Depth Anything V2)
  Dense monocular relative depth estimation
       ↓
METRIC DEPTH FUSION
  Scale/shift alignment: Z_metric = s·(1/Z_AI) + t
  RANSAC-robust estimation using SfM correspondences
       ↓
DENSE MESH GENERATION (NumPy + Trimesh)
  Depth back-projection, sky masking, triangle regularization
       ↓
CONFIDENCE-AWARE DIGITAL TWIN
  Class 1 (Green)  — Geometrically verified (near SfM points)
  Class 2 (Yellow) — AI-supported
  Class 3 (Red)    — Inferred / AI-only
```

## Quick Start

### Prerequisites

- Python 3.13+
- macOS ARM64 (Apple Silicon) — uses MPS backend for PyTorch
- ~4GB disk space for models and outputs

### Installation

```bash
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python3 run_pipeline.py flight_pass.mp4 ./dataset
```

Or run stages individually:

```bash
# 1. Extract frames and create dynamic-object masks
python3 preprocess_drone_video.py

# 2. Run Structure-from-Motion
python3 run_sfm.py

# 3. Generate metric depth maps
python3 metric_depth_fusion.py

# 4. Build dense 3D mesh
python3 generate_mesh.py
```

### Outputs

| File | Description |
|------|-------------|
| `dataset/metric_reconstruction.ply` | Dense textured 3D mesh (PLY) |
| `dataset/metric_reconstruction.glb` | Dense textured 3D mesh (GLB — web-ready) |
| `dataset/confidence_reconstruction.ply` | Confidence-colored mesh |
| `dataset/sfm_metadata.json` | SfM reconstruction statistics |
| `dataset/depth_fusion_metadata.json` | Per-frame scale/shift parameters |
| `dataset/mesh_metadata.json` | Mesh generation statistics |

## Key Design Decisions

### Forward-Flight SfM Relaxation
Standard photogrammetry parameters reject the weak geometry from a forward flight. AeroMetric-3D uses:
- `init_min_tri_angle = 1.0°` (default: 16°)
- `init_max_forward_motion = 0.99` (default: 0.95)
- Shared single camera model across all frames
- Two-view track triangulation enabled

### Metric Scale Fusion
Monocular depth from Depth Anything V2 is **relative, not metric**. The fusion step aligns AI depth with SfM geometry using RANSAC-robust least-squares:

```
Z_metric = s · (1 / (Z_AI + ε)) + t
```

Scale (s) and shift (t) are estimated per-frame from SfM↔depth correspondences.

### Confidence Classification
Not a cosmetic heatmap — confidence is based on actual proximity to SfM geometric evidence:
- **Verified**: vertex within median/2 distance of an SfM point
- **AI-Supported**: within 2× median distance
- **Inferred**: beyond 2× median distance

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| SfM | PyCOLMAP 4.2 |
| AI Depth | Depth Anything V2 (Small) |
| Object Detection | YOLO11 |
| Mesh | Trimesh + NumPy |
| Computer Vision | OpenCV |
| Inference | PyTorch (MPS backend) |

## Project Structure

```
AeroMetric-3D/
├── preprocess_drone_video.py   # Frame extraction + YOLO masking
├── run_sfm.py                  # Structure-from-Motion pipeline
├── metric_depth_fusion.py      # AI depth + SfM metric alignment
├── generate_mesh.py            # Dense mesh generation + confidence
├── run_pipeline.py             # Full pipeline orchestrator
├── parse_telemetry.py          # GPS/IMU → ENU conversion
├── flight_pass.mp4             # Input UAV video
├── yolo11n-seg.pt              # YOLO segmentation model
├── requirements.txt            # Python dependencies
└── dataset/
    ├── images/                 # Extracted keyframes
    ├── masks/                  # Dynamic-object masks
    ├── sparse/0/               # SfM reconstruction
    ├── metric_depths/          # Scaled depth maps (.npy)
    ├── metric_reconstruction.ply
    ├── metric_reconstruction.glb
    └── confidence_reconstruction.ply
```

## Roadmap

- [x] Video preprocessing + YOLO masking
- [x] Forward-flight SfM with relaxed parameters
- [x] AI depth estimation (Depth Anything V2)
- [x] Metric scale fusion (RANSAC)
- [x] Dense mesh generation with sky masking
- [x] Confidence-aware classification
- [ ] Full-video reconstruction validation
- [ ] Geospatial referencing (WGS84/UTM)
- [ ] FastAPI backend
- [ ] Next.js + Three.js web viewer
- [ ] Nerfstudio/3DGS photorealistic layer
- [ ] Quantitative evaluation benchmarks

## License

This project is developed for Smart India Hackathon 2026.
