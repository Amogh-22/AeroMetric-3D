"""
AeroMetric-3D — Geospatial Export Pipeline (Stage 5)
======================================================
Projects local metric coordinates into a real-world UTM
coordinate reference system using a synthesized GPS anchor.
Exports the result as a LAS point cloud and a GeoTIFF DSM.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
import trimesh

try:
    import laspy
    from pyproj import Proj, Transformer
    import rasterio
    from rasterio.transform import from_origin
except ImportError as e:
    print(f"ERROR: Missing geospatial dependencies. Please run:\n  pip install laspy pyproj rasterio")
    sys.exit(1)


# Default synthesized GPS anchor (Noida, UP - SIH default location) if metadata is missing
DEFAULT_ANCHOR_LAT = 28.535517
DEFAULT_ANCHOR_LON = 77.391029
DEFAULT_ANCHOR_ALT = 200.0

def extract_gps_from_video(video_path):
    """Attempt to extract GPS coordinates from video metadata using ffprobe."""
    import subprocess
    try:
        # ffprobe command to dump format and stream tags
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Check for location in format tags
            if "format" in data and "tags" in data["format"]:
                tags = data["format"]["tags"]
                if "location" in tags:
                    loc = tags["location"]
                    # Location string format varies, typically +Lat+Lon+Alt/
                    # Let's do a simple regex or parsing if needed, but for now we try to parse standard ISO6709
                    import re
                    match = re.search(r'([+-][0-9.]+)([+-][0-9.]+)([+-][0-9.]+)?', loc)
                    if match:
                        lat = float(match.group(1))
                        lon = float(match.group(2))
                        alt = float(match.group(3)) if match.group(3) else DEFAULT_ANCHOR_ALT
                        return lat, lon, alt
    except Exception as e:
        print(f"  ⚠️ Warning: Failed to extract GPS via ffprobe: {e}")
    return None

def get_anchor(video_path, dataset_path):
    """Determine the GPS anchor for georeferencing."""
    # 1. Try extracting from video
    if video_path and Path(video_path).exists():
        gps = extract_gps_from_video(video_path)
        if gps:
            print(f"  ✓ Found GPS in video metadata: {gps}")
            return gps
            
    # 2. Try exif_overrides.json
    overrides_path = Path("exif_overrides.json")
    if overrides_path.exists():
        try:
            with open(overrides_path, 'r') as f:
                overrides = json.load(f)
                # Just take the first frame's GPS if it exists
                if len(overrides) > 0:
                    first_frame = list(overrides.keys())[0]
                    data = overrides[first_frame]
                    lat = data.get("gps_latitude", DEFAULT_ANCHOR_LAT)
                    lon = data.get("gps_longitude", DEFAULT_ANCHOR_LON)
                    alt = data.get("altitude", DEFAULT_ANCHOR_ALT)
                    print(f"  ✓ Found GPS in exif_overrides.json: {(lat, lon, alt)}")
                    return lat, lon, alt
        except Exception:
            pass
            
    # 3. Fallback to default
    print("  ⚠️ Using default fallback GPS anchor (Noida).")
    return DEFAULT_ANCHOR_LAT, DEFAULT_ANCHOR_LON, DEFAULT_ANCHOR_ALT



def create_las(points, colors, out_path, utm_offset_x, utm_offset_y, utm_zone):
    """Export points to LAS format with UTM coordinates."""
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0.0, 0.0, 0.0]
    header.scales = [0.01, 0.01, 0.01]  # Centimeter precision
    
    # 2. Create a LasData object
    las = laspy.LasData(header)
    
    # 3. Assign coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    # 4. Assign colors (LAS expects 16-bit colors, so scale 0-255 to 0-65535)
    las.red = (colors[:, 0].astype(np.uint16) * 256)
    las.green = (colors[:, 1].astype(np.uint16) * 256)
    las.blue = (colors[:, 2].astype(np.uint16) * 256)
    
    # 5. Write
    las.write(str(out_path))
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  ✓ LAS Point Cloud: {out_path.name} ({size_mb:.1f} MB)")


def create_dsm_geotiff(points, out_path, utm_offset_x, utm_offset_y, crs_epsg, resolution=0.5):
    """
    Rasterize point cloud into a Digital Surface Model (DSM) and save as GeoTIFF.
    resolution: pixel size in meters (default 0.5m).
    """
    if len(points) == 0:
        return
        
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    
    # Grid dimensions
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    
    if width <= 0 or height <= 0 or width > 10000 or height > 10000:
        print(f"  ⚠️ Warning: Invalid or excessively large DSM grid ({width}x{height}). Skipping GeoTIFF.")
        return
        
    # Rasterize: take maximum Z value in each grid cell
    dsm = np.full((height, width), -9999.0, dtype=np.float32)
    
    # Map points to grid indices
    # Y is inverted in images (top is max_y)
    col = ((x_coords - min_x) / resolution).astype(np.int32)
    row = ((max_y - y_coords) / resolution).astype(np.int32)
    
    # Ensure indices are within bounds
    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)
    
    # Naive rasterization (takes max Z)
    # For a large number of points, a vectorized approach using np.maximum.at is fastest
    np.maximum.at(dsm, (row, col), z_coords)
    
    # Setup geotransform (top-left X, X pixel size, rotation, top-left Y, rotation, Y pixel size)
    transform = from_origin(min_x, max_y, resolution, resolution)
    
    with rasterio.open(
        str(out_path),
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dsm.dtype,
        crs=f'EPSG:{crs_epsg}',
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(dsm, 1)
        
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  ✓ GeoTIFF DSM:     {out_path.name} ({size_mb:.1f} MB)")


def run_geospatial_export(dataset_dir, video_path=None):
    """Project the metric mesh into a geospatial coordinate system."""
    dataset_path = Path(dataset_dir)
    mesh_path = dataset_path / "metric_reconstruction.ply"
    
    print("\n" + "=" * 60)
    print("GEOSPATIAL PROJECTION & EXPORT")
    print("=" * 60)
    
    if not mesh_path.exists():
        print(f"ERROR: No mesh found at {mesh_path}")
        return False
        
    print("1. Loading Metric Mesh...")
    mesh = trimesh.load(mesh_path)
    vertices = mesh.vertices
    if hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3]
    else:
        colors = np.ones((len(vertices), 3), dtype=np.uint8) * 128
        
    print(f"   Loaded {len(vertices):,} vertices")
    
    print(f"\n2. Determining GPS Anchor...")
    ANCHOR_LAT, ANCHOR_LON, ANCHOR_ALT_METERS = get_anchor(video_path, dataset_path)
    print(f"   Anchor: {ANCHOR_LAT}°N, {ANCHOR_LON}°E, Alt {ANCHOR_ALT_METERS}m")
    
    # Setup coordinate transforms (WGS84 -> UTM Zone 43N for Noida/Delhi)
    # Determine UTM zone from longitude: zone = floor((lon + 180) / 6) + 1
    utm_zone = int(np.floor((ANCHOR_LON + 180) / 6.0)) + 1
    crs_epsg = 32600 + utm_zone # 32600 for Northern Hemisphere
    
    transformer = Transformer.from_crs("epsg:4326", f"epsg:{crs_epsg}", always_xy=True)
    utm_x, utm_y = transformer.transform(ANCHOR_LON, ANCHOR_LAT)
    
    print(f"   UTM Zone {utm_zone}N (EPSG:{crs_epsg}): X={utm_x:.1f}, Y={utm_y:.1f}")
    
    # 3. Apply arbitrary rotation to make Z pointing up (SfM Z is usually forward)
    # We apply a -90 degree rotation around X axis so SfM-Z becomes Up
    rot_x = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    rotated_vertices = vertices @ rot_x.T
    
    # Shift vertices so their mean is at the anchor location
    mean_xyz = np.mean(rotated_vertices, axis=0)
    projected_vertices = rotated_vertices - mean_xyz
    
    # Apply UTM offsets
    projected_vertices[:, 0] += utm_x
    projected_vertices[:, 1] += utm_y
    projected_vertices[:, 2] += ANCHOR_ALT_METERS
    
    print("\n3. Exporting Geospatial Formats...")
    
    out_las = dataset_path / "metric_reconstruction.las"
    create_las(projected_vertices, colors, out_las, utm_x, utm_y, utm_zone)
    
    out_tif = dataset_path / "orthomosaic_dsm.tif"
    create_dsm_geotiff(projected_vertices, out_tif, utm_x, utm_y, crs_epsg, resolution=0.25)
    
    # Save metadata
    meta = {
        "anchor_lat": ANCHOR_LAT,
        "anchor_lon": ANCHOR_LON,
        "anchor_alt": ANCHOR_ALT_METERS,
        "utm_zone": utm_zone,
        "epsg": crs_epsg,
        "utm_x": utm_x,
        "utm_y": utm_y,
        "num_points": len(vertices)
    }
    with open(dataset_path / "geospatial_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print("\n============================================================")
    print("GEOSPATIAL EXPORT COMPLETE")
    print("============================================================")
    return True

if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    video = sys.argv[2] if len(sys.argv) > 2 else None
    run_geospatial_export(dataset_dir, video)
