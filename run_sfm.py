import os
import pycolmap
from pathlib import Path

def run_sparse_reconstruction(dataset_dir):
    dataset_path = Path(dataset_dir)
    image_dir = dataset_path / "images"
    database_path = dataset_path / "database.db"
    sfm_output_path = dataset_path / "sparse/0" 
    
    sfm_output_path.mkdir(parents=True, exist_ok=True)
    
    if database_path.exists():
        database_path.unlink()

    print("1. Extracting Features...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
    )

    print("2. Matching Features...")
    pycolmap.match_exhaustive(database_path)

    print("3. Running Incremental SfM with Relaxed Priors...")
    # Initialize the main pipeline options
    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    pipeline_opts.min_num_matches = 15
    
    # Initialize the specific mapper options and inject them into the pipeline
    mapper_opts = pycolmap.IncrementalMapperOptions()
    mapper_opts.init_min_tri_angle = 2.0 # Allow flatter initial triangulation
    pipeline_opts.mapper = mapper_opts   # Nest the options correctly
    
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sfm_output_path.parent,
        options=pipeline_opts
    )
    
    if not maps:
        print("\nReconstruction failed! Parallax is still too low.")
        return
        
    reconstruction = maps[0]
    print(f"\nReconstruction complete! Generated {len(reconstruction.points3D)} sparse points.")
    
    reconstruction.write_text(str(sfm_output_path))
    print(f"Poses saved to {sfm_output_path}")

run_sparse_reconstruction("./dataset")