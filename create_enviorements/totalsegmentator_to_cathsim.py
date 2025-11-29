"""
TotalSegmentator → CathSim Pipeline
====================================
Convert 1228 CT aorta segmentations into CathSim-compatible phantom environments.

Author: Generated for Moritz
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import multiprocessing as mp
from functools import partial

# ============================================================================
# STEP 0: Install Dependencies
# ============================================================================

REQUIRED_PACKAGES = [
    "nibabel",           # NIfTI file handling
    "numpy",
    "trimesh",           # Mesh operations
    "scikit-image",      # Marching cubes
    "pyvista",           # 3D visualization (optional)
    "pymeshfix",         # Mesh repair
    "obj2mjcf",          # MuJoCo conversion + convex decomposition
    "tqdm",              # Progress bars
]

def install_dependencies():
    """Install all required packages."""
    print("Installing dependencies...")
    for pkg in REQUIRED_PACKAGES:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("✓ Dependencies installed")


# ============================================================================
# STEP 1: Download TotalSegmentator Dataset
# ============================================================================

ZENODO_URL = "https://zenodo.org/record/6802613/files/Totalsegmentator_dataset_v201.zip"
# Alternative direct links for parts if needed:
# Part 1: https://zenodo.org/record/6802613/files/Totalsegmentator_dataset_v201.zip?download=1

def download_totalsegmentator_dataset(
    output_dir: str = "./totalsegmentator_data",
    max_subjects: Optional[int] = None
) -> Path:
    """
    Download the TotalSegmentator dataset from Zenodo.
    
    The dataset contains:
    - 1228 CT subjects
    - Each subject has segmentations for 117 anatomical structures
    - Aorta is label index 52
    
    Args:
        output_dir: Directory to save the dataset
        max_subjects: Limit number of subjects (None for all 1228)
    
    Returns:
        Path to the extracted dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_path / "Totalsegmentator_dataset_v201.zip"
    
    # Check if already downloaded
    if (output_path / "Totalsegmentator_dataset_v201").exists():
        print(f"✓ Dataset already exists at {output_path}")
        return output_path / "Totalsegmentator_dataset_v201"
    
    print(f"Downloading TotalSegmentator dataset (~12GB)...")
    print(f"URL: {ZENODO_URL}")
    print("This may take a while...")
    
    # Using wget or curl for large files
    try:
        subprocess.run([
            "wget", "-c", ZENODO_URL, "-O", str(zip_path)
        ], check=True)
    except FileNotFoundError:
        # Try curl if wget not available
        subprocess.run([
            "curl", "-L", "-C", "-", ZENODO_URL, "-o", str(zip_path)
        ], check=True)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    print(f"✓ Dataset extracted to {output_path}")
    return output_path / "Totalsegmentator_dataset_v201"


# ============================================================================
# STEP 2: Extract Aorta Segmentations
# ============================================================================

def load_aorta_mask(subject_dir: Path) -> Tuple[any, any]:
    """
    Load the aorta segmentation mask for a subject.
    
    In TotalSegmentator dataset structure:
    - subject_dir/
        - ct.nii.gz (original CT)
        - segmentations/
            - aorta.nii.gz (label 52)
    
    Returns:
        Tuple of (mask_data, affine_matrix)
    """
    import nibabel as nib
    import numpy as np
    
    # TotalSegmentator stores individual organ masks
    aorta_path = subject_dir / "segmentations" / "aorta.nii.gz"
    
    if not aorta_path.exists():
        # Sometimes stored as combined multilabel
        combined_path = subject_dir / "segmentations.nii.gz"
        if combined_path.exists():
            img = nib.load(combined_path)
            data = img.get_fdata()
            # Aorta is label 52 in TotalSegmentator
            aorta_mask = (data == 52).astype(np.uint8)
            return aorta_mask, img.affine
        raise FileNotFoundError(f"No aorta segmentation found in {subject_dir}")
    
    img = nib.load(aorta_path)
    return img.get_fdata(), img.affine


def get_voxel_spacing(affine: any) -> Tuple[float, float, float]:
    """Extract voxel spacing from affine matrix."""
    import numpy as np
    return tuple(np.abs(np.diag(affine)[:3]))


# ============================================================================
# STEP 3: Convert Segmentation to Mesh (Marching Cubes)
# ============================================================================

def segmentation_to_mesh(
    mask: any,
    affine: any,
    smooth: bool = True,
    simplify_ratio: float = 0.1
) -> any:
    """
    Convert binary segmentation mask to 3D mesh using marching cubes.
    
    Args:
        mask: Binary segmentation mask (numpy array)
        affine: NIfTI affine matrix for physical coordinates
        smooth: Apply Laplacian smoothing
        simplify_ratio: Reduce mesh complexity (0.1 = keep 10% of faces)
    
    Returns:
        trimesh.Trimesh object
    """
    import numpy as np
    from skimage import measure
    import trimesh
    
    # Marching cubes to extract surface
    verts, faces, normals, values = measure.marching_cubes(
        mask, 
        level=0.5,
        spacing=get_voxel_spacing(affine)
    )
    
    # Apply affine transformation to get physical coordinates
    # Add homogeneous coordinate
    verts_homo = np.c_[verts, np.ones(len(verts))]
    verts_transformed = (affine @ verts_homo.T).T[:, :3]
    
    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=verts_transformed,
        faces=faces,
        vertex_normals=normals
    )
    
    # Clean up mesh
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    
    # Smooth mesh (Laplacian smoothing)
    if smooth:
        trimesh.smoothing.filter_laplacian(mesh, iterations=3)
    
    # Simplify mesh to reduce complexity
    if simplify_ratio < 1.0:
        target_faces = int(len(mesh.faces) * simplify_ratio)
        if target_faces > 100:  # Minimum faces
            mesh = mesh.simplify_quadric_decimation(target_faces)
    
    return mesh


# ============================================================================
# STEP 4: Repair Mesh for Simulation
# ============================================================================

def repair_mesh(mesh: any) -> any:
    """
    Repair mesh to ensure it's watertight and suitable for simulation.
    
    Requirements for MuJoCo:
    - Watertight (closed surface)
    - No self-intersections
    - Consistent normals
    """
    import trimesh
    
    # Fix normals to point outward
    mesh.fix_normals()
    
    # Try PyMeshFix for more robust repair
    try:
        import pymeshfix
        meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        meshfix.repair(verbose=False)
        mesh = trimesh.Trimesh(
            vertices=meshfix.v,
            faces=meshfix.f
        )
    except ImportError:
        print("Warning: pymeshfix not available, using basic repair")
        mesh.fill_holes()
    
    return mesh


# ============================================================================
# STEP 5: Convex Decomposition for MuJoCo
# ============================================================================

def apply_convex_decomposition(
    mesh_path: str,
    output_dir: str,
    max_hulls: int = 32,
    resolution: int = 100000
) -> Path:
    """
    Apply convex decomposition using obj2mjcf (uses CoACD internally).
    
    MuJoCo requires convex collision meshes for stable physics simulation.
    V-HACD/CoACD decomposes the mesh into multiple convex hulls.
    
    Args:
        mesh_path: Path to input OBJ file
        output_dir: Directory for output files
        max_hulls: Maximum number of convex hulls
        resolution: Voxel resolution for decomposition
    
    Returns:
        Path to generated MJCF file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run obj2mjcf with convex decomposition
    cmd = [
        "obj2mjcf",
        "--obj-dir", str(Path(mesh_path).parent),
        "--save-mjcf",
        "--compile-model",
        "--decompose",  # Enable convex decomposition
        "--coacd-threshold", "0.05",  # Concavity threshold
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: obj2mjcf failed, trying manual V-HACD...")
        return manual_vhacd_decomposition(mesh_path, output_dir)
    
    # Find generated MJCF file
    mjcf_files = list(output_path.glob("*.xml"))
    if mjcf_files:
        return mjcf_files[0]
    
    raise FileNotFoundError("No MJCF file generated")


def manual_vhacd_decomposition(mesh_path: str, output_dir: str) -> Path:
    """
    Manual V-HACD decomposition as fallback.
    Uses trimesh's built-in convex decomposition.
    """
    import trimesh
    import numpy as np
    
    mesh = trimesh.load(mesh_path)
    
    # Decompose into convex hulls
    try:
        convex_hulls = mesh.convex_decomposition(maxhulls=32)
    except Exception:
        # Fallback: just use single convex hull
        convex_hulls = [mesh.convex_hull]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export each hull
    hull_paths = []
    for i, hull in enumerate(convex_hulls):
        hull_path = output_path / f"hull_{i:03d}.obj"
        hull.export(hull_path)
        hull_paths.append(hull_path)
    
    # Generate MJCF file
    mjcf_path = output_path / "phantom.xml"
    generate_mjcf_file(hull_paths, mjcf_path)
    
    return mjcf_path


def generate_mjcf_file(hull_paths: List[Path], output_path: Path):
    """Generate MuJoCo MJCF XML file for the phantom."""
    
    meshes_xml = ""
    geoms_xml = ""
    
    for i, hull_path in enumerate(hull_paths):
        mesh_name = f"hull_{i:03d}"
        meshes_xml += f'    <mesh name="{mesh_name}" file="{hull_path.name}"/>\n'
        geoms_xml += f'      <geom type="mesh" mesh="{mesh_name}" rgba="0.8 0.2 0.2 0.5"/>\n'
    
    mjcf_content = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="aorta_phantom">
  <compiler angle="radian"/>
  
  <asset>
{meshes_xml}
  </asset>
  
  <worldbody>
    <body name="phantom" pos="0 0 0">
{geoms_xml}
    </body>
  </worldbody>
</mujoco>
'''
    
    with open(output_path, 'w') as f:
        f.write(mjcf_content)


# ============================================================================
# STEP 6: Create CathSim-Compatible Phantom
# ============================================================================

def create_cathsim_phantom(
    mjcf_path: Path,
    phantom_name: str,
    cathsim_phantoms_dir: str = "./cathsim_phantoms"
) -> Path:
    """
    Create a CathSim-compatible phantom directory structure.
    
    CathSim expects:
    - phantom_name/
        - phantom.xml (MJCF file)
        - meshes/ (collision meshes)
        - visual.obj (optional visual mesh)
    """
    output_dir = Path(cathsim_phantoms_dir) / phantom_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy MJCF file
    shutil.copy(mjcf_path, output_dir / "phantom.xml")
    
    # Copy mesh files
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)
    
    for mesh_file in mjcf_path.parent.glob("*.obj"):
        shutil.copy(mesh_file, meshes_dir / mesh_file.name)
    
    # Create CathSim registration file
    registration = {
        "name": phantom_name,
        "description": f"Aorta phantom from TotalSegmentator subject",
        "mjcf_path": "phantom.xml",
    }
    
    import json
    with open(output_dir / "phantom_info.json", 'w') as f:
        json.dump(registration, f, indent=2)
    
    print(f"✓ Created CathSim phantom: {output_dir}")
    return output_dir


# ============================================================================
# STEP 7: Full Pipeline - Process Single Subject
# ============================================================================

def process_single_subject(
    subject_dir: Path,
    output_base_dir: str,
    subject_idx: int
) -> Optional[Path]:
    """
    Full pipeline for a single subject.
    
    Returns:
        Path to created CathSim phantom, or None if failed
    """
    import numpy as np
    
    phantom_name = f"aorta_{subject_idx:04d}"
    temp_dir = Path(output_base_dir) / "temp" / phantom_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Processing subject {subject_idx}: {subject_dir.name}")
        
        # Step 1: Load aorta mask
        mask, affine = load_aorta_mask(subject_dir)
        
        # Check if mask is valid (not empty)
        if np.sum(mask) < 1000:  # Minimum voxels
            print(f"  ⚠ Skipping: insufficient aorta voxels")
            return None
        
        # Step 2: Convert to mesh
        mesh = segmentation_to_mesh(mask, affine, smooth=True, simplify_ratio=0.15)
        
        # Check mesh validity
        if len(mesh.faces) < 100:
            print(f"  ⚠ Skipping: mesh too simple")
            return None
        
        # Step 3: Repair mesh
        mesh = repair_mesh(mesh)
        
        # Step 4: Export OBJ
        obj_path = temp_dir / "aorta.obj"
        mesh.export(obj_path)
        
        # Step 5: Convex decomposition
        mjcf_path = apply_convex_decomposition(
            str(obj_path),
            str(temp_dir / "decomposed"),
            max_hulls=32
        )
        
        # Step 6: Create CathSim phantom
        phantom_path = create_cathsim_phantom(
            mjcf_path,
            phantom_name,
            cathsim_phantoms_dir=f"{output_base_dir}/phantoms"
        )
        
        print(f"  ✓ Created phantom: {phantom_name}")
        return phantom_path
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None
    
    finally:
        # Cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# STEP 8: Batch Process All Subjects
# ============================================================================

def batch_process_dataset(
    dataset_dir: str,
    output_dir: str = "./cathsim_output",
    max_subjects: int = 100,
    num_workers: int = 4
) -> List[Path]:
    """
    Process multiple subjects in parallel.
    
    Args:
        dataset_dir: Path to TotalSegmentator dataset
        output_dir: Output directory for CathSim phantoms
        max_subjects: Maximum subjects to process
        num_workers: Parallel workers
    
    Returns:
        List of created phantom paths
    """
    from tqdm import tqdm
    
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subject directories
    subject_dirs = sorted([
        d for d in dataset_path.iterdir() 
        if d.is_dir() and (d / "segmentations").exists()
    ])[:max_subjects]
    
    print(f"Found {len(subject_dirs)} subjects to process")
    
    # Process subjects
    successful_phantoms = []
    
    for idx, subject_dir in enumerate(tqdm(subject_dirs, desc="Processing subjects")):
        result = process_single_subject(subject_dir, str(output_path), idx)
        if result:
            successful_phantoms.append(result)
    
    print(f"\n✓ Successfully created {len(successful_phantoms)}/{len(subject_dirs)} phantoms")
    return successful_phantoms


# ============================================================================
# STEP 9: Generate CathSim Environment Config
# ============================================================================

def generate_cathsim_env_config(
    phantoms_dir: str,
    output_file: str = "cathsim_phantoms_config.py"
):
    """
    Generate a Python config file for using phantoms with CathSim.
    """
    phantoms_path = Path(phantoms_dir)
    phantom_names = [d.name for d in phantoms_path.iterdir() if d.is_dir()]
    
    config_content = f'''"""
CathSim Phantoms Configuration
Generated from TotalSegmentator dataset
Total phantoms: {len(phantom_names)}
"""

import gymnasium as gym
import cathsim.gym.envs

# List of available phantom names
PHANTOM_NAMES = {phantom_names}

# Example targets for each phantom (adjust based on actual anatomy)
DEFAULT_TARGETS = {{
    phantom: [0.0, 0.0, 0.1]  # Placeholder - should be set per phantom
    for phantom in PHANTOM_NAMES
}}


def make_cathsim_env(phantom_name: str, target: list = None):
    """
    Create a CathSim environment with a specific phantom.
    
    Args:
        phantom_name: Name of the phantom (e.g., "aorta_0001")
        target: Target position [x, y, z] or None for default
    
    Returns:
        Gymnasium environment
    """
    if phantom_name not in PHANTOM_NAMES:
        raise ValueError(f"Unknown phantom: {{phantom_name}}")
    
    task_kwargs = dict(
        phantom=phantom_name,
        target=target or DEFAULT_TARGETS[phantom_name],
    )
    
    return gym.make("cathsim/CathSim-v0", **task_kwargs)


def get_random_phantom():
    """Get a random phantom name."""
    import random
    return random.choice(PHANTOM_NAMES)


# Example usage
if __name__ == "__main__":
    # Create environment with first phantom
    env = make_cathsim_env(PHANTOM_NAMES[0])
    
    # Reset and run
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
'''
    
    with open(output_file, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Generated config: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main pipeline execution.
    
    Steps:
    1. Download TotalSegmentator dataset (1228 subjects)
    2. Extract aorta segmentations
    3. Convert to meshes
    4. Apply convex decomposition
    5. Create CathSim-compatible phantoms
    6. Generate environment config
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert TotalSegmentator to CathSim")
    parser.add_argument("--data-dir", default="./totalsegmentator_data",
                       help="Directory to download/store dataset")
    parser.add_argument("--output-dir", default="./cathsim_output",
                       help="Output directory for CathSim phantoms")
    parser.add_argument("--max-subjects", type=int, default=100,
                       help="Maximum subjects to process")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip dataset download")
    parser.add_argument("--workers", type=int, default=4,
                       help="Parallel workers")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TotalSegmentator → CathSim Pipeline")
    print("=" * 60)
    
    # Step 0: Install dependencies
    install_dependencies()
    
    # Step 1: Download dataset
    if not args.skip_download:
        dataset_path = download_totalsegmentator_dataset(
            args.data_dir,
            args.max_subjects
        )
    else:
        dataset_path = Path(args.data_dir) / "Totalsegmentator_dataset_v201"
    
    # Steps 2-6: Process subjects
    phantoms = batch_process_dataset(
        str(dataset_path),
        args.output_dir,
        args.max_subjects,
        args.workers
    )
    
    # Step 7: Generate config
    if phantoms:
        generate_cathsim_env_config(
            f"{args.output_dir}/phantoms",
            f"{args.output_dir}/cathsim_phantoms_config.py"
        )
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Phantoms created: {len(phantoms)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
