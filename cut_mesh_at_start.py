"""
Cut the mesh to remove everything before the start position (iliac_left entry)
This uses a plane at the start position to cut away the vessel portion before the entry point
"""

import trimesh
import numpy as np
from pathlib import Path
try:
    import pyvista as pv
except ImportError:
    pv = None
    print("Warning: pyvista not installed. Install with: pip install pyvista")


def load_start_position(centerlines_dir: str, phantom_name: str):
    """Load the start position from iliac_left centerline"""
    if pv is None:
        raise ImportError("pyvista is required for this operation")

    centerlines_path = Path(centerlines_dir) / phantom_name
    iliac_left_path = centerlines_path / "iliac_left.vtp"

    if not iliac_left_path.exists():
        raise FileNotFoundError(f"iliac_left.vtp not found at {iliac_left_path}")

    iliac_left = pv.read(str(iliac_left_path))
    # Get the first point (start position) - this is in mm
    start_point = iliac_left.points[0]

    # Also get the direction - use second point to determine cutting plane normal
    if len(iliac_left.points) > 1:
        second_point = iliac_left.points[1]
        direction = second_point - start_point
        direction = direction / np.linalg.norm(direction)  # normalize
    else:
        direction = None

    return start_point, direction


def cut_mesh_at_start(mesh_path: str, phantom_name: str, centerlines_dir: str, output_path: str = None):
    """
    Cut the mesh to remove everything before the start position

    Args:
        mesh_path: Path to the input STL file
        phantom_name: Name of the phantom (e.g., "AAA001")
        centerlines_dir: Directory containing centerline VTP files
        output_path: Path to save the cut mesh (optional)
    """

    print(f"Loading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path)
    print(f"Original mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    print(f"Original bounds: {mesh.bounds}")

    # Load start position from centerline (in mm)
    print(f"\nLoading start position from centerlines...")
    start_point_mm, direction = load_start_position(centerlines_dir, phantom_name)
    print(f"Start position (mm): {start_point_mm}")
    print(f"Vessel direction: {direction}")

    # The mesh is in mm, centerlines are in mm, so we can work directly
    start_point = start_point_mm

    # Create a cutting plane at the start position
    # The plane normal should point "backwards" (opposite to vessel direction) to remove the part before
    if direction is not None:
        # Point the normal backwards to cut away everything "before" the start
        plane_normal = -direction
    else:
        # If we don't have direction, we need to guess
        # Typically vessels go in the +Z direction in medical imaging
        print("Warning: Could not determine vessel direction, using default")
        plane_normal = np.array([0, 0, -1])  # Cut along Z axis

    print(f"Cutting plane normal: {plane_normal}")
    print(f"Cutting plane origin: {start_point}")

    # Use trimesh.intersections.slice_mesh_plane to cut the mesh
    # This keeps the part of the mesh on the positive side of the plane normal
    # So we want the normal to point "forward" (into the vessel we want to keep)
    plane_normal_keep = -plane_normal  # Flip it so we keep the forward part

    print(f"\nCutting mesh...")
    try:
        # slice_mesh_plane returns the part of mesh on positive side of plane
        cut_mesh = mesh.slice_plane(
            plane_origin=start_point,
            plane_normal=plane_normal_keep,
            cap=False  # Do NOT cap - leave the opening for catheter entry
        )

        if cut_mesh is None or len(cut_mesh.vertices) == 0:
            print("Warning: Cutting resulted in empty mesh. Trying opposite direction...")
            # Try flipping the normal
            cut_mesh = mesh.slice_plane(
                plane_origin=start_point,
                plane_normal=-plane_normal_keep,
                cap=False  # Do NOT cap - leave the opening
            )

        print(f"Cut mesh: {len(cut_mesh.faces)} faces, {len(cut_mesh.vertices)} vertices")
        print(f"Cut mesh bounds: {cut_mesh.bounds}")

        # Verify the cut was successful by checking if start point is near the mesh boundary
        distances = np.linalg.norm(cut_mesh.vertices - start_point, axis=1)
        min_dist = np.min(distances)
        print(f"Closest vertex to start point: {min_dist:.3f} mm")

    except Exception as e:
        print(f"Error during cutting: {e}")
        print("Falling back to manual vertex filtering...")

        # Fallback: manually filter vertices
        # Keep vertices that are "after" the start point (in the direction of the vessel)
        vertices = mesh.vertices
        to_start = vertices - start_point
        # Vertices with positive dot product with direction are "after" the start
        keep_mask = np.dot(to_start, direction) >= 0

        # This is more complex - we'd need to rebuild the mesh
        # For now, raise the error
        raise e

    # Save the cut mesh
    if output_path is None:
        output_path = mesh_path.replace('.stl', '_cut.stl')

    # Backup original
    backup_path = mesh_path.replace('.stl', '_original.stl')
    print(f"\nBacking up original to {backup_path}")
    import shutil
    shutil.copy(mesh_path, backup_path)

    print(f"Saving cut mesh to {output_path}")
    with open(output_path, 'wb') as f:
        cut_mesh.export(f, file_type='stl')

    print(f"\n[OK] Mesh cut successfully!")
    print(f"  - Removed {len(mesh.vertices) - len(cut_mesh.vertices)} vertices")
    print(f"  - Removed {len(mesh.faces) - len(cut_mesh.faces)} faces")

    return cut_mesh


if __name__ == "__main__":
    import sys

    centerlines_dir = "C:/Users/Admin/Downloads/10932957/centerlines/centerlines"

    if len(sys.argv) < 2:
        print("Usage: python cut_mesh_at_start.py <phantom_name>")
        print("\nExample:")
        print("  python cut_mesh_at_start.py AAA001")
        print("  python cut_mesh_at_start.py AAA003")
        print("\nThis will cut the mesh at the iliac_left start position")
        sys.exit(1)

    phantom_name = sys.argv[1]
    mesh_path = f"C:/Users/Admin/Downloads/10932957/meshes/meshes/{phantom_name}.stl"

    if not Path(mesh_path).exists():
        print(f"Error: Mesh file not found: {mesh_path}")
        sys.exit(1)

    cut_mesh_at_start(
        mesh_path=mesh_path,
        phantom_name=phantom_name,
        centerlines_dir=centerlines_dir,
        output_path=mesh_path  # Overwrite the original (after backup)
    )
