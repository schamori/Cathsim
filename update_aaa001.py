"""
Update AAA001 phantom with centerline-based sites
"""

from pathlib import Path
import trimesh
import numpy as np
try:
    import pyvista as pv
except ImportError:
    pv = None
    print("Warning: pyvista not installed. Install with: pip install pyvista")


def load_centerline_endpoints(centerlines_dir: str, phantom_name: str):
    """
    Load centerline endpoints for start and goal sites.

    Args:
        centerlines_dir: Directory containing centerline VTP files
        phantom_name: Name of the phantom (e.g., "AAA001")

    Returns:
        dict with 'start', 'goal', and 'start_direction' positions
    """
    if pv is None:
        print("Warning: Cannot load centerlines without pyvista. Using default positions.")
        return {"start": [0.0, 0.0, 0.0], "goal": [0.0, 0.0, 0.15], "start_direction": [0.0, 0.0, 1.0]}

    centerlines_path = Path(centerlines_dir) / phantom_name

    # Load iliac_left centerline (for start point)
    iliac_left_path = centerlines_path / "iliac_left.vtp"
    # Load abdominal_aorta centerline (for goal point)
    aorta_path = centerlines_path / "abdominal_aorta.vtp"

    sites = {}

    try:
        if iliac_left_path.exists():
            iliac_left = pv.read(str(iliac_left_path))
            # Get the first point as the start
            start_point = iliac_left.points[0]
            # Convert to meters (assuming centerlines are in mm)
            sites['start'] = (start_point / 1000.0).tolist()
            print(f"[OK] Start site from iliac_left: {sites['start']}")

            # Calculate direction from first two points of centerline
            if len(iliac_left.points) > 1:
                second_point = iliac_left.points[1]
                direction = second_point - start_point
                direction = direction / np.linalg.norm(direction)  # normalize
                sites['start_direction'] = direction.tolist()
                print(f"[OK] Start direction from iliac_left: {sites['start_direction']}")
            else:
                sites['start_direction'] = [0.0, 0.0, 1.0]
        else:
            print(f"Warning: {iliac_left_path} not found")
            sites['start'] = [0.0, 0.0, 0.0]
            sites['start_direction'] = [0.0, 0.0, 1.0]

        if aorta_path.exists():
            aorta = pv.read(str(aorta_path))
            # Get the last point as the goal (end of abdominal aorta)
            goal_point = aorta.points[-1]
            # Convert to meters (assuming centerlines are in mm)
            sites['goal'] = (goal_point / 1000.0).tolist()
            print(f"[OK] Goal site from aorta end: {sites['goal']}")
        else:
            print(f"Warning: {aorta_path} not found")
            sites['goal'] = [0.0, 0.0, 0.15]

    except Exception as e:
        print(f"Error loading centerlines: {e}")
        sites = {"start": [0.0, 0.0, 0.0], "goal": [0.0, 0.0, 0.15], "start_direction": [0.0, 0.0, 1.0]}

    return sites


def update_aaa001_phantom(stl_path: str, phantom_name: str, centerlines_dir: str):
    """Update AAA001 phantom with centerline-based sites

    Args:
        stl_path: Path to the STL file
        phantom_name: Name for the phantom (e.g., "AAA001")
        centerlines_dir: Directory containing centerline VTP files
    """

    # Load the mesh
    print(f"Loading mesh from {stl_path}")
    mesh = trimesh.load(stl_path)
    print(f"Original mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

    # Scale mesh from millimeters to meters (divide by 1000)
    print("Scaling mesh from mm to meters (÷1000)...")
    mesh.apply_scale(0.001)
    print(f"Scaled mesh bounds: {mesh.bounds}")

    # Load centerlines and extract sites
    print(f"\nLoading centerlines from {centerlines_dir}")
    sites = load_centerline_endpoints(centerlines_dir, phantom_name)

    # Create output directories
    phantom_assets = Path("src/cathsim/dm/components/phantom_assets")
    meshes_dir = phantom_assets / "meshes" / phantom_name
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Simplify visual mesh to stay under MuJoCo's 200k face limit
    target_faces = 100000
    if len(mesh.faces) > target_faces:
        print(f"Simplifying mesh to ~{target_faces} faces (MuJoCo limit is 200k)...")
        try:
            visual_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            print(f"Simplified to {len(visual_mesh.faces)} faces")
        except Exception as e:
            print(f"  Warning: quadric decimation failed ({e})")
            print(f"  Using convex hull instead")
            visual_mesh = mesh.convex_hull
            print(f"Simplified to {len(visual_mesh.faces)} faces")
    else:
        visual_mesh = mesh

    # Export visual mesh
    visual_path = meshes_dir / "visual.stl"
    with open(visual_path, 'wb') as f:
        visual_mesh.export(f, file_type='stl')
    print(f"[OK] Created visual mesh: {visual_path} ({len(visual_mesh.faces)} faces)")

    # Create simplified collision mesh
    target_collision_faces = min(10000, int(len(visual_mesh.faces) * 0.1))
    try:
        simplified = visual_mesh.simplify_quadric_decimation(face_count=target_collision_faces)
    except:
        print("  Warning: Using convex hull for collision")
        simplified = visual_mesh.convex_hull
    simplified_path = meshes_dir / "simplified.stl"
    with open(simplified_path, 'wb') as f:
        simplified.export(f, file_type='stl')
    print(f"[OK] Created simplified mesh: {simplified_path} ({len(simplified.faces)} faces)")

    # Create convex hull
    print("Creating convex hulls...")
    hulls = [simplified.convex_hull]
    for i, hull in enumerate(hulls):
        hull_path = meshes_dir / f"hull_{i}.stl"
        with open(hull_path, 'wb') as f:
            hull.export(f, file_type='stl')
    print(f"[OK] Created {len(hulls)} convex hulls")

    # Generate XML file with centerline sites
    xml_content = generate_xml(phantom_name, len(hulls), sites)
    xml_path = phantom_assets / f"{phantom_name}.xml"
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    print(f"[OK] Created XML: {xml_path}")

    print(f"\n[OK] {phantom_name} phantom updated successfully!")
    print(f"  - Start position set to iliac_left: {sites['start']}")
    print(f"  - Goal position set to end of abdominal_aorta: {sites['goal']}")


def generate_xml(phantom_name: str, num_hulls: int, sites: dict = None) -> str:
    """Generate MuJoCo XML for the phantom

    Args:
        phantom_name: Name of the phantom
        num_hulls: Number of convex hulls
        sites: Dictionary of site positions {'start': [x,y,z], 'goal': [x,y,z]}
    """

    # Generate mesh references
    mesh_refs = [f'    <mesh name="visual" file="{phantom_name}/visual.stl" />']
    for i in range(num_hulls):
        mesh_refs.append(f'    <mesh name="hull_{i}" file="{phantom_name}/hull_{i}.stl" />')

    # Generate geometry references
    geom_refs = ['      <geom name="visual" type="mesh" mesh="visual" contype="0" conaffinity="0" group="2" />']
    for i in range(num_hulls):
        geom_refs.append(f'      <geom type="mesh" mesh="hull_{i}" />')

    # Generate site definitions
    site_defs = []
    if sites:
        if 'start' in sites:
            pos = sites['start']
            site_defs.append(f'      <site name="start" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" size="0.003" rgba="0 1 0 1" />')
        if 'start_direction' in sites:
            # Store direction as a site position (will be used to calculate orientation)
            direction = sites['start_direction']
            site_defs.append(f'      <site name="start_direction" pos="{direction[0]:.6f} {direction[1]:.6f} {direction[2]:.6f}" size="0.001" rgba="0 1 0 0.5" />')
        if 'goal' in sites:
            pos = sites['goal']
            site_defs.append(f'      <site name="goal" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" size="0.003" rgba="1 0 0 1" />')

    # Add default target sites
    site_defs.append('      <site name="bca" pos="0.0 0.02 0.15" size="0.003" />')
    site_defs.append('      <site name="lcca" pos="-0.01 0.02 0.15" size="0.003" />')
    site_defs.append('      <site name="lsa" pos="0.01 0.02 0.15" size="0.003" />')

    xml = f'''<mujoco model="phantom">
  <compiler meshdir="meshes" />
  <asset>
{chr(10).join(mesh_refs)}
  </asset>
  <worldbody>
    <body name="phantom" mocap="false">
      <inertial pos="0 0 0" mass="1000" diaginertia="1000 1000 1000" />
{chr(10).join(geom_refs)}

      <!-- Centerline-based sites -->
{chr(10).join(site_defs)}
    </body>
  </worldbody>
</mujoco>
'''
    return xml


if __name__ == "__main__":
    # Update AAA001 phantom with centerline sites
    update_aaa001_phantom(
        stl_path="C:/Users/Admin/Downloads/10932957/meshes/meshes/AAA001.stl",
        phantom_name="AAA001",
        centerlines_dir="C:/Users/Admin/Downloads/10932957/centerlines/centerlines"
    )
