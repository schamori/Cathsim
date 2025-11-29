"""
Quick script to create AAA001 phantom for CathSim
Assumes you have a mesh file (STL/OBJ) for your AAA001 model
"""

import trimesh
import shutil
from pathlib import Path

def create_aaa001_phantom(input_mesh_path: str):
    """
    Create AAA001 phantom from a mesh file

    Args:
        input_mesh_path: Path to your AAA001 mesh file (STL/OBJ)
    """

    # Load the mesh
    print(f"Loading mesh from {input_mesh_path}")
    mesh = trimesh.load(input_mesh_path)
    print(f"Original mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

    # Create output directories
    phantom_assets = Path("src/cathsim/dm/components/phantom_assets")
    meshes_dir = phantom_assets / "meshes" / "AAA001"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Simplify visual mesh to stay under MuJoCo's 200k face limit
    # Target ~100k faces for visual (leaving room for collision meshes)
    target_faces = 100000
    if len(mesh.faces) > target_faces:
        print(f"Simplifying mesh to ~{target_faces} faces (MuJoCo limit is 200k)...")
        try:
            # Calculate reduction ratio (target_reduction must be between 0 and 1)
            reduction_ratio = 1.0 - (target_faces / len(mesh.faces))
            visual_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            print(f"Simplified to {len(visual_mesh.faces)} faces")
        except Exception as e:
            print(f"  Warning: quadric decimation failed ({e})")
            print(f"  Using convex hull instead (will be much simpler)")
            visual_mesh = mesh.convex_hull
            print(f"Simplified to {len(visual_mesh.faces)} faces")
    else:
        visual_mesh = mesh

    # Export visual mesh
    visual_path = meshes_dir / "visual.stl"
    # Export as binary STL (default for trimesh)
    with open(visual_path, 'wb') as f:
        visual_mesh.export(f, file_type='stl')
    print(f"[OK] Created visual mesh: {visual_path} ({len(visual_mesh.faces)} faces)")

    # Create simplified collision mesh (even more simplified for collision)
    # Target ~10k faces for collision
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

    # Create convex decomposition (simplified - just use convex hull for now)
    print("Creating convex hulls...")
    try:
        # Try to decompose into multiple convex hulls
        hulls = mesh.convex_decomposition(maxhulls=10)
    except:
        # Fallback: use single convex hull
        print("  Using single convex hull (install CoACD for better decomposition)")
        hulls = [mesh.convex_hull]

    # Export hulls
    for i, hull in enumerate(hulls):
        hull_path = meshes_dir / f"hull_{i}.stl"
        with open(hull_path, 'wb') as f:
            hull.export(f, file_type='stl')
    print(f"[OK] Created {len(hulls)} convex hulls")

    # Generate XML file
    xml_content = generate_xml(len(hulls))
    xml_path = phantom_assets / "AAA001.xml"
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    print(f"[OK] Created XML: {xml_path}")

    print("\n[OK] AAA001 phantom created successfully!")
    print("\nYou can now use it with:")
    print('  phantom="AAA001" in your quickstart')


def generate_xml(num_hulls: int) -> str:
    """Generate MuJoCo XML for AAA001"""

    # Generate mesh references
    mesh_refs = ['    <mesh name="visual" file="AAA001/visual.stl" />']
    for i in range(num_hulls):
        mesh_refs.append(f'    <mesh name="hull_{i}" file="AAA001/hull_{i}.stl" />')

    # Generate geometry references
    geom_refs = ['      <geom name="visual" type="mesh" mesh="visual" contype="0" conaffinity="0" group="2" />']
    for i in range(num_hulls):
        geom_refs.append(f'      <geom type="mesh" mesh="hull_{i}" />')

    xml = f'''<mujoco model="phantom">
  <compiler meshdir="meshes" />
  <asset>
{chr(10).join(mesh_refs)}
  </asset>
  <worldbody>
    <body name="phantom">
{chr(10).join(geom_refs)}

      <!-- Target sites (adjust coordinates based on your anatomy) -->
      <site name="bca" pos="0.0 0.02 0.15" size="0.003" />
      <site name="lcca" pos="-0.01 0.02 0.15" size="0.003" />
      <site name="lsa" pos="0.01 0.02 0.15" size="0.003" />
    </body>
  </worldbody>
</mujoco>
'''
    return xml


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python create_aaa001.py <path_to_aaa001_mesh.stl>")
        print("\nExample:")
        print("  python create_aaa001.py my_aaa001_model.stl")
        sys.exit(1)

    mesh_path = sys.argv[1]
    if not Path(mesh_path).exists():
        print(f"Error: File not found: {mesh_path}")
        sys.exit(1)

    create_aaa001_phantom(mesh_path)
