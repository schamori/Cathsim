import trimesh

# Try different loading methods
stl_path = "C:/Users/Admin/Downloads/10932957/meshes/meshes/AAA004.stl"

print("Method 1: Default load")
m1 = trimesh.load(stl_path)
print(f"Type: {type(m1)}")
if hasattr(m1, 'faces'):
    print(f"Faces: {len(m1.faces)}")
elif hasattr(m1, 'geometry'):
    print(f"Geometries: {len(m1.geometry)}")

print("\nMethod 2: Load with file_type='stl'")
m2 = trimesh.load(stl_path, file_type='stl')
print(f"Type: {type(m2)}")
if hasattr(m2, 'faces'):
    print(f"Faces: {len(m2.faces)}")

print("\nMethod 3: Load STL directly")
with open(stl_path, 'rb') as f:
    m3 = trimesh.load(f, file_type='stl')
print(f"Type: {type(m3)}")
if hasattr(m3, 'faces'):
    print(f"Faces: {len(m3.faces)}")
