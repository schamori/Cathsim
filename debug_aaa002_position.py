"""
Debug AAA002 phantom position and camera settings
"""

import numpy as np
from cathsim.dm import make_dm_env

# Create the environment with AAA002 phantom
env = make_dm_env(
    phantom="AAA002",
    target="start",
    use_pixels=True,
    image_size=480,
)

time_step = env.reset()
physics = env.physics
phantom = env._task._phantom

print("=== Phantom Position Analysis ===")

# Get all phantom geom positions
phantom_geoms = [name for name in physics.named.data.geom_xpos.axes.row.names if 'phantom' in name]
print(f"\nPhantom geoms: {phantom_geoms}")

for geom_name in phantom_geoms:
    pos = physics.named.data.geom_xpos[geom_name]
    print(f"{geom_name}: position = {pos}")

# Get phantom body position
print(f"\nPhantom body position: {physics.named.data.xpos['phantom']}")

# Get bounding box of phantom visual mesh
visual_path = phantom.phantom_visual
print(f"\nPhantom visual mesh path: {visual_path}")

if visual_path.exists():
    import trimesh
    mesh = trimesh.load(str(visual_path))
    print(f"Mesh bounds: {mesh.bounds}")
    print(f"Mesh center: {mesh.centroid}")
    print(f"Mesh extents: {mesh.extents}")

    # Scale the mesh according to phantom scale
    scale = phantom.get_scale()
    print(f"\nPhantom scale: {scale}")
    scaled_bounds = mesh.bounds * scale[0]
    print(f"Scaled bounds: {scaled_bounds}")
    scaled_center = mesh.centroid * scale[0]
    print(f"Scaled center: {scaled_center}")

# Check camera position
camera = env._task._arena.top_camera
print(f"\n=== Camera Settings ===")
print(f"Top camera position: {camera.pos}")
print(f"Top camera quat: {camera.quat}")

# Get guidewire position
guidewire_tip = physics.named.data.geom_xpos['guidewire/tip/head']
print(f"\n=== Guidewire Position ===")
print(f"Guidewire tip: {guidewire_tip}")

# Compute suggested camera position
print(f"\n=== Suggested Fix ===")
print(f"The phantom appears to be at position {physics.named.data.xpos['phantom']}")
print(f"The camera is at {camera.pos}")
print("This mismatch may be why nothing is visible in the renders.")
