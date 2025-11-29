"""
Check phantom position and determine correct camera placement for AAA002
"""

import numpy as np
from cathsim.dm import make_dm_env
import matplotlib.pyplot as plt
from pathlib import Path

env = make_dm_env(phantom="AAA002", target="goal", use_pixels=True, image_size=480)
time_step = env.reset()
physics = env.physics

# Get phantom bounds
phantom_geoms = [name for name in physics.named.data.geom_xpos.axes.row.names if 'phantom' in name]
print("=== Phantom Geometry Positions ===")
for geom_name in phantom_geoms:
    pos = physics.named.data.geom_xpos[geom_name]
    print(f"{geom_name}: {pos}")

# Get site positions
sites = env._task._phantom.sites
print(f"\n=== Site Positions ===")
for name, pos in sites.items():
    print(f"{name}: {pos}")

# Calculate phantom center
start = np.array(sites['start'])
goal = np.array(sites['goal'])
center = (start + goal) / 2
print(f"\n=== Calculated Center (between start and goal) ===")
print(f"Center: {center}")
print(f"Distance from start to goal: {np.linalg.norm(goal - start):.4f} m")

# Check current camera
camera = env._task._arena.top_camera
print(f"\n=== Current Camera Settings ===")
print(f"Position: {camera.pos}")
print(f"Quat: {camera.quat}")

# Suggest better camera position
# For top-down view, camera should be above the center looking down
camera_distance = 0.3  # 30cm above
suggested_pos = [center[0], center[1], center[2] + camera_distance]
print(f"\n=== Suggested Camera Position ===")
print(f"Position: {suggested_pos}")
print(f"Quat: [1, 0, 0, 0] (looking down)")

# Manually render with adjusted camera
from dm_control.mujoco import wrapper

# Temporarily modify camera position for testing
camera.pos = suggested_pos

scene_option = wrapper.MjvOption()
scene_option.geomgroup[:] = 0
scene_option.geomgroup[1] = 1  # Catheter
scene_option.geomgroup[2] = 1  # Phantom

camera_id = physics.model.name2id(camera.full_identifier, 'camera')
img = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option)

output_dir = Path("aaa002_camera_fix")
output_dir.mkdir(exist_ok=True)
plt.imsave(output_dir / "adjusted_camera.png", img)
print(f"\nSaved adjusted camera view to {output_dir / 'adjusted_camera.png'}")

# Try closer camera
camera.pos = [center[0], center[1], center[2] + 0.15]  # 15cm above
img_close = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option)
plt.imsave(output_dir / "close_camera.png", img_close)
print(f"Saved close camera view to {output_dir / 'close_camera.png'}")
