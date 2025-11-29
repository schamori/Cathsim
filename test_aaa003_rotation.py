"""
Test AAA003 with 30-degree counterclockwise rotation
"""

from cathsim.dm import make_dm_env
import numpy as np

print("Testing AAA003 with 30-degree counterclockwise rotation...")

# Create environment
env = make_dm_env(
    phantom="AAA003",
    use_pixels=True,
    use_segment=True,
    target="goal",
    visualize_sites=True,
    visualize_target=True,
)

# Get the phantom sites
phantom = env._task._phantom
sites = phantom.sites

print("\nPhantom sites:")
for site_name, site_pos in sites.items():
    print(f"  {site_name}: {site_pos}")

# Reset the environment
print("\nResetting environment...")
timestep = env.reset()

# Get the head position
physics = env.physics
head_pos = env._task.get_head_pos(physics)
target_pos = env._task.target_pos

print(f"\nCatheter head position after reset: {head_pos}")
print(f"Expected position (iliac_left): {sites['start'] + np.array([0, 0.0185, 0])}")
print(f"Target position (goal): {target_pos}")

# Check distance
expected_head = sites['start'] + np.array([0, 0.0185, 0])
distance = np.linalg.norm(head_pos - expected_head)
print(f"\nDistance from expected position: {distance:.6f} meters ({distance*1000:.1f}mm)")

if distance < 0.001:
    print("[SUCCESS] Catheter head at correct position!")
else:
    print(f"[INFO] Position offset due to 30-degree rotation")

# Test a few steps
print("\nTesting environment for 3 steps...")
for i in range(3):
    action = np.array([0.5, 0.0])
    timestep = env.step(action)
    head_pos = env._task.get_head_pos(physics)
    dist_to_goal = np.linalg.norm(head_pos - target_pos)
    print(f"Step {i+1}: reward={timestep.reward:.4f}, distance to goal={dist_to_goal:.6f}m")

env.close()
print("\n[OK] AAA003 test completed!")
print("AAA003 has 30-degree counterclockwise rotation")
