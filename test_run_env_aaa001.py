"""
Test that run_env properly initializes catheter at start position for AAA001
"""

from cathsim.dm import make_dm_env
import numpy as np

print("Testing AAA001 with run_env configuration...")

# Create environment exactly as run_env does
env = make_dm_env(
    phantom="AAA001",
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

# Reset the environment (this should use the start site now)
print("\nResetting environment...")
timestep = env.reset()

# Get the head position
physics = env.physics
head_pos = env._task.get_head_pos(physics)
target_pos = env._task.target_pos

print(f"\nCatheter head position after reset: {head_pos}")
print(f"Expected head position (iliac_left): {sites['start'] + np.array([0, 0.0185, 0])}")
print(f"Target position (goal): {target_pos}")

# Check if head is at the correct position
expected_head = sites['start'] + np.array([0, 0.0185, 0])  # Account for guidewire offset
distance = np.linalg.norm(head_pos - expected_head)
print(f"\nDistance from expected position: {distance:.6f} meters")

if distance < 0.001:  # Within 1mm
    print("[SUCCESS] Catheter head starts at the correct position!")
else:
    print(f"[FAIL] Catheter head is {distance*1000:.1f}mm away from expected position")

# Test a few steps
print("\nTesting environment for 3 steps...")
for i in range(3):
    action = np.array([0.5, 0.0])  # Move forward
    timestep = env.step(action)
    head_pos = env._task.get_head_pos(physics)
    dist_to_goal = np.linalg.norm(head_pos - target_pos)
    print(f"Step {i+1}: reward={timestep.reward:.4f}, distance to goal={dist_to_goal:.6f}m")

env.close()
print("\n[OK] Test completed!")
