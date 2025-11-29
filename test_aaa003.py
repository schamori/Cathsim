"""
Test AAA003 phantom with catheter starting at iliac_left and goal at end of abdominal aorta
"""

import cathsim.gym.envs
import gymnasium as gym
import numpy as np


# Create environment with AAA003 phantom
# The catheter will start at the "start" site (iliac_left)
# The goal will be set to the "goal" site (end of abdominal_aorta)
task_kwargs = dict(
    dense_reward=True,
    success_reward=10.0,
    delta=0.004,
    use_pixels=False,
    use_segment=False,
    image_size=64,
    phantom="AAA003",  # Use AAA003 phantom
    target="goal",     # Set target to the "goal" site (end of abdominal aorta)
    random_init_distance=0.0,  # Set to 0 to start exactly at the start site
    visualize_sites=True,      # Visualize all sites
    visualize_target=True,     # Visualize the target
)

print("Creating environment with AAA003 phantom...")
print(f"  - Phantom: AAA003")
print(f"  - Target: goal (end of abdominal aorta)")
print(f"  - Start position: start (iliac_left)")
print(f"  - Random init distance: 0.0 (no randomization)")

env = gym.make("cathsim/CathSim-v0", **task_kwargs)

# Get the phantom sites to check positions
# Access through the wrapped environment - unwrap until we get to the dm_control env
unwrapped_env = env
while hasattr(unwrapped_env, 'env'):
    unwrapped_env = unwrapped_env.env
    if hasattr(unwrapped_env, '_env'):
        dm_env = unwrapped_env._env
        break

# The CathSim gym wrapper stores the dm_env in _env
if hasattr(env, '_env'):
    dm_env = env._env
elif hasattr(env.unwrapped, '_env'):
    dm_env = env.unwrapped._env
else:
    raise AttributeError("Could not find dm_control environment")

task = dm_env._task
phantom = task._phantom
sites = phantom.sites

print("\nPhantom sites:")
for site_name, site_pos in sites.items():
    print(f"  {site_name}: {site_pos}")

# Override the initial pose to use the "start" site
start_pos = np.array(sites["start"])
goal_pos = np.array(sites["goal"])

print(f"\nStart position (iliac_left): {start_pos}")
print(f"Goal position (end of abdominal aorta): {goal_pos}")

# Create a custom initialization function
def custom_initialize_episode(physics, random_state):
    """Custom initialization that sets guidewire at the start site"""
    # Set guidewire pose to the start site
    task._guidewire.set_pose(physics, position=start_pos)
    task.success = False

# Override the initialize_episode method
task.initialize_episode = custom_initialize_episode

print("\nResetting environment...")
obs, info = env.reset()

# Get the current head position
# Access through the unwrapped CathSim environment
cathsim_env = env.unwrapped
head_pos = cathsim_env.head_pos
target_pos = cathsim_env.target

print(f"\nCatheter head position after reset: {head_pos}")
print(f"Target position: {target_pos}")
print(f"Distance to target: {np.linalg.norm(head_pos - target_pos):.6f} meters")

# Test a few steps
print("\nTesting environment for 5 steps...")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    head_pos = cathsim_env.head_pos
    dist = np.linalg.norm(head_pos - target_pos)
    print(f"Step {i+1}: reward={reward:.4f}, distance={dist:.6f}m, terminated={terminated}")

print("\n[OK] AAA003 test completed!")
print("The catheter now starts at iliac_left and targets the end of abdominal aorta.")

env.close()
