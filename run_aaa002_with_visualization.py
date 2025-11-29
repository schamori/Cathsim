"""
Run AAA002 environment with proper visualization showing both phantom and catheter
"""

import numpy as np
from cathsim.dm import make_dm_env
from pathlib import Path

def run_aaa002_visualization():
    """Run AAA002 environment with proper visualization"""

    # Create environment with AAA002
    env = make_dm_env(
        phantom="AAA002",
        target="goal",
        use_pixels=True,
        use_segment=True,  # Enable segmentation
        image_size=480,
        visualize_sites=True,
        visualize_target=True,
    )

    time_step = env.reset()
    physics = env.physics

    print("=== AAA002 Environment ===")
    print(f"Available sites: {list(env._task._phantom.sites.keys())}")
    print(f"Start site: {env._task._phantom.sites['start']}")
    print(f"Goal site: {env._task._phantom.sites['goal']}")

    # Render with different geom group configurations
    from dm_control.mujoco import wrapper
    import matplotlib.pyplot as plt

    output_dir = Path("aaa002_final_visualization")
    output_dir.mkdir(exist_ok=True)

    camera = env._task._arena.top_camera
    camera_id = physics.model.name2id(camera.full_identifier, 'camera')

    # Configuration 1: Show only catheter (group 1)
    scene_option_catheter = wrapper.MjvOption()
    scene_option_catheter.geomgroup[:] = 0
    scene_option_catheter.geomgroup[1] = 1  # Only catheter
    img_catheter = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option_catheter)
    plt.imsave(output_dir / "1_catheter_only.png", img_catheter)
    print(f"Saved catheter-only view to {output_dir / '1_catheter_only.png'}")

    # Configuration 2: Show only phantom (group 2)
    scene_option_phantom = wrapper.MjvOption()
    scene_option_phantom.geomgroup[:] = 0
    scene_option_phantom.geomgroup[2] = 1  # Only phantom visual
    img_phantom = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option_phantom)
    plt.imsave(output_dir / "2_phantom_only.png", img_phantom)
    print(f"Saved phantom-only view to {output_dir / '2_phantom_only.png'}")

    # Configuration 3: Show both catheter and phantom (groups 1 and 2)
    scene_option_both = wrapper.MjvOption()
    scene_option_both.geomgroup[:] = 0
    scene_option_both.geomgroup[1] = 1  # Catheter
    scene_option_both.geomgroup[2] = 1  # Phantom visual
    img_both = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option_both)
    plt.imsave(output_dir / "3_both.png", img_both)
    print(f"Saved combined view (catheter + phantom) to {output_dir / '3_both.png'}")

    # Configuration 4: All groups
    scene_option_all = wrapper.MjvOption()
    scene_option_all.geomgroup[:] = 1
    img_all = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option_all)
    plt.imsave(output_dir / "4_all_groups.png", img_all)
    print(f"Saved all-groups view to {output_dir / '4_all_groups.png'}")

    # Run simulation and capture frames
    print("\n=== Running Simulation ===")
    frames = []
    for step in range(50):
        action = np.array([0.5, 0.0])  # Push forward
        time_step = env.step(action)

        if step % 10 == 0:
            img = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option_both)
            frames.append(img)
            plt.imsave(output_dir / f"frame_{step:03d}.png", img)
            print(f"Step {step}: Captured frame")

    print(f"\n✓ Visualization complete! Check the '{output_dir}' directory for images.")
    print("\nTo use AAA002 in your code:")
    print('  env = make_dm_env(phantom="AAA002", target="goal", use_pixels=True)')

if __name__ == "__main__":
    run_aaa002_visualization()
