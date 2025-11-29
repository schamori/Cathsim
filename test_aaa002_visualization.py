"""
Test visualization of AAA002 phantom with catheter
"""

import numpy as np
from cathsim.dm import make_dm_env
from pathlib import Path

def test_aaa002_with_catheter():
    """Create AAA002 environment and visualize with catheter"""

    # Create the environment with AAA002 phantom
    env = make_dm_env(
        phantom="AAA002",  # Note: just "AAA002", not "AAA002.xml"
        target="start",    # Use the start site as target
        use_pixels=True,
        image_size=480,
        visualize_sites=True,
        visualize_target=True,
    )

    # Reset environment
    time_step = env.reset()

    print("Environment created successfully!")
    print(f"Observation keys: {time_step.observation.keys()}")

    # Check if sites exist
    physics = env.physics
    phantom = env._task._phantom
    sites = phantom.sites
    print(f"\nAvailable sites: {list(sites.keys())}")
    print(f"Start site position: {sites['start']}")
    print(f"Goal site position: {sites['goal']}")

    # Check guidewire
    guidewire = env._task._guidewire
    print(f"\nGuidewire bodies: {guidewire._n_bodies}")

    # Find all geoms in the simulation
    all_geoms = [name for name in physics.named.data.geom_xpos.axes.row.names]
    print(f"All geoms in simulation: {all_geoms}")

    # Check if guidewire/catheter geoms exist
    catheter_geoms = [g for g in all_geoms if 'B' in g or 'guidewire' in g.lower() or 'tip' in g.lower()]
    print(f"Catheter geoms found: {catheter_geoms}")

    if not catheter_geoms:
        print("\n⚠️  WARNING: No catheter geoms found in the simulation!")
        print("This means the catheter/guidewire is not visible.")
        print(f"Guidewire object exists: {guidewire is not None}")
        print(f"Guidewire attached: {hasattr(guidewire, 'mjcf_model')}")

    # Run a few steps to see the catheter move
    print("\nRunning simulation steps...")
    for step in range(10):
        action = np.array([0.5, 0.0])  # Push forward, no rotation
        time_step = env.step(action)

        if catheter_geoms:
            tip_pos = physics.named.data.geom_xpos[catheter_geoms[-1]]
            print(f"Step {step}: Tip position = {tip_pos}")
        else:
            print(f"Step {step}: Catheter not visible")

    # Check geom groups and visualization settings
    print("\n=== Checking Visibility Settings ===")

    # Check guidewire geom attributes
    guidewire_geom = guidewire.mjcf_model.find('geom', 'guidewire_geom_0')
    print(f"Guidewire geom group: {guidewire_geom.group if hasattr(guidewire_geom, 'group') and guidewire_geom.group else 'default (0)'}")
    print(f"Guidewire rgba: {guidewire_geom.rgba if hasattr(guidewire_geom, 'rgba') and guidewire_geom.rgba else 'default'}")

    # Get guidewire geom ID and check its properties in physics
    geom_id = physics.model.name2id('guidewire/guidewire_geom_0', 'geom')
    print(f"Guidewire geom ID: {geom_id}")
    print(f"Guidewire geom group in physics: {physics.model.geom_group[geom_id]}")
    print(f"Guidewire geom rgba in physics: {physics.model.geom_rgba[geom_id]}")

    # Save visualization images if pixels are available
    if 'pixels' in time_step.observation:
        import matplotlib.pyplot as plt

        output_dir = Path("aaa002_visualization")
        output_dir.mkdir(exist_ok=True)

        # Save camera views
        pixels = time_step.observation['pixels']
        print(f"\nPixels shape: {pixels.shape}")

        # If it's a 3D array (H, W, C), handle differently
        if len(pixels.shape) == 3:
            if pixels.shape[2] == 3:
                # RGB image
                plt.imsave(output_dir / "top_view.png", pixels)
            else:
                # Single channel or other
                plt.imsave(output_dir / "top_view.png", pixels[:, :, 0], cmap='gray')
        else:
            plt.imsave(output_dir / "top_view.png", pixels, cmap='gray')

        print(f"Saved top view to {output_dir / 'top_view.png'}")

        # Also render manually with different geom groups to debug visibility
        camera = env._task._arena.top_camera
        camera_id = physics.model.name2id(camera.full_identifier, 'camera')

        # Render with all geom groups enabled
        from dm_control.mujoco import wrapper
        scene_option = wrapper.MjvOption()
        scene_option.geomgroup[:] = 1  # Enable all geom groups

        manual_img = physics.render(height=480, width=480, camera_id=camera_id, scene_option=scene_option)
        plt.imsave(output_dir / "manual_all_groups.png", manual_img)
        print(f"Saved manual render (all groups) to {output_dir / 'manual_all_groups.png'}")

    print("\nTest completed!")

if __name__ == "__main__":
    test_aaa002_with_catheter()
