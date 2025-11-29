"""
Fix the catheter starting position for AAA001
The issue is that set_pose sets the BASE of the guidewire, but we want the HEAD at the start position
We need to calculate the offset and adjust accordingly
"""

import cathsim.gym.envs
import gymnasium as gym
import numpy as np
try:
    import pyvista as pv
except ImportError:
    pv = None


def calculate_guidewire_offset(phantom_name="AAA001"):
    """
    Calculate the offset between guidewire base and head when positioned at origin
    """
    print(f"Analyzing guidewire offset for {phantom_name}...")

    # Create a test environment
    env = gym.make("cathsim/CathSim-v0",
        phantom=phantom_name,
        target="goal",
        random_init_distance=0.0,
        use_pixels=False,
    )

    # Access the task
    dm_env = env.unwrapped._env
    task = dm_env._task
    physics = dm_env.physics

    # Set guidewire at origin
    test_position = np.array([0.0, 0.0, 0.0])
    task._guidewire.set_pose(physics, position=test_position)

    # Get the head position
    head_pos = task.get_head_pos(physics)

    # Calculate offset
    offset = head_pos - test_position

    print(f"Base position: {test_position}")
    print(f"Head position: {head_pos}")
    print(f"Offset (head - base): {offset}")
    print(f"Offset magnitude: {np.linalg.norm(offset):.6f} meters")

    env.close()

    return offset


def get_corrected_base_position(start_pos, offset):
    """
    Given the desired head position and the offset, calculate where to put the base
    """
    # base_pos + offset = start_pos (desired head position)
    # base_pos = start_pos - offset
    base_pos = start_pos - offset
    return base_pos


def update_aaa001_with_corrected_position(centerlines_dir, phantom_name="AAA001"):
    """
    Update AAA001 phantom with corrected start position that accounts for guidewire offset
    """
    if pv is None:
        raise ImportError("pyvista is required")

    # Load the current start position (where we want the HEAD to be)
    centerlines_path = centerlines_dir + f"/{phantom_name}"
    iliac_left = pv.read(f"{centerlines_path}/iliac_left.vtp")
    desired_head_pos_mm = iliac_left.points[0]
    desired_head_pos = desired_head_pos_mm / 1000.0  # Convert to meters

    print(f"\nDesired head position (iliac_left entry): {desired_head_pos}")

    # Calculate guidewire offset
    offset = calculate_guidewire_offset(phantom_name)

    # Calculate corrected base position
    corrected_base_pos = get_corrected_base_position(desired_head_pos, offset)

    print(f"\nCorrected base position (for phantom XML): {corrected_base_pos}")
    print(f"This will put the catheter HEAD at: {desired_head_pos}")

    # Update the phantom XML with the corrected position
    from pathlib import Path
    xml_path = Path(f"src/cathsim/dm/components/phantom_assets/{phantom_name}.xml")

    if not xml_path.exists():
        print(f"Error: {xml_path} not found")
        return

    # Read the XML
    with open(xml_path, 'r') as f:
        xml_content = f.read()

    # Find and replace the start site position
    import re
    # Match: <site name="start" pos="X Y Z" ...>
    pattern = r'(<site name="start" pos=")([^"]+)(".*?>)'
    old_pos = re.search(pattern, xml_content).group(2)

    print(f"\nOld start position in XML: {old_pos}")

    # Create new position string
    new_pos = f"{corrected_base_pos[0]:.6f} {corrected_base_pos[1]:.6f} {corrected_base_pos[2]:.6f}"
    print(f"New start position in XML: {new_pos}")

    # Replace in XML
    xml_content = re.sub(pattern, rf'\g<1>{new_pos}\g<3>', xml_content)

    # Backup original
    backup_path = xml_path.with_suffix('.xml.backup')
    print(f"\nBacking up original XML to {backup_path}")
    import shutil
    shutil.copy(xml_path, backup_path)

    # Write updated XML
    print(f"Writing updated XML to {xml_path}")
    with open(xml_path, 'w') as f:
        f.write(xml_content)

    print("\n[OK] AAA001 phantom updated with corrected start position!")
    print("The catheter head will now start exactly at the iliac_left entry point.")

    return corrected_base_pos, offset


if __name__ == "__main__":
    centerlines_dir = "C:/Users/Admin/Downloads/10932957/centerlines/centerlines"

    print("=" * 60)
    print("Fixing AAA001 Catheter Start Position")
    print("=" * 60)

    corrected_pos, offset = update_aaa001_with_corrected_position(
        centerlines_dir=centerlines_dir,
        phantom_name="AAA001"
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Guidewire offset: {offset}")
    print(f"  Corrected base position: {corrected_pos}")
    print("=" * 60)
