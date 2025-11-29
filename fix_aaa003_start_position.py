"""
Fix the catheter starting position for AAA003
Apply the same guidewire offset correction as AAA001
"""

import re
from pathlib import Path


def fix_aaa003_start_position():
    """
    Update AAA003 phantom XML with corrected start position that accounts for guidewire offset
    """
    phantom_name = "AAA003"

    # The guidewire offset is consistent: [0, 0.0185, 0] meters (18.5mm in Y direction)
    offset_y = 0.0185

    xml_path = Path(f"src/cathsim/dm/components/phantom_assets/{phantom_name}.xml")

    if not xml_path.exists():
        print(f"Error: {xml_path} not found")
        return

    # Read the XML
    with open(xml_path, 'r') as f:
        xml_content = f.read()

    # Find the current start site position
    pattern = r'<site name="start" pos="([^"]+)"'
    match = re.search(pattern, xml_content)

    if not match:
        print("Error: Could not find start site in XML")
        return

    current_pos = match.group(1)
    pos_values = [float(x) for x in current_pos.split()]

    print(f"Current start position: {pos_values}")

    # Apply the offset: move backwards in Y direction to compensate for guidewire length
    corrected_pos = [pos_values[0], pos_values[1] - offset_y, pos_values[2]]

    print(f"Corrected start position: {corrected_pos}")
    print(f"Applied offset: [0, {-offset_y}, 0] (move base back so head is at iliac_left)")

    # Create new position string
    new_pos = f"{corrected_pos[0]:.6f} {corrected_pos[1]:.6f} {corrected_pos[2]:.6f}"

    # Replace in XML
    xml_content = re.sub(
        r'(<site name="start" pos=")([^"]+)(")',
        rf'\g<1>{new_pos}\g<3>',
        xml_content
    )

    # Backup original
    backup_path = xml_path.with_suffix('.xml.backup2')
    print(f"\nBacking up original XML to {backup_path}")
    import shutil
    shutil.copy(xml_path, backup_path)

    # Write updated XML
    print(f"Writing updated XML to {xml_path}")
    with open(xml_path, 'w') as f:
        f.write(xml_content)

    print(f"\n[OK] {phantom_name} phantom updated with corrected start position!")
    print("The catheter head will now start exactly at the iliac_left entry point.")


if __name__ == "__main__":
    print("=" * 60)
    print("Fixing AAA003 Catheter Start Position")
    print("=" * 60)

    fix_aaa003_start_position()

    print("=" * 60)
