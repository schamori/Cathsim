"""
Fix iliac_left centerline by removing/trimming the closed end
This script reads the iliac_left centerline, removes some points from the start to "open it up",
and saves the modified centerline.
"""

import pyvista as pv
from pathlib import Path
import numpy as np


def fix_iliac_left_centerline(centerlines_dir: str, phantom_name: str, points_to_remove: int = 5):
    """
    Fix the iliac_left centerline by removing points from the start.

    Args:
        centerlines_dir: Directory containing centerline VTP files
        phantom_name: Name of the phantom (e.g., "AAA001")
        points_to_remove: Number of points to remove from the start (default: 5)
    """
    centerlines_path = Path(centerlines_dir) / phantom_name
    iliac_left_path = centerlines_path / "iliac_left.vtp"

    if not iliac_left_path.exists():
        print(f"Error: {iliac_left_path} not found")
        return

    print(f"Loading iliac_left from {iliac_left_path}")
    iliac_left = pv.read(str(iliac_left_path))

    print(f"Original iliac_left: {len(iliac_left.points)} points")
    print(f"First point: {iliac_left.points[0]}")
    print(f"Point {points_to_remove}: {iliac_left.points[points_to_remove]}")

    # Remove the first N points to "open" the vessel
    new_points = iliac_left.points[points_to_remove:]

    # Create a new polydata with the trimmed points
    new_centerline = pv.PolyData(new_points)

    # If the original has lines, we need to update them too
    if iliac_left.lines is not None and len(iliac_left.lines) > 0:
        # Lines are stored as: [n_points, point_0, point_1, ..., point_n-1, n_points, ...]
        # We need to rebuild the lines with adjusted indices
        n_points = len(new_points)
        lines = np.zeros(n_points + 1, dtype=np.int64)
        lines[0] = n_points
        lines[1:] = np.arange(n_points)
        new_centerline.lines = lines

    print(f"Modified iliac_left: {len(new_centerline.points)} points")
    print(f"New first point (start): {new_centerline.points[0]}")

    # Save the modified centerline
    backup_path = centerlines_path / "iliac_left_original.vtp"
    output_path = centerlines_path / "iliac_left.vtp"

    # Backup the original
    print(f"Backing up original to {backup_path}")
    iliac_left.save(str(backup_path))

    # Save the modified version
    print(f"Saving modified iliac_left to {output_path}")
    new_centerline.save(str(output_path))

    print(f"[OK] iliac_left fixed successfully!")
    print(f"  - Removed {points_to_remove} points from the start")
    print(f"  - New start position: {new_centerline.points[0]}")


if __name__ == "__main__":
    import sys

    # Default parameters
    centerlines_dir = "C:/Users/Admin/Downloads/10932957/centerlines/centerlines"

    if len(sys.argv) < 2:
        print("Usage: python fix_iliac_left.py <phantom_name> [points_to_remove]")
        print("\nExample:")
        print("  python fix_iliac_left.py AAA001 5")
        print("  python fix_iliac_left.py AAA003 10")
        print("\nThis will remove N points from the start of iliac_left to 'open it up'")
        sys.exit(1)

    phantom_name = sys.argv[1]
    points_to_remove = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    fix_iliac_left_centerline(centerlines_dir, phantom_name, points_to_remove)
