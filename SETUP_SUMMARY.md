# AAA001 and AAA003 Setup Summary

## What was done

### 1. Updated AAA001 Phantom
- **Location**: `src/cathsim/dm/components/phantom_assets/AAA001.xml`
- **Changes**:
  - Added `start` site at iliac_left entry point (first point of iliac_left centerline)
  - Added `goal` site at end of abdominal aorta (last point of abdominal_aorta centerline)
  - Fixed iliac_left centerline by removing 5 points from the start to "open it up"
  - **Cut mesh to remove everything before the start position**

**Start position (iliac_left)**: `[0.024084, -0.161127, 0.848294]` meters
**Goal position (end of abdominal aorta)**: `[0.003177, -0.154139, 1.053505]` meters
**Mesh cut**: Removed 3129 vertices and 6258 faces before the start position

### 2. Created AAA003 Phantom
- **Location**: `src/cathsim/dm/components/phantom_assets/AAA003.xml`
- **Changes**:
  - Created phantom from STL file
  - Added `start` site at iliac_left entry point (first point of iliac_left centerline)
  - Added `goal` site at end of abdominal aorta (last point of abdominal_aorta centerline)
  - Fixed iliac_left centerline by removing 5 points from the start to "open it up"
  - **Cut mesh to remove everything before the start position**

**Start position (iliac_left)**: `[0.044097, 0.164242, -0.488647]` meters
**Goal position (end of abdominal aorta)**: `[0.009994, 0.174523, -0.292637]` meters
**Mesh cut**: Removed 4592 vertices and 9184 faces before the start position

### 3. Fixed iliac_left Centerlines
- Removed 5 points from the start of iliac_left centerlines for both AAA001 and AAA003
- This "opens up" the vessel entry point so the catheter can start properly
- Original centerlines backed up to `iliac_left_original.vtp`

### 4. Cut Meshes at Start Position
- Used a cutting plane at the iliac_left entry point to remove all mesh geometry before the start
- This ensures the catheter starts right at the vessel opening with no geometry blocking it
- Original meshes backed up to `AAA001_original.stl` and `AAA003_original.stl`
- Cutting plane oriented perpendicular to vessel direction for clean cut

## Scripts Created

1. **update_aaa001.py** - Updates AAA001 phantom with centerline-based sites
2. **create_aaa003.py** - Creates AAA003 phantom with centerline-based sites
3. **fix_iliac_left.py** - Fixes iliac_left centerlines by removing closed end points
4. **cut_mesh_at_start.py** - Cuts mesh to remove everything before the start position
5. **test_aaa001.py** - Tests AAA001 phantom with correct start and goal positions
6. **test_aaa003.py** - Tests AAA003 phantom with correct start and goal positions

## How to Use

### Basic Usage
```python
import cathsim.gym.envs
import gymnasium as gym

# Create environment with AAA001 or AAA003
env = gym.make("cathsim/CathSim-v0",
    phantom="AAA001",  # or "AAA003"
    target="goal",     # Use the goal site (end of abdominal aorta)
    random_init_distance=0.0,  # Start exactly at the start site
    visualize_sites=True,
    visualize_target=True,
)
```

### Custom Initialization (Start at iliac_left)
```python
# Access the task and sites
dm_env = env.unwrapped._env
task = dm_env._task
phantom = task._phantom
sites = phantom.sites

# Get start and goal positions
start_pos = np.array(sites["start"])
goal_pos = np.array(sites["goal"])

# Override initialization to start at iliac_left
def custom_initialize_episode(physics, random_state):
    task._guidewire.set_pose(physics, position=start_pos)
    task.success = False

task.initialize_episode = custom_initialize_episode
```

## Files Modified

### Phantom Assets
- `src/cathsim/dm/components/phantom_assets/AAA001.xml` - Updated with centerline sites
- `src/cathsim/dm/components/phantom_assets/AAA003.xml` - Created with centerline sites
- `src/cathsim/dm/components/phantom_assets/meshes/AAA001/` - Regenerated meshes
- `src/cathsim/dm/components/phantom_assets/meshes/AAA003/` - Created meshes

### Centerlines (Modified in Downloads)
- `C:/Users/Admin/Downloads/10932957/centerlines/centerlines/AAA001/iliac_left.vtp` - Trimmed 5 points
- `C:/Users/Admin/Downloads/10932957/centerlines/centerlines/AAA003/iliac_left.vtp` - Trimmed 5 points
- Backups saved as `iliac_left_original.vtp`

## Problems Fixed

1. ✅ **Random catheter start position** - Now starts at iliac_left (start site)
2. ✅ **Incorrect goal position** - Now targets end of abdominal aorta (goal site)
3. ✅ **iliac_left closed/cut** - Removed 5 points from start to open the vessel
4. ✅ **AAA003 not configured** - Created AAA003 with proper centerline-based sites
5. ✅ **Mesh geometry before start** - Cut away all mesh parts before the start position

## Test Results

Both AAA001 and AAA003 tested successfully:
- Catheter starts at iliac_left entry point
- Goal is set to end of abdominal aorta
- Distance from start to goal: ~0.2 meters (20cm)
- Environment runs without errors
