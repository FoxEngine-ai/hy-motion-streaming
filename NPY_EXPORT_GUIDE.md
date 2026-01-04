# SMPL-H NPY Export Guide

This guide explains how to export and use raw SMPL-H motion data as `.npy` files from the HY-Motion-1.0 Gradio UI.

## Overview

The Gradio UI now supports three output formats:
- **FBX**: 3D animation files for use in Blender, Maya, etc.
- **NPY**: Raw SMPL-H data in NumPy format for programmatic use
- **Dict**: Internal format (no file export)

## Exporting NPY Files

### Step 1: Select NPY Format

1. Open the Gradio UI
2. In the "Advanced Settings" section, locate the "Output Format" radio button
3. Select **npy** as the output format

### Step 2: Generate Motion

1. Enter your text prompt
2. Click "Rewrite Text" (if available) to optimize the prompt
3. Click "Generate Motion"
4. Wait for the generation to complete

### Step 3: Download Files

After generation completes, you'll see a download section with:
- `.npy` files containing SMPL-H data for each generated animation
- `.txt` files containing the original text prompt

## NPY File Format

Each `.npy` file contains a dictionary with the following keys:

### Data Structure

```python
{
    'poses': np.ndarray,           # Shape: (num_frames, 156)
                                   # Axis-angle rotations for all 52 joints
                                   # Each joint has 3 values (x, y, z)
    
    'trans': np.ndarray,           # Shape: (num_frames, 3)
                                   # Global translations (x, y, z)
    
    'betas': np.ndarray,           # Shape: (1, 16)
                                   # Shape parameters for body morphology
    
    'gender': str,                 # 'neutral', 'male', or 'female'
    
    'Rh': np.ndarray,              # Shape: (num_frames, 3)
                                   # Root rotation (axis-angle)
    
    'mocap_framerate': int,        # Frame rate (usually 30 fps)
    
    'num_frames': int              # Total number of frames
}
```

### Joint Order

The 52 joints in SMPL-H follow this order:
1. **Body (22 joints)**: Root, spine, shoulders, arms, legs, etc.
2. **Left Hand (15 joints)**: Wrist + 4 fingers × 3 joints each
3. **Right Hand (15 joints)**: Wrist + 4 fingers × 3 joints each

## Loading NPY Files

### Basic Loading

```python
import numpy as np

# Load the NPY file
smplh_data = np.load('path/to/file_smplh.npy', allow_pickle=True).item()

# Access data
poses = smplh_data['poses']      # (num_frames, 156)
trans = smplh_data['trans']      # (num_frames, 3)
betas = smplh_data['betas']      # (1, 16)
gender = smplh_data['gender']    # 'neutral'
```

### Reshaping Poses

```python
# Reshape to (frames, joints, 3) for easier manipulation
num_frames = smplh_data['num_frames']
joint_rotations = smplh_data['poses'].reshape(num_frames, 52, 3)

# Access specific joint
root_rotation = joint_rotations[:, 0, :]  # All frames, root joint
left_wrist = joint_rotations[:, 22, :]    # All frames, left wrist
```

## Using with SMPL-H Body Model

### Using SMPL-X Library

```python
import torch
from smplx import SMPLH

# Load SMPL-H model
model = SMPLH(
    model_path='path/to/smplh/models',
    gender=smplh_data['gender'],
    num_betas=16
)

# Convert to tensors
poses_tensor = torch.tensor(smplh_data['poses'], dtype=torch.float32)
trans_tensor = torch.tensor(smplh_data['trans'], dtype=torch.float32)
betas_tensor = torch.tensor(smplh_data['betas'], dtype=torch.float32)

# Generate mesh
output = model(
    body_pose=poses_tensor[:, 3:],  # Exclude root rotation
    global_orient=poses_tensor[:, :3],  # Root rotation
    transl=trans_tensor,
    betas=betas_tensor
)

vertices = output.vertices  # (num_frames, 6890, 3)
joints = output.joints      # (num_frames, 52, 3)
```

### Using with PyTorch3D

```python
import torch
from pytorch3d.io import save_obj

# Save as OBJ for each frame
for frame_idx in range(num_frames):
    frame_vertices = vertices[frame_idx]  # (6890, 3)
    save_obj(
        f'frame_{frame_idx:04d}.obj',
        verts=frame_vertices,
        faces=model.faces
    )
```

## Example Scripts

### Load and Analyze Motion

See [`examples/load_npy_example.py`](examples/load_npy_example.py) for a complete example:

```bash
python examples/load_npy_example.py
```

### Convert to Other Formats

```python
import numpy as np
import json

# Load NPY
smplh_data = np.load('motion_smplh.npy', allow_pickle=True).item()

# Convert to JSON
json_data = {
    'poses': smplh_data['poses'].tolist(),
    'trans': smplh_data['trans'].tolist(),
    'betas': smplh_data['betas'].tolist(),
    'gender': smplh_data['gender'],
    'num_frames': int(smplh_data['num_frames']),
    'framerate': int(smplh_data['mocap_framerate'])
}

with open('motion.json', 'w') as f:
    json.dump(json_data, f, indent=2)
```

### Visualize Motion

```python
import matplotlib.pyplot as plt
import numpy as np

smplh_data = np.load('motion_smplh.npy', allow_pickle=True).item()
trans = smplh_data['trans']

# Plot trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(trans[:, 0], trans[:, 1], trans[:, 2], 'b-', linewidth=2)
ax.scatter(trans[0, 0], trans[0, 1], trans[0, 2], c='g', s=100, label='Start')
ax.scatter(trans[-1, 0], trans[-1, 1], trans[-1, 2], c='r', s=100, label='End')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Motion Trajectory')
ax.legend()

plt.tight_layout()
plt.savefig('trajectory.png')
plt.show()
```

## Data Processing Tips

### Smooth Motion

```python
from scipy.ndimage import gaussian_filter1d

# Smooth translations
sigma = 2.0  # Adjust for more/less smoothing
trans_smooth = np.zeros_like(trans)
for i in range(3):
    trans_smooth[:, i] = gaussian_filter1d(trans[:, i], sigma=sigma)
```

### Resample Motion

```python
from scipy import interpolate

# Change frame rate
original_fps = 30
target_fps = 60

original_time = np.arange(num_frames) / original_fps
target_time = np.arange(int(num_frames * target_fps / original_fps)) / target_fps

# Interpolate translations
trans_resampled = np.zeros((len(target_time), 3))
for i in range(3):
    f = interpolate.interp1d(original_time, trans[:, i], kind='cubic')
    trans_resampled[:, i] = f(target_time)
```

### Extract Joint Angles

```python
# Extract specific joint rotations
joint_rotations = smplh_data['poses'].reshape(num_frames, 52, 3)

# Joint indices (example)
ROOT = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 14
LEFT_ELBOW = 12
RIGHT_ELBOW = 15

# Get joint angles over time
left_shoulder_angles = joint_rotations[:, LEFT_SHOULDER, :]
right_shoulder_angles = joint_rotations[:, RIGHT_SHOULDER, :]
```

## Troubleshooting

### File Not Found

Ensure you've generated motion with NPY format selected. Check the `output/gradio/` directory.

### Shape Mismatch

Verify the expected shapes:
- `poses`: (num_frames, 156)
- `trans`: (num_frames, 3)
- `betas`: (1, 16)

### Import Errors

Install required packages:
```bash
pip install numpy scipy matplotlib
```

For SMPL-H body model:
```bash
pip install smplx
```

## Additional Resources

- [SMPL-X GitHub](https://github.com/vchoutas/smplx)
- [NumPy Documentation](https://numpy.org/doc/)
- [Example Script](examples/load_npy_example.py)

## Support

For issues or questions:
1. Check the example script in `examples/load_npy_example.py`
2. Review the SMPL-H documentation
3. Open an issue on the project repository