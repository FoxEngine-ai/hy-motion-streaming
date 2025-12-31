# SMPL-H Data Printing Guide

## 🎯 Overview

This guide explains how to enable SMPL-H data printing in both the Streaming Gradio interface and the CLI. The feature prints detailed motion data for each frame, including:

- **Keypoints3D**: 3D joint positions (XYZ coordinates)
- **Translation**: Global translation vectors
- **Rotation6D**: 6D rotation representations
- **Root Rotations**: Root joint rotation matrices

## 🚀 Streaming Gradio Interface

### Automatic Printing

The streaming interface **automatically prints SMPL-H data** for each frame as it's generated. No additional configuration is needed.

### Example Output

```
=== Frame 5/50 SMPL-H Data ===
Progress: 10.0%
Keypoints3D shape: (1, 50, 22, 3) (Batch, Frames, Joints, XYZ)
First frame keypoints sample: tensor([[0.123, 0.456, 0.789], ...]) (first 3 joints)
Translation shape: (1, 50, 3) (Batch, Frames, XYZ)
First frame translation: [0.123, 0.456, 0.789]
Rotation6D shape: (1, 50, 22, 6) (Batch, Frames, Joints, 6D)
First frame root rotation: [0.123, 0.456, 0.789, 0.111, 0.222, 0.333]

=== Final Frame SMPL-H Data ===
Generation completed!
Final Keypoints3D shape: (1, 50, 22, 3)
Final frame keypoints sample: tensor([[0.999, 0.888, 0.777], ...]) (last frame, first 3 joints)
Final Translation shape: (1, 50, 3)
Final frame translation: [0.999, 0.888, 0.777]
Final Rotation6D shape: (1, 50, 22, 6)
Final frame root rotation: [0.999, 0.888, 0.777, 0.111, 0.222, 0.333]
```

### How to Use

1. **Run the streaming app**:
   ```bash
   make run-gradio-streaming
   ```

2. **Enter your prompt** and click "Generate Motion (Streaming)"

3. **Watch the console output** for frame-by-frame SMPL-H data

## 💻 CLI Interface

### Enable SMPL-H Data Printing

Use the `--print_smplh_data` flag to enable detailed motion data printing:

```bash
python local_infer.py --model_path ckpts/tencent/HY-Motion-1.0 --print_smplh_data
```

### Example Output

```
============================================================
SMPL-H Data for: 'A person walking forward'
============================================================
📊 Keypoints3D:
   Shape: (1, 50, 22, 3) (Batch=1, Frames=50, Joints=22)
   First frame, first 3 joints XYZ:
     Joint 0: [0.123, 0.456, 0.789]
     Joint 1: [0.111, 0.222, 0.333]
     Joint 2: [0.444, 0.555, 0.666]
   Last frame, first 3 joints XYZ:
     Joint 0: [0.999, 0.888, 0.777]
     Joint 1: [0.777, 0.666, 0.555]
     Joint 2: [0.444, 0.333, 0.222]

📊 Translation:
   Shape: (1, 50, 3) (Batch=1, Frames=50)
   First frame: [0.123, 0.456, 0.789]
   Last frame:  [0.999, 0.888, 0.777]

📊 Rotation6D:
   Shape: (1, 50, 22, 6) (Batch=1, Frames=50, Joints=22)
   First frame, root joint: [0.123 0.456 0.789 0.111 0.222 0.333]
   Last frame, root joint:  [0.999 0.888 0.777 0.111 0.222 0.333]

📊 Root Rotations:
   Shape: (1, 50, 3, 3) (Batch=1, Frames=50)
   First frame rotation matrix:
     [[0.999 0.001 0.002]
      [0.001 0.998 0.003]
      [0.002 0.003 0.997]]
============================================================
```

### CLI Usage Examples

```bash
# Basic usage with SMPL-H data printing
python local_infer.py --model_path ckpts/tencent/HY-Motion-1.0 --print_smplh_data

# With additional options
python local_infer.py \
    --model_path ckpts/tencent/HY-Motion-1.0 \
    --print_smplh_data \
    --disable_rewrite \
    --num_seeds 2

# Using Makefile
make run-cli PRINT_SMPLH_DATA=true
```

## 📊 Data Format Explanation

### Keypoints3D
- **Shape**: `(Batch, Frames, Joints, 3)`
- **Description**: 3D positions of all joints in meters
- **Joints**: Typically 22 joints for SMPL-H (including hands)
- **Coordinates**: XYZ in world space

### Translation
- **Shape**: `(Batch, Frames, 3)`
- **Description**: Global translation of the root joint
- **Coordinates**: XYZ displacement from origin

### Rotation6D
- **Shape**: `(Batch, Frames, Joints, 6)`
- **Description**: 6D rotation representation (more stable than quaternions)
- **Format**: Continuous 6D representation of 3D rotations

### Root Rotations
- **Shape**: `(Batch, Frames, 3, 3)`
- **Description**: Rotation matrices for the root joint
- **Format**: 3×3 orthogonal rotation matrices

## 🔧 Advanced Usage

### Save SMPL-H Data to File

```bash
# Redirect output to file
python local_infer.py --model_path ckpts/tencent/HY-Motion-1.0 --print_smplh_data > smplh_data.txt

# Process with Python
python -c "
import json
import re

# Extract data from output file
with open('smplh_data.txt', 'r') as f:
    content = f.read()
    
# Parse keypoints, translations, etc.
# (Add your parsing logic here)
"
```

### Analyze Motion Data

```python
# Example analysis script
import numpy as np
import torch

# Load motion data (example)
def analyze_motion_data(output_dict):
    k3d = output_dict['keypoints3d']  # (1, L, 22, 3)
    transl = output_dict['transl']    # (1, L, 3)
    
    # Calculate motion statistics
    motion_range = k3d.max() - k3d.min()
    motion_speed = torch.diff(k3d, dim=1).abs().mean()
    
    print(f"Motion range: {motion_range}")
    print(f"Motion speed: {motion_speed}")
    
    return {
        'range': motion_range,
        'speed': motion_speed,
        'trajectory_length': transl.norm(dim=-1).sum()
    }
```

## 🎯 Use Cases

### 1. **Debugging Motion Generation**
- Verify joint positions and orientations
- Check for unrealistic movements
- Validate motion smoothness

### 2. **Motion Analysis**
- Calculate motion statistics
- Analyze trajectory patterns
- Extract motion features

### 3. **Data Export**
- Save motion data for offline analysis
- Convert to other formats (BVH, FBX)
- Create motion datasets

### 4. **Research & Development**
- Study motion generation quality
- Compare different models
- Develop new motion metrics

## 🔄 Integration with Other Tools

### Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

# Plot joint trajectories
def plot_joint_trajectories(k3d_data, joint_idx=0):
    """Plot XYZ trajectories of a specific joint."""
    k3d = k3d_data[0].cpu().numpy()  # (L, 22, 3)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(k3d[:, joint_idx, 0], label='X')
    plt.title('X Position')
    
    plt.subplot(132)
    plt.plot(k3d[:, joint_idx, 1], label='Y', color='g')
    plt.title('Y Position')
    
    plt.subplot(133)
    plt.plot(k3d[:, joint_idx, 2], label='Z', color='r')
    plt.title('Z Position')
    
    plt.suptitle(f'Joint {joint_idx} Trajectory')
    plt.tight_layout()
    plt.show()
```

### Motion Comparison
```python
# Compare two motion sequences
def compare_motions(k3d_1, k3d_2):
    """Calculate similarity between two motion sequences."""
    diff = torch.abs(k3d_1 - k3d_2)
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Max difference: {max_diff:.4f}")
    
    return mean_diff, max_diff
```

## 📋 Summary

| Feature | Streaming Gradio | CLI |
|---------|----------------|-----|
| **Automatic Printing** | ✅ Yes | ❌ No |
| **Manual Control** | ❌ No | ✅ `--print_smplh_data` |
| **Frame-by-Frame** | ✅ Yes | ✅ Yes (per generation) |
| **Data Format** | SMPL-H | SMPL-H |
| **Output** | Console | Console/File |

## 🎓 Technical Notes

- **SMPL-H Model**: Uses 22 joints (body + hands)
- **Coordinate System**: Right-handed (X-right, Y-up, Z-forward)
- **Units**: Meters for positions, radians for rotations
- **Batch Processing**: Supports multiple seeds in one generation

## 🚨 Troubleshooting

### "Model output is not a dictionary"
- **Cause**: The motion generation failed
- **Solution**: Check model loading and input parameters

### No SMPL-H data printed
- **Streaming**: Ensure the app is running and generating motion
- **CLI**: Make sure `--print_smplh_data` flag is used

### Incomplete data
- **Cause**: Motion generation was interrupted
- **Solution**: Check for errors and try again

The SMPL-H data printing feature provides comprehensive insights into the motion generation process, enabling debugging, analysis, and research applications! 🎬