"""
Example script demonstrating how to load and use SMPL-H data exported as .npy files.

This script shows how to:
1. Load the .npy files containing SMPL-H data
2. Access poses, translations, and other parameters
3. Visualize or process the motion data
"""

import numpy as np
import os
from pathlib import Path


def load_smplh_npy(npy_path: str) -> dict:
    """
    Load SMPL-H data from a .npy file.
    
    Args:
        npy_path: Path to the .npy file
        
    Returns:
        Dictionary containing SMPL-H data with keys:
        - poses: (num_frames, 156) - axis-angle rotations for all joints
        - trans: (num_frames, 3) - global translations
        - betas: (1, 16) - shape parameters
        - gender: str - gender ('neutral', 'male', 'female')
        - Rh: (num_frames, 3) - root rotation
        - mocap_framerate: int - frame rate (usually 30)
        - num_frames: int - number of frames
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"NPY file not found: {npy_path}")
    
    smplh_data = np.load(npy_path, allow_pickle=True).item()
    
    print(f"Loaded SMPL-H data from: {npy_path}")
    print(f"  - Number of frames: {smplh_data['num_frames']}")
    print(f"  - Frame rate: {smplh_data['mocap_framerate']} fps")
    print(f"  - Gender: {smplh_data['gender']}")
    print(f"  - Poses shape: {smplh_data['poses'].shape}")
    print(f"  - Translations shape: {smplh_data['trans'].shape}")
    
    return smplh_data


def analyze_motion(smplh_data: dict):
    """
    Analyze the motion data and print statistics.
    
    Args:
        smplh_data: Dictionary containing SMPL-H data
    """
    poses = smplh_data['poses']
    trans = smplh_data['trans']
    
    print("\n=== Motion Analysis ===")
    print(f"Duration: {smplh_data['num_frames'] / smplh_data['mocap_framerate']:.2f} seconds")
    
    # Analyze joint rotations
    joint_rotations = poses.reshape(smplh_data['num_frames'], -1, 3)  # (frames, joints, 3)
    print(f"Number of joints: {joint_rotations.shape[1]}")
    
    # Calculate rotation magnitudes
    rot_magnitudes = np.linalg.norm(joint_rotations, axis=2)
    print(f"Average rotation magnitude: {np.mean(rot_magnitudes):.4f}")
    print(f"Max rotation magnitude: {np.max(rot_magnitudes):.4f}")
    
    # Analyze translations
    trans_magnitude = np.linalg.norm(trans, axis=1)
    print(f"Total displacement: {np.sum(trans_magnitude):.4f}")
    print(f"Average displacement per frame: {np.mean(trans_magnitude):.4f}")
    print(f"Max displacement in single frame: {np.max(trans_magnitude):.4f}")


def save_to_text_format(smplh_data: dict, output_path: str):
    """
    Save SMPL-H data to a human-readable text format.
    
    Args:
        smplh_data: Dictionary containing SMPL-H data
        output_path: Path to save the text file
    """
    with open(output_path, 'w') as f:
        f.write("SMPL-H Motion Data\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Gender: {smplh_data['gender']}\n")
        f.write(f"Number of frames: {smplh_data['num_frames']}\n")
        f.write(f"Frame rate: {smplh_data['mocap_framerate']} fps\n")
        f.write(f"Duration: {smplh_data['num_frames'] / smplh_data['mocap_framerate']:.2f} seconds\n\n")
        
        f.write("Shape parameters (betas):\n")
        f.write(str(smplh_data['betas'].flatten()) + "\n\n")
        
        f.write("Translations (first 5 frames):\n")
        for i in range(min(5, smplh_data['num_frames'])):
            f.write(f"Frame {i}: {smplh_data['trans'][i]}\n")
        
        if smplh_data['num_frames'] > 5:
            f.write(f"... ({smplh_data['num_frames'] - 5} more frames)\n")
    
    print(f"Saved text representation to: {output_path}")


def main():
    """Main function demonstrating usage."""
    
    # Example: Load a generated NPY file
    # Replace this with the actual path to your generated file
    npy_path = "output/gradio/20240104_123456_000_smplh.npy"
    
    # Check if file exists, otherwise use a placeholder
    if not os.path.exists(npy_path):
        print(f"Example file not found: {npy_path}")
        print("Please generate a motion with NPY format first, or update the path.")
        print("\nTo generate NPY files:")
        print("1. Run the Gradio UI")
        print("2. Select 'npy' as the output format")
        print("3. Generate a motion")
        print("4. Download the generated .npy files")
        return
    
    # Load the data
    smplh_data = load_smplh_npy(npy_path)
    
    # Analyze the motion
    analyze_motion(smplh_data)
    
    # Save to text format for inspection
    txt_output = npy_path.replace("_smplh.npy", "_analysis.txt")
    save_to_text_format(smplh_data, txt_output)
    
    print("\n=== Usage Examples ===")
    print("# Access poses (axis-angle rotations):")
    print("poses = smplh_data['poses']  # Shape: (num_frames, 156)")
    print()
    print("# Access translations:")
    print("trans = smplh_data['trans']  # Shape: (num_frames, 3)")
    print()
    print("# Access shape parameters:")
    print("betas = smplh_data['betas']  # Shape: (1, 16)")
    print()
    print("# Reshape poses to (frames, joints, 3):")
    print("joint_rotations = poses.reshape(num_frames, 52, 3)")
    print()
    print("# Use with SMPL-H body model:")
    print("# from smplx import SMPLH")
    print("# model = SMPLH(model_path='path/to/smplh', gender=smplh_data['gender'])")
    print("# output = model(poses=poses, trans=trans, betas=betas)")


if __name__ == "__main__":
    main()