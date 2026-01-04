#!/usr/bin/env python3
"""
Example script demonstrating how to load and use JSON motion files
exported from HY-Motion-1.0.
"""

import json
import numpy as np
import os
from typing import Dict, Any


def load_json_motion(json_path: str) -> Dict[str, Any]:
    """
    Load a JSON motion file exported from HY-Motion-1.0.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary containing motion data and metadata
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Reshape flattened arrays back to their original shapes
    frame_count = data['frameCount']
    
    if 'poses' in data:
        data['poses_array'] = np.array(data['poses']).reshape(frame_count, 156)
    
    if 'trans' in data:
        data['trans_array'] = np.array(data['trans']).reshape(frame_count, 3)
    
    if 'Rh' in data:
        data['Rh_array'] = np.array(data['Rh']).reshape(frame_count, 3)
    
    return data


def analyze_motion(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze motion data and extract statistics.
    
    Args:
        data: Motion data dictionary from load_json_motion()
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        'text': data.get('text', 'Unknown'),
        'timestamp': data.get('timestamp', 'Unknown'),
        'batch_index': data.get('batch_index', 0),
    }
    
    if 'poses_array' in data:
        poses = data['poses_array']
        analysis['frame_count'] = poses.shape[0]
        analysis['duration_seconds'] = poses.shape[0] / 30.0  # Assuming 30fps
        analysis['joint_count'] = poses.shape[1] // 3  # 3 values per joint
        
        # Rotation statistics
        analysis['rotation_stats'] = {
            'mean': float(poses.mean()),
            'std': float(poses.std()),
            'min': float(poses.min()),
            'max': float(poses.max()),
        }
    
    if 'trans_array' in data:
        trans = data['trans_array']
        
        # Translation statistics
        analysis['translation_stats'] = {
            'mean': trans.mean(axis=0).tolist(),
            'std': trans.std(axis=0).tolist(),
            'min': trans.min(axis=0).tolist(),
            'max': trans.max(axis=0).tolist(),
        }
        
        # Calculate total distance traveled
        distances = np.linalg.norm(np.diff(trans, axis=0), axis=1)
        analysis['total_distance'] = float(distances.sum())
        analysis['average_speed'] = float(distances.mean() * 30)  # m/s at 30fps
    
    if 'Rh_array' in data:
        rh = data['Rh_array']
        analysis['root_rotation_stats'] = {
            'mean': rh.mean(axis=0).tolist(),
            'std': rh.std(axis=0).tolist(),
        }
    
    return analysis


def visualize_trajectory(data: Dict[str, Any], save_path: str = None):
    """
    Visualize the motion trajectory using matplotlib.
    
    Args:
        data: Motion data dictionary from load_json_motion()
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    if 'trans_array' not in data:
        print("No translation data available for visualization")
        return
    
    trans = data['trans_array']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Motion Analysis: {data.get('text', 'Unknown')}", fontsize=14)
    
    # Plot 1: XYZ over time
    axes[0, 0].plot(trans[:, 0], label='X', alpha=0.7)
    axes[0, 0].plot(trans[:, 1], label='Y', alpha=0.7)
    axes[0, 0].plot(trans[:, 2], label='Z', alpha=0.7)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].set_title('Translation Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: 2D trajectory (X-Y plane)
    axes[0, 1].plot(trans[:, 0], trans[:, 1], alpha=0.7)
    axes[0, 1].scatter(trans[0, 0], trans[0, 1], c='green', s=100, label='Start', zorder=5)
    axes[0, 1].scatter(trans[-1, 0], trans[-1, 1], c='red', s=100, label='End', zorder=5)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('2D Trajectory (X-Y Plane)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Plot 3: 3D trajectory
    ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax_3d.plot(trans[:, 0], trans[:, 1], trans[:, 2], alpha=0.7)
    ax_3d.scatter(trans[0, 0], trans[0, 1], trans[0, 2], c='green', s=100, label='Start', zorder=5)
    ax_3d.scatter(trans[-1, 0], trans[-1, 1], trans[-1, 2], c='red', s=100, label='End', zorder=5)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Trajectory')
    ax_3d.legend()
    
    # Plot 4: Speed over time
    if len(trans) > 1:
        distances = np.linalg.norm(np.diff(trans, axis=0), axis=1)
        speed = distances * 30  # Convert to m/s at 30fps
        axes[1, 1].plot(speed, alpha=0.7)
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Speed (m/s)')
        axes[1, 1].set_title('Speed Over Time')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def convert_json_to_npy(json_path: str, npy_path: str = None):
    """
    Convert a JSON motion file to NPY format.
    
    Args:
        json_path: Path to the JSON file
        npy_path: Path to save the NPY file (default: same as JSON but .npy extension)
    """
    data = load_json_motion(json_path)
    
    if npy_path is None:
        npy_path = json_path.replace('.json', '.npy')
    
    # Create SMPL-H dictionary
    smplh_dict = {
        'poses': data['poses_array'],
        'trans': data['trans_array'],
        'Rh': data['Rh_array'],
        'frameCount': data['frameCount'],
        'text': data.get('text', ''),
        'timestamp': data.get('timestamp', ''),
    }
    
    np.save(npy_path, smplh_dict)
    print(f"Converted {json_path} to {npy_path}")


def main():
    """Main function demonstrating usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and analyze JSON motion files")
    parser.add_argument('json_file', help="Path to JSON motion file")
    parser.add_argument('--visualize', action='store_true', help="Visualize trajectory")
    parser.add_argument('--convert-npy', action='store_true', help="Convert to NPY format")
    parser.add_argument('--output', help="Output path for visualization or NPY file")
    
    args = parser.parse_args()
    
    # Load JSON file
    print(f"Loading JSON file: {args.json_file}")
    data = load_json_motion(args.json_file)
    
    # Print basic info
    print(f"\n{'='*60}")
    print(f"Motion Information")
    print(f"{'='*60}")
    print(f"Text: {data.get('text', 'Unknown')}")
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    print(f"Batch Index: {data.get('batch_index', 0)}")
    print(f"Frame Count: {data.get('frameCount', 0)}")
    
    # Analyze motion
    print(f"\n{'='*60}")
    print(f"Motion Analysis")
    print(f"{'='*60}")
    analysis = analyze_motion(data)
    
    if 'duration_seconds' in analysis:
        print(f"Duration: {analysis['duration_seconds']:.2f} seconds")
        print(f"Joint Count: {analysis['joint_count']}")
    
    if 'total_distance' in analysis:
        print(f"Total Distance: {analysis['total_distance']:.3f} meters")
        print(f"Average Speed: {analysis['average_speed']:.3f} m/s")
    
    if 'translation_stats' in analysis:
        stats = analysis['translation_stats']
        print(f"\nTranslation Statistics:")
        print(f"  Mean: [{stats['mean'][0]:.3f}, {stats['mean'][1]:.3f}, {stats['mean'][2]:.3f}]")
        print(f"  Std:  [{stats['std'][0]:.3f}, {stats['std'][1]:.3f}, {stats['std'][2]:.3f}]")
        print(f"  Range: [{stats['min'][0]:.3f}, {stats['min'][1]:.3f}, {stats['min'][2]:.3f}] to")
        print(f"         [{stats['max'][0]:.3f}, {stats['max'][1]:.3f}, {stats['max'][2]:.3f}]")
    
    # Visualize if requested
    if args.visualize:
        print(f"\nGenerating visualization...")
        visualize_trajectory(data, save_path=args.output)
    
    # Convert to NPY if requested
    if args.convert_npy:
        print(f"\nConverting to NPY format...")
        convert_json_to_npy(args.json_file, args.output)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()