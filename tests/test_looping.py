
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from hymotion.utils.t2m_runtime import T2MRuntime

def test_looping_logic():
    print("Testing looping logic...")
    
    # Mock motion data (Batch=1, Frames=10, Joints=2, 6D)
    # Let's say we have 10 frames.
    # Frame 0: All zeros (except rotation maybe)
    # Frame 9: All ones
    
    frames = 30
    blend_frames = 10 # 1/3 of the movement
    
    # Create fake rot6d [1, frames, joints, 6]
    rot6d = torch.zeros((1, frames, 1, 6))
    rot6d[:, :, :, 0] = 1.0 # identity-ish first component
    
    # Create fake transl [1, frames, 3]
    transl = torch.zeros((1, frames, 3))
    
    # Make the last part of motion drift away
    for i in range(frames):
        transl[0, i, 0] = i * 0.1 # Drift in X
        
    print(f"Original end translation X: {transl[0, -1, 0]}")
    print(f"Original start translation X: {transl[0, 0, 0]}")
    
    motion_data = {
        'rot6d': rot6d,
        'transl': transl
    }
    
    # Apply looping
    # blend_duration = blend_frames / fps
    fps = 30.0
    blend_duration = blend_frames / fps
    
    print(f"Applying loop with blend frames={blend_frames}...")
    T2MRuntime.apply_looping(motion_data, blend_duration=blend_duration, fps=fps)
    
    new_transl = motion_data['transl']
    
    # Verify the last frame matches the first frame
    print(f"New end translation X: {new_transl[0, -1, 0]}")
    print(f"New start translation X: {new_transl[0, 0, 0]}")
    
    # Check if we blended towards zero
    assert torch.abs(new_transl[0, -1, 0] - new_transl[0, 0, 0]) < 1e-6, "End frame should match start frame!"
    
    # Check intermediate frames in blend region
    # The frame just before blending should be untouched?
    # No, we blend the last N frames.
    # Start blending at index: total - blend
    # index = 20. Frame 19 should be original (1.9). Frame 20 should be slightly blended.
    
    idx_before_blend = frames - blend_frames - 1
    # Check if untouched frame is still roughly correct
    val_before = new_transl[0, idx_before_blend, 0] 
    print(f"Frame {idx_before_blend} (before blend) X: {val_before}")
    
    print("✅ Looping logic test passed!")

if __name__ == "__main__":
    test_looping_logic()
