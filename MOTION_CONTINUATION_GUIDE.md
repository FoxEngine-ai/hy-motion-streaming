# Motion Continuation Guide

## Overview

The motion continuation feature allows you to generate new motion that seamlessly continues from an existing motion file. This is useful for:
- Extending existing animations
- Creating longer sequences from shorter clips
- Building complex multi-stage motions
- Iterative motion refinement

## How It Works

1. **Upload a JSON file** containing an existing motion
2. **Enable continuation mode** using the checkbox
3. **Generate new motion** that starts from the last frame of the uploaded motion
4. The system concatenates the original motion with the newly generated motion

## Using the Feature

### Step 1: Generate an Initial Motion

First, generate a motion using the standard generation process:

1. Enter your text prompt (e.g., "A person walking forward")
2. Set duration, seed, and CFG scale
3. Click "Generate Motion (Streaming)"
4. Download the generated JSON file

### Step 2: Continue from the Generated Motion

1. Click on the "🔄 Motion Continuation" accordion to expand it
2. Upload the JSON file you downloaded in step 1
3. Check the "Enable Continuation Mode" checkbox
4. Enter a new text prompt for the continuation (e.g., "then starts running")
5. Click "Generate Motion (Streaming)"

The system will:
- Load the original motion from the JSON file
- Extract the last frame as the starting point
- Generate new motion based on your new prompt
- Concatenate the original motion with the new motion
- Return the combined motion as a single file

## JSON File Format

The continuation feature expects JSON files in the standard HY-Motion format:

```json
{
  "frameCount": 150,
  "poses": [[...156 values per frame...]],
  "trans": [[x, y, z] per frame],
  "Rh": [[rx, ry, rz] per frame],
  "text": "Original motion description",
  "timestamp": "1234567890",
  "batchIndex": 0
}
```

### Key Fields

- **frameCount**: Number of frames in the motion
- **poses**: Body pose parameters (156 values per frame)
- **trans**: Root translation (3 values per frame)
- **Rh**: Root rotation (3 values per frame)
- **text**: Original text prompt used to generate the motion

## Example Workflow

### Example 1: Walking to Running

```python
# Step 1: Generate walking motion
prompt = "A person walking forward at a moderate pace"
duration = 5.0  # 5 seconds
# Generate and download as walking.json

# Step 2: Continue to running
prompt = "then starts running faster"
duration = 3.0  # 3 seconds
# Upload walking.json, enable continuation
# Result: 8-second motion (5s walking + 3s running)
```

### Example 2: Dance Sequence

```python
# Step 1: Initial dance move
prompt = "A person doing a pirouette"
duration = 2.0
# Generate and download as pirouette.json

# Step 2: Add next move
prompt = "then transitions into a leap"
duration = 2.0
# Upload pirouette.json, enable continuation
# Generate and download as pirouette_leap.json

# Step 3: Add final move
prompt = "then lands gracefully"
duration = 1.5
# Upload pirouette_leap.json, enable continuation
# Result: 5.5-second dance sequence
```

## Technical Details

### Concatenation Process

The continuation feature works by:

1. **Loading the original motion** from the JSON file
2. **Extracting all frames** from the original motion
3. **Generating new motion** based on the new prompt
4. **Concatenating** the original motion with the new motion along the frame dimension

```python
# Original motion from JSON: (orig_frames, 156) - poses
# Reshaped to rot6d format: (orig_frames, 22, 6)
# New motion: (1, new_frames, 22, 6) - rot6d
# Combined motion: (1, orig_frames + new_frames, 22, 6)
combined_rot6d = np.concatenate([orig_rot6d, new_rot6d], axis=1)
```

**Note**: The JSON format uses `poses` (156 values per frame) while the internal representation uses `rot6d` (22 joints × 6 values = 132 values per frame). The system automatically converts between these formats during concatenation.

### Limitations

1. **No temporal smoothing**: The transition between the original and new motion may not be perfectly smooth
2. **No conditioning on last frame**: The new motion is generated independently, not conditioned on the last frame of the original motion
3. **Keypoints not updated**: The keypoints3d field is not regenerated for the combined motion (visualization handles this)
4. **Format conversion**: The system converts between JSON format (poses: 156 values) and internal format (rot6d: 22×6 values), which may result in slight differences

### Future Improvements

Potential enhancements for the continuation feature:

1. **Temporal smoothing**: Apply smoothing at the transition point
2. **Frame conditioning**: Use the last frame as conditioning for the new motion generation
3. **Transition blending**: Blend the last few frames of the original motion with the first few frames of the new motion
4. **Automatic duration adjustment**: Adjust the new motion duration based on the original motion

## Troubleshooting

### Issue: "Error loading continuation JSON"

**Cause**: The JSON file is not in the correct format or is corrupted.

**Solution**:
- Verify the JSON file contains all required fields (frameCount, poses, trans)
- Check that the poses array has the correct shape (frameCount × 156)
- Ensure the trans array has the correct shape (frameCount × 3)
- The Rh field is optional and not used in continuation mode

### Issue: Motion doesn't look continuous

**Cause**: The new motion is generated independently, not conditioned on the last frame.

**Solution**:
- This is expected behavior with the current implementation
- Try using similar prompts for better continuity
- Consider using shorter durations for smoother transitions

### Issue: Combined motion is too long

**Cause**: Concatenating multiple motions can result in very long sequences.

**Solution**:
- Use shorter durations for each continuation step
- Consider breaking the sequence into multiple files
- Use the standard generation for shorter motions

## API Usage

You can also use the continuation feature programmatically:

```python
import json
import numpy as np
from gradio_app_streaming import streaming_generate_motion

# Load the original motion
with open('original_motion.json', 'r') as f:
    original_data = json.load(f)

# Generate continuation
for html, files in streaming_generate_motion(
    text="then starts running",
    seeds_csv="42",
    motion_duration=3.0,
    cfg_scale=7.0,
    output_format="json",
    continue_from_json='original_motion.json',
    continue_mode=True,
):
    # Process streaming updates
    pass

# The final files list will contain the combined motion
```

## Best Practices

1. **Start with shorter motions**: Begin with 2-3 second motions for easier continuation
2. **Use descriptive prompts**: Clear, specific prompts help with continuity
3. **Match motion styles**: Use similar motion styles for better transitions
4. **Test transitions**: Generate and review each continuation step before adding more
5. **Keep backups**: Save intermediate results in case you need to retry

## Related Features

- **JSON Export**: Save motions in JSON format for continuation
- **NPY Export**: Export raw SMPL-H data for programmatic access
- **FBX Export**: Export to FBX for use in 3D animation software
- **Streaming Generation**: Watch motion generation in real-time

## Version History

- **v1.1**: Fixed field name compatibility issues
  - Corrected motion_data field names (rot6d, transl instead of poses, trans)
  - Added proper format conversion between JSON and internal representation
  - Fixed Gradio component compatibility (removed unsupported `info` parameter)
  - Updated documentation with correct technical details

- **v1.0**: Initial implementation of motion continuation feature
  - JSON file upload
  - Basic concatenation
  - Streaming generation support

## Support

For issues or questions about the motion continuation feature:
1. Check this guide for common solutions
2. Review the JSON_EXPORT_GUIDE.md for JSON format details
3. Examine the console output for error messages
4. Test with simple examples first

## Version History

- **v1.0**: Initial implementation of motion continuation feature
  - JSON file upload
  - Basic concatenation
  - Streaming generation support