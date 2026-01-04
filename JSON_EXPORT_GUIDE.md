# JSON Export Guide for HY-Motion-1.0

## Overview

HY-Motion-1.0 now supports exporting generated motions as JSON files, making it easy to integrate with other applications and workflows.

## JSON Format

The exported JSON files contain the following structure:

```json
{
    "frameCount": 150,
    "poses": [/* flattened rotation data */],
    "trans": [/* flattened translation data */],
    "Rh": [/* flattened root rotation data */],
    "text": "A person is running",
    "timestamp": "20240104_123456",
    "batch_index": 0
}
```

### Fields

- **`frameCount`**: Number of frames in the motion
- **`poses`**: Flattened array of rotation data (axis-angle format)
  - Shape: `(num_frames, 156)` flattened to 1D array
  - Contains joint rotations for all 52 SMPL-H joints (3 values per joint)
- **`trans`**: Flattened array of translation data
  - Shape: `(num_frames, 3)` flattened to 1D array
  - Contains XYZ translation for each frame
- **`Rh`**: Flattened array of root rotation data
  - Shape: `(num_frames, 3)` flattened to 1D array
  - Contains root joint rotation for each frame
- **`text`**: The text prompt used to generate the motion
- **`timestamp`**: Generation timestamp
- **`batch_index`**: Index in the batch (0 for single generation)

## Using JSON Export in Gradio UI

1. **Select Output Format**: Choose "json" from the "📁 Output Format" dropdown
2. **Generate Motion**: Click "🚀 Generate Motion" as usual
3. **Download Files**: The generated JSON files will appear in the download section

### Example Workflow

```
1. Enter text: "A person jumps and lands"
2. Click "🔄 Rewrite Text" (optional)
3. Select "json" from Output Format dropdown
4. Click "🚀 Generate Motion"
5. Download the generated JSON files
```

## File Naming

JSON files are named using the pattern:
```
{timestamp}_{batch_index:03d}_{text_sanitized}.json
```

Example:
```
20240104_123456_000_A_person_jumps_and_lands.json
```

A corresponding prompt text file is also generated:
```
20240104_123456_000_A_person_jumps_and_lands_prompt.txt
```

## Programmatic Usage

### Loading JSON Files

```python
import json
import numpy as np

# Load JSON file
with open('motion.json', 'r') as f:
    data = json.load(f)

# Extract data
frame_count = data['frameCount']
poses = np.array(data['poses']).reshape(frame_count, 156)
trans = np.array(data['trans']).reshape(frame_count, 3)
rh = np.array(data['Rh']).reshape(frame_count, 3)

print(f"Loaded {frame_count} frames")
print(f"Poses shape: {poses.shape}")
print(f"Translation shape: {trans.shape}")
```

### Converting to Other Formats

```python
# Convert JSON to NPZ
import numpy as np

with open('motion.json', 'r') as f:
    data = json.load(f)

smplh_dict = {
    'poses': np.array(data['poses']).reshape(data['frameCount'], 156),
    'trans': np.array(data['trans']).reshape(data['frameCount'], 3),
    'Rh': np.array(data['Rh']).reshape(data['frameCount'], 3),
}

np.save('motion.npz', smplh_dict)
```

## Comparison with Other Formats

| Format | Size | Readability | Use Case |
|--------|------|-------------|----------|
| **JSON** | Large | High | Web apps, APIs, human-readable |
| **NPY** | Small | Low | Python workflows, ML pipelines |
| **FBX** | Medium | Low | 3D software (Blender, Maya) |
| **Dict** | N/A | High | In-memory Python objects |

## Advantages of JSON Format

1. **Human-Readable**: Easy to inspect and debug
2. **Universal**: Supported by all programming languages
3. **Web-Friendly**: Can be directly used in web applications
4. **API-Ready**: Perfect for REST APIs and microservices
5. **Version Control**: Can be tracked in Git (for small motions)

## Disadvantages of JSON Format

1. **Large File Size**: Text-based format is larger than binary formats
2. **Slower Loading**: Parsing JSON is slower than loading binary formats
3. **No Compression**: Files are not compressed by default

## Best Practices

1. **For Production Use**: Use NPY or FBX for better performance
2. **For Development/Debugging**: Use JSON for easy inspection
3. **For Web Applications**: Use JSON for direct browser consumption
4. **For Large Motions**: Consider compression or binary formats

## Troubleshooting

### Issue: JSON files are too large

**Solution**: Use NPY format instead, or compress JSON files:
```bash
gzip motion.json
```

### Issue: Loading JSON is slow

**Solution**: Convert to NPY format once and use that:
```python
import json
import numpy as np

# One-time conversion
with open('motion.json', 'r') as f:
    data = json.load(f)

smplh_dict = {
    'poses': np.array(data['poses']).reshape(data['frameCount'], 156),
    'trans': np.array(data['trans']).reshape(data['frameCount'], 3),
    'Rh': np.array(data['Rh']).reshape(data['frameCount'], 3),
}

np.save('motion.npz', smplh_dict)
```

### Issue: JSON parsing errors

**Solution**: Ensure the JSON file is valid:
```python
import json

try:
    with open('motion.json', 'r') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

## API Reference

### `_generate_json_files()`

```python
def _generate_json_files(
    self,
    visualization_data: dict,
    output_dir: Optional[str] = None,
    json_filename: Optional[str] = None,
) -> List[str]:
    """
    Generate .json files from SMPL-H data for each animation.
    
    Args:
        visualization_data: Dictionary containing SMPL data and metadata
        output_dir: Directory to save .json files (default: output/gradio)
        json_filename: Base filename for the .json files
        
    Returns:
        List of paths to generated .json files
    """
```

## Example: Complete Workflow

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. Load JSON file
with open('motion.json', 'r') as f:
    data = json.load(f)

# 2. Extract motion data
frame_count = data['frameCount']
poses = np.array(data['poses']).reshape(frame_count, 156)
trans = np.array(data['trans']).reshape(frame_count, 3)

# 3. Analyze motion
print(f"Motion duration: {frame_count / 30:.2f} seconds @ 30fps")
print(f"Translation range: {trans.min():.3f} to {trans.max():.3f}")

# 4. Visualize trajectory
plt.figure(figsize=(10, 6))
plt.plot(trans[:, 0], label='X')
plt.plot(trans[:, 1], label='Y')
plt.plot(trans[:, 2], label='Z')
plt.xlabel('Frame')
plt.ylabel('Position')
plt.title('Motion Trajectory')
plt.legend()
plt.grid(True)
plt.show()

# 5. Save analysis
analysis = {
    'duration': frame_count / 30,
    'translation_range': {
        'min': trans.min().tolist(),
        'max': trans.max().tolist()
    },
    'text': data['text']
}

with open('motion_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=4)
```

## Support

For issues or questions about JSON export:
- Check the console output for detailed logging
- Verify the JSON file structure using a JSON validator
- Refer to the main HY-Motion-1.0 documentation