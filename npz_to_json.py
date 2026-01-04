import numpy as np
import json
import os
import glob

def convert_npz_to_json(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load the npz file
    data = np.load(input_path, allow_pickle=True)
    
    # Debug: See exactly what we are working with
    available_keys = data.files
    print(f"Found keys in NPZ: {available_keys}")

    # Initialize our export dictionary
    export_dict = {}

    # 1. Handle Poses (The rotations)
    if 'poses' in data:
        p = data['poses'].astype(np.float32)
        export_dict["frameCount"] = int(p.shape[0])
        export_dict["poses"] = p.flatten().tolist()
        print(f"Added 'poses': {p.shape}")
    
    # 2. Handle Translation (The movement)
    if 'trans' in data:
        t = data['trans'].astype(np.float32)
        export_dict["trans"] = t.flatten().tolist()
        print(f"Added 'trans': {t.shape}")
    
    # 3. Handle Root Rotation (Rh) - Often useful if 'poses' is local-only
    if 'Rh' in data:
        r = data['Rh'].astype(np.float32)
        export_dict["Rh"] = r.flatten().tolist()
        print(f"Added 'Rh': {r.shape}")

    # Write to JSON with indentation so you can read it in a text editor
    with open(output_path, 'w') as f:
        json.dump(export_dict, f, indent=4)
        
    print(f"\nSUCCESS: Created {output_path}")
    print(f"Final JSON Keys: {list(export_dict.keys())}")

def process_all_pairs(input_dir, output_dir):
    """Process all npz/meta.json pairs from input_dir and save to output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npz files
    npz_files = glob.glob(os.path.join(input_dir, "**", "*.npz"), recursive=True)
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files to process\n")
    
    for npz_path in npz_files:
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        
        # Check for corresponding meta.json
        meta_path = npz_path.replace('.npz', '').rsplit('_', 1)[0] + '_meta.json'
        has_meta = os.path.exists(meta_path)
        
        print(f"Processing: {base_name}")
        print(f"  NPZ: {npz_path}")
        print(f"  Meta: {meta_path} ({'found' if has_meta else 'not found'})")
        
        # Determine output filename from meta.json text field
        output_name = base_name  # fallback
        if has_meta:
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            if 'text' in meta_data:
                # Use text as filename, replace spaces with underscores
                output_name = meta_data['text'].replace(' ', '_')
                # Remove any problematic characters for filenames
                output_name = "".join(c for c in output_name if c.isalnum() or c in '_-')
        
        # Convert npz to json
        output_path = os.path.join(output_dir, f"{output_name}.json")
        convert_npz_to_json(npz_path, output_path)
        
        print()

if __name__ == "__main__":
    input_dir = "output"
    output_dir = "output_converted"
    process_all_pairs(input_dir, output_dir)
