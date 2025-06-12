#!/usr/bin/env python3

import h5py
import numpy as np
import os
import glob

def inspect_enhanced_hdf5(hdf5_path):
    """Inspect an enhanced HDF5 file to see the motion descriptions and segmentation labels."""
    
    print(f"\n=== Inspecting: {hdf5_path} ===")
    
    with h5py.File(hdf5_path, 'r') as f:
        print(f"Root groups: {list(f.keys())}")
        
        if 'data' in f:
            data_group = f['data']
            episodes = [key for key in data_group.keys() if key.startswith('demo_')]
            print(f"Found {len(episodes)} episodes: {episodes[:3]}{'...' if len(episodes) > 3 else ''}")
            
            # Look at first episode
            if episodes:
                episode_key = episodes[0]
                episode_data = data_group[episode_key]
                print(f"\nEpisode {episode_key} structure:")
                print(f"  Keys: {list(episode_data.keys())}")
                
                # Check for enhanced features
                if 'motion_descriptions' in episode_data:
                    motion_desc = episode_data['motion_descriptions'][:]
                    print(f"\n  Motion descriptions ({len(motion_desc)} steps):")
                    for i, desc in enumerate(motion_desc[:5]):  # Show first 5
                        if isinstance(desc, bytes):
                            desc = desc.decode('utf-8')
                        print(f"    Step {i}: {desc}")
                    if len(motion_desc) > 5:
                        print(f"    ... and {len(motion_desc) - 5} more")
                else:
                    print("\n  No motion_descriptions found (using fallback)")
                
                if 'seg_labels' in episode_data:
                    seg_labels = episode_data['seg_labels'][()]
                    if isinstance(seg_labels, bytes):
                        seg_labels = seg_labels.decode('utf-8')
                    print(f"\n  Segmentation labels: {seg_labels}")
                else:
                    print("\n  No seg_labels found (using fallback)")
                
                # Show observation structure
                if 'obs' in episode_data:
                    obs_group = episode_data['obs']
                    print(f"\n  Observation keys: {list(obs_group.keys())}")
                    
                    # Show shapes
                    for key in obs_group.keys():
                        data = obs_group[key]
                        print(f"    {key}: {data.shape} {data.dtype}")
                
                # Show actions
                if 'actions' in episode_data:
                    actions = episode_data['actions'][:]
                    print(f"\n  Actions: {actions.shape} {actions.dtype}")
                    print(f"    First action: {actions[0]}")
                    print(f"    Last action: {actions[-1]}")

def main():
    """Main function to inspect enhanced HDF5 files."""
    
    # Check if LIBERO_DATA_DIR is set
    data_dir = os.environ.get('LIBERO_DATA_DIR', None)
    if data_dir is None:
        print("LIBERO_DATA_DIR environment variable is not set.")
        print("Please set it to point to your libero no-noops dataset directory.")
        return
    
    print(f"Looking for HDF5 files in: {data_dir}")
    
    # Find HDF5 files
    hdf5_pattern = os.path.join(data_dir, '*_demo.hdf5')
    hdf5_files = glob.glob(hdf5_pattern)
    
    if not hdf5_files:
        print(f"No HDF5 files found matching pattern: {hdf5_pattern}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Inspect first few files
    for hdf5_file in hdf5_files[:3]:  # Just inspect first 3 files
        try:
            inspect_enhanced_hdf5(hdf5_file)
        except Exception as e:
            print(f"Error inspecting {hdf5_file}: {e}")
    
    print("\n=== Summary ===")
    print("If you see 'motion_descriptions' and 'seg_labels' in the files above,")
    print("then you have enhanced HDF5 files and the dataset builder will use real data.")
    print("\nIf you see 'No motion_descriptions found' and 'No seg_labels found',")
    print("then you have basic HDF5 files and the dataset builder will use fallback placeholders.")
    print("\nTo generate enhanced HDF5 files, run:")
    print("python experiments/robot/libero/regenerate_libero_dataset_enhanced.py \\")
    print("  --libero_task_suite <SUITE> \\")
    print("  --libero_raw_data_dir <RAW_DATA_DIR> \\")
    print("  --libero_target_dir <TARGET_DIR>")

if __name__ == "__main__":
    main() 