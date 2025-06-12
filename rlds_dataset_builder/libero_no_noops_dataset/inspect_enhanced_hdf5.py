#!/usr/bin/env python3
"""
Inspection script for Libero no-noops HDF5 files to identify enhanced features.

This script analyzes HDF5 files to determine whether they contain enhanced features
(motion descriptions, segmentation labels) or require fallback processing.
"""

import argparse
import glob
import h5py
import os
import sys
from collections import defaultdict


def inspect_hdf5_file(file_path):
    """Inspect a single HDF5 file and return feature information."""
    file_info = {
        'file_path': file_path,
        'has_enhanced_motion': False,
        'has_enhanced_seg_labels': False,
        'num_episodes': 0,
        'total_steps': 0,
        'sample_motion_descriptions': [],
        'sample_seg_labels': '',
        'episodes_info': []
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            data_group = f['data']
            episode_keys = [key for key in data_group.keys() if key.startswith('demo_')]
            file_info['num_episodes'] = len(episode_keys)
            
            for episode_key in episode_keys:
                episode_data = data_group[episode_key]
                episode_info = {
                    'episode_key': episode_key,
                    'num_steps': len(episode_data['actions'][:]),
                    'has_motion_descriptions': 'motion_descriptions' in episode_data,
                    'has_seg_labels': 'seg_labels' in episode_data,
                }
                
                file_info['total_steps'] += episode_info['num_steps']
                
                # Check for enhanced motion descriptions
                if 'motion_descriptions' in episode_data:
                    file_info['has_enhanced_motion'] = True
                    episode_info['has_motion_descriptions'] = True
                    
                    # Sample some motion descriptions
                    if len(file_info['sample_motion_descriptions']) < 5:
                        motion_desc_bytes = episode_data['motion_descriptions'][:]
                        motion_descriptions = [
                            desc.decode('utf-8') if isinstance(desc, bytes) else str(desc) 
                            for desc in motion_desc_bytes
                        ]
                        # Take first few motion descriptions as samples
                        sample_size = min(3, len(motion_descriptions))
                        file_info['sample_motion_descriptions'].extend(
                            motion_descriptions[:sample_size]
                        )
                
                # Check for enhanced segmentation labels
                if 'seg_labels' in episode_data:
                    file_info['has_enhanced_seg_labels'] = True
                    episode_info['has_seg_labels'] = True
                    
                    if not file_info['sample_seg_labels']:
                        seg_labels_data = episode_data['seg_labels'][()]
                        if isinstance(seg_labels_data, bytes):
                            file_info['sample_seg_labels'] = seg_labels_data.decode('utf-8')
                        else:
                            file_info['sample_seg_labels'] = str(seg_labels_data)
                
                file_info['episodes_info'].append(episode_info)
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    return file_info


def print_file_summary(file_info):
    """Print a summary of a single file's features."""
    filename = os.path.basename(file_info['file_path'])
    
    print(f"\n📁 {filename}")
    print(f"   Episodes: {file_info['num_episodes']}, Total steps: {file_info['total_steps']}")
    
    # Enhanced features status
    motion_status = "✅ Enhanced" if file_info['has_enhanced_motion'] else "❌ Fallback"
    seg_status = "✅ Enhanced" if file_info['has_enhanced_seg_labels'] else "❌ Fallback"
    
    print(f"   Motion descriptions: {motion_status}")
    print(f"   Segmentation labels: {seg_status}")
    
    # Show samples if available
    if file_info['sample_motion_descriptions']:
        print(f"   Sample motions: {file_info['sample_motion_descriptions'][:3]}")
    
    if file_info['sample_seg_labels']:
        # Truncate long seg labels for display
        seg_labels = file_info['sample_seg_labels']
        if len(seg_labels) > 80:
            seg_labels = seg_labels[:80] + "..."
        print(f"   Seg labels: {seg_labels}")


def print_detailed_summary(all_files_info):
    """Print detailed summary statistics."""
    total_files = len(all_files_info)
    enhanced_motion_files = sum(1 for f in all_files_info if f['has_enhanced_motion'])
    enhanced_seg_files = sum(1 for f in all_files_info if f['has_enhanced_seg_labels'])
    fully_enhanced_files = sum(1 for f in all_files_info 
                               if f['has_enhanced_motion'] and f['has_enhanced_seg_labels'])
    
    total_episodes = sum(f['num_episodes'] for f in all_files_info)
    total_steps = sum(f['total_steps'] for f in all_files_info)
    
    print(f"\n{'='*60}")
    print(f"📊 DETAILED SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {total_files}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"")
    print(f"Enhanced motion descriptions: {enhanced_motion_files}/{total_files} files ({enhanced_motion_files/total_files*100:.1f}%)")
    print(f"Enhanced segmentation labels: {enhanced_seg_files}/{total_files} files ({enhanced_seg_files/total_files*100:.1f}%)")
    print(f"Fully enhanced (both features): {fully_enhanced_files}/{total_files} files ({fully_enhanced_files/total_files*100:.1f}%)")
    
    # Collect unique motion vocabularies
    all_motions = set()
    all_seg_objects = set()
    
    for file_info in all_files_info:
        all_motions.update(file_info['sample_motion_descriptions'])
        if file_info['sample_seg_labels']:
            all_seg_objects.update(file_info['sample_seg_labels'].split('|'))
    
    print(f"\n📝 Motion vocabulary diversity: {len(all_motions)} unique motions")
    if len(all_motions) > 0:
        sample_motions = sorted(list(all_motions))[:10]
        print(f"   Sample motions: {sample_motions}")
    
    print(f"\n🏷️ Segmentation object diversity: {len(all_seg_objects)} unique objects")
    if len(all_seg_objects) > 0:
        # Filter out empty strings and sort
        objects = sorted([obj for obj in all_seg_objects if obj.strip()])[:10]
        print(f"   Sample objects: {objects}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect Libero HDF5 files for enhanced features'
    )
    parser.add_argument(
        'data_dir', 
        help='Directory containing HDF5 files or glob pattern'
    )
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed per-file information'
    )
    parser.add_argument(
        '--summary-only', '-s',
        action='store_true', 
        help='Show only summary statistics'
    )
    
    args = parser.parse_args()
    
    # Find HDF5 files
    if os.path.isdir(args.data_dir):
        pattern = os.path.join(args.data_dir, '*_demo.hdf5')
    else:
        pattern = args.data_dir
    
    hdf5_files = glob.glob(pattern)
    
    if not hdf5_files:
        print(f"❌ No HDF5 files found matching pattern: {pattern}")
        sys.exit(1)
    
    print(f"🔍 Inspecting {len(hdf5_files)} HDF5 files...")
    print(f"Pattern: {pattern}")
    
    # Inspect all files
    all_files_info = []
    for file_path in sorted(hdf5_files):
        file_info = inspect_hdf5_file(file_path)
        if file_info:
            all_files_info.append(file_info)
    
    if not all_files_info:
        print("❌ No valid HDF5 files found or all files had errors.")
        sys.exit(1)
    
    # Print results
    if not args.summary_only:
        print(f"\n{'='*60}")
        print(f"📋 FILE-BY-FILE ANALYSIS")
        print(f"{'='*60}")
        
        for file_info in all_files_info:
            print_file_summary(file_info)
            
            if args.detailed:
                print("   Episode details:")
                for ep_info in file_info['episodes_info']:
                    motion_flag = "✅" if ep_info['has_motion_descriptions'] else "❌"
                    seg_flag = "✅" if ep_info['has_seg_labels'] else "❌"
                    print(f"     {ep_info['episode_key']}: {ep_info['num_steps']} steps, "
                          f"Motion:{motion_flag}, Seg:{seg_flag}")
    
    # Always print detailed summary
    print_detailed_summary(all_files_info)
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*60}")
    
    fully_enhanced = sum(1 for f in all_files_info 
                        if f['has_enhanced_motion'] and f['has_enhanced_seg_labels'])
    total_files = len(all_files_info)
    
    if fully_enhanced == total_files:
        print("🎉 All files have enhanced features! Your dataset is ready for optimal performance.")
        print("   The RLDS dataset builder will use real motion descriptions and segmentation labels.")
    elif fully_enhanced > 0:
        print(f"⚠️  Mixed dataset: {fully_enhanced}/{total_files} files have enhanced features.")
        print("   Consider regenerating remaining files with regenerate_libero_dataset_enhanced.py")
        print("   for consistent high-quality motion descriptions and segmentation labels.")
    else:
        print("❗ No enhanced features found. All files will use fallback processing.")
        print("   Recommendation: Regenerate your dataset using regenerate_libero_dataset_enhanced.py")
        print("   for improved motion descriptions and real segmentation labels.")
        print("\n   Example command:")
        print("   python experiments/robot/libero/regenerate_libero_dataset_enhanced.py \\")
        print("       --libero_task_suite libero_10 \\")
        print("       --libero_raw_data_dir ./LIBERO/libero/datasets/libero_10 \\")
        print("       --libero_target_dir ./datasets/libero_10_no_noops_enhanced")
    
    print(f"\n🏗️  Building RLDS dataset:")
    print(f"   export LIBERO_DATA_DIR='{os.path.dirname(hdf5_files[0])}'")
    print(f"   tfds build libero_no_noops_dataset")


if __name__ == "__main__":
    main() 