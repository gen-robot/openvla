#!/usr/bin/env python3
"""
Debug script to identify issues in RLDS datasets by comparing a working dataset with a problematic one.
Uses the OpenVLA dataset loading pipeline to replicate the exact issue.
"""

import argparse
import sys
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
import dlimp as dl

# Import OpenVLA dataset utilities
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from prismatic.vla.datasets.rlds.utils.data_utils import get_dataset_statistics
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights, OXE_NAMED_MIXTURES


def load_openvla_dataset(data_dir: str, task_name: str, train: bool = True, skip_statistic_computation: bool = False):
    """Load dataset using OpenVLA's pipeline."""
    try:
        # Check if it's a named mixture or single dataset
        if task_name in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[task_name]
        else:
            mixture_spec = [(task_name, 1.0)]
        
        # Configure camera views based on dataset type
        load_camera_views = dict()
        for name, _ in mixture_spec:
            if "aloha" in name or "cobot" in name:
                load_camera_views[name] = ("primary", "left_wrist", "right_wrist")
            elif "libero" in name:
                load_camera_views[name] = ("primary", "wrist")
            elif "bridge" in name:
                load_camera_views[name] = ("primary",)
            else:
                load_camera_views[name] = ("primary",)
        
        # Get dataset kwargs
        per_dataset_kwargs, weights, MAX_ACTION_DIM = get_oxe_dataset_kwargs_and_weights(
            Path(data_dir),
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=False,
            load_language=True,
        )
        
        print(f"Found {len(per_dataset_kwargs)} datasets in mixture")
        for i, kwargs in enumerate(per_dataset_kwargs):
            print(f"  Dataset {i}: {kwargs['name']}")
        
        # Load the first dataset for debugging
        dataset_kwargs = per_dataset_kwargs[0]
        print(f"Loading dataset: {dataset_kwargs['name']}")
        
        dataset, stats = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            shuffle=False,  # Important for debugging
            num_parallel_reads=1,  # Sequential for debugging
            num_parallel_calls=1,
            skip_statistic_computation=skip_statistic_computation,
        )
        
        return dataset, stats, dataset_kwargs
        
    except Exception as e:
        print(f"Error loading OpenVLA dataset: {e}")
        traceback.print_exc()
        return None, None, None


def analyze_dataset_structure(dataset, name: str, max_samples: int = 5):
    """Analyze the basic structure of a dataset."""
    print(f"\n{'='*50}")
    print(f"ANALYZING DATASET: {name}")
    print(f"{'='*50}")
    
    try:
        # Get dataset info
        print(f"Dataset cardinality: {dataset.cardinality().numpy()}")
        
        # Analyze first few samples
        print(f"\nAnalyzing first {max_samples} samples...")
        iterator = dataset.iterator()
        for i in range(max_samples):
            try:
                sample = next(iterator)
                print(f"\n--- Sample {i+1} ---")
                print(f"Top-level keys: {list(sample.keys())}")
                
                # Check key structures
                for key, value in sample.items():
                    if hasattr(value, 'shape'):
                        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                        # Check for problematic values
                        if key == 'action' and hasattr(value, 'numpy'):
                            action_np = value.numpy()
                            if np.any(np.isnan(action_np)):
                                print(f"  WARNING: Found NaN values in {key}")
                            if np.any(np.isinf(action_np)):
                                print(f"  WARNING: Found infinite values in {key}")
                            print(f"  Action range: [{np.min(action_np):.3f}, {np.max(action_np):.3f}]")
                    elif isinstance(value, dict):
                        print(f"{key}: dict with keys {list(value.keys())}")
                        for subkey, subvalue in value.items():
                            if hasattr(subvalue, 'shape'):
                                print(f"  {subkey}: shape={subvalue.shape}, dtype={subvalue.dtype}")
                    else:
                        print(f"{key}: type={type(value)}")
                        
            except StopIteration:
                print(f"Dataset exhausted after {i} samples")
                break
            except Exception as e:
                print(f"ERROR analyzing sample {i+1}: {e}")
                break
            
    except Exception as e:
        print(f"ERROR analyzing {name}: {e}")
        traceback.print_exc()


def find_problematic_record(dataset, name: str, start_from: int = 0):
    """Iterate through dataset to find the exact record that causes issues."""
    print(f"\n{'='*50}")
    print(f"SEARCHING FOR PROBLEMATIC RECORDS IN: {name}")
    print(f"{'='*50}")
    
    problematic_records = []
    record_count = 0
    
    try:
        # Get total cardinality if possible
        total_records = dataset.cardinality().numpy()
        if total_records == tf.data.UNKNOWN_CARDINALITY:
            total_records = None
            print("Dataset cardinality: UNKNOWN")
        else:
            print(f"Dataset cardinality: {total_records}")
        
        print(f"Starting search from record {start_from}...")
        
        # Skip to start position if needed
        if start_from > 0:
            dataset = dataset.skip(start_from)
            record_count = start_from
        
        # Create progress bar
        pbar = tqdm(desc=f"Processing {name}", unit="trajectories", total=total_records, initial=start_from)
        
        iterator = dataset.iterator()
        # Note: we use a while loop because the iterator can be advanced manually
        while total_records is None or record_count < total_records:
            try:
                pbar.set_description(f"Processing {name} - Trajectory {record_count}")
                
                # Try to get next trajectory
                trajectory = next(iterator)
                
                # Validate trajectory data
                traj_keys = list(trajectory.keys())
                
                # Check each key in the trajectory
                for key, value in trajectory.items():
                    if hasattr(value, 'numpy'):
                        # Force evaluation of tensor
                        tensor_value = value.numpy()
                        
                        # Check for problematic values
                        if key == 'action':
                            if np.any(np.isnan(tensor_value)):
                                print(f"\nWARNING: Found NaN in action at trajectory {record_count}")
                                print(f"Action shape: {tensor_value.shape}")
                                print(f"NaN positions: {np.where(np.isnan(tensor_value))}")
                            if np.any(np.isinf(tensor_value)):
                                print(f"\nWARNING: Found inf in action at trajectory {record_count}")
                                print(f"Action shape: {tensor_value.shape}")
                                print(f"Inf positions: {np.where(np.isinf(tensor_value))}")
                        
                    elif isinstance(value, dict):
                        # Handle nested dictionaries (like observation)
                        for subkey, subvalue in value.items():
                            if hasattr(subvalue, 'numpy'):
                                _ = subvalue.numpy()
                
                # Progress checkpoint every 1000 trajectories
                if (record_count + 1) % 1000 == 0:
                    print(f"\n✓ Processed {record_count + 1} trajectories so far...")
                
            except StopIteration:
                # Dataset exhausted normally
                break
                
            except (tf.errors.DataLossError, tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError) as e:
                print(f"\n❌ FOUND PROBLEMATIC TRAJECTORY!")
                print(f"Trajectory: {record_count}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                try:
                    print(f"Trajectory keys: {traj_keys}")
                except:
                    print("Could not access trajectory keys")
                problematic_records.append((record_count, None, e))
                
            except Exception as e:
                print(f"\n⚠️  UNEXPECTED ERROR in trajectory!")
                print(f"Trajectory: {record_count}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                traceback.print_exc()
                problematic_records.append((record_count, None, e))

            pbar.update(1)
            record_count += 1

        pbar.close()
        if not problematic_records:
            print(f"\n✅ Successfully processed entire dataset!")
            print(f"Total trajectories: {record_count}")

        return problematic_records
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR during iteration!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        problematic_records.append((record_count, None, e))
        return problematic_records


def compare_datasets(working_dataset, working_stats, problem_dataset, problem_stats):
    """Compare structure and properties of two datasets."""
    print(f"\n{'='*50}")
    print(f"COMPARING DATASETS")
    print(f"{'='*50}")
    
    # Compare basic properties
    working_card = working_dataset.cardinality().numpy()
    problem_card = problem_dataset.cardinality().numpy()
    
    print(f"Working dataset cardinality: {working_card}")
    print(f"Problematic dataset cardinality: {problem_card}")
    
    # Compare dataset statistics if available
    if working_stats and problem_stats:
        print(f"\nComparing dataset statistics...")
        
        # Compare action statistics
        if 'action' in working_stats and 'action' in problem_stats:
            print(f"Working action mean: {working_stats['action']['mean']}")
            print(f"Problem action mean: {problem_stats['action']['mean']}")
            print(f"Working action std: {working_stats['action']['std']}")
            print(f"Problem action std: {problem_stats['action']['std']}")
            print(f"Working action range: [{working_stats['action']['min']}, {working_stats['action']['max']}]")
            print(f"Problem action range: [{problem_stats['action']['min']}, {problem_stats['action']['max']}]")
        
        # Compare trajectory counts
        print(f"Working trajectories: {working_stats.get('num_trajectories', 'unknown')}")
        print(f"Problem trajectories: {problem_stats.get('num_trajectories', 'unknown')}")
        print(f"Working transitions: {working_stats.get('num_transitions', 'unknown')}")
        print(f"Problem transitions: {problem_stats.get('num_transitions', 'unknown')}")
    elif working_stats:
        print("\nWorking dataset statistics available, but not for problematic dataset (this is expected when debugging).")
    else:
        print("\nDataset statistics not available for comparison.")
    
    # Sample from both datasets to compare structure
    print(f"\nComparing first trajectory structure...")
    
    try:
        working_sample = next(iter(working_dataset.iterator()))
        print(f"Working dataset trajectory keys: {list(working_sample.keys())}")
        
        for key, value in working_sample.items():
            if hasattr(value, 'shape'):
                print(f"  Working {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"  Working {key}: dict with keys {list(value.keys())}")
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'shape'):
                        print(f"    {subkey}: shape={subvalue.shape}, dtype={subvalue.dtype}")
    except Exception as e:
        print(f"Error accessing working dataset: {e}")
    
    try:
        problem_sample = next(iter(problem_dataset.iterator()))
        print(f"Problematic dataset trajectory keys: {list(problem_sample.keys())}")
        
        for key, value in problem_sample.items():
            if hasattr(value, 'shape'):
                print(f"  Problem {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"  Problem {key}: dict with keys {list(value.keys())}")
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'shape'):
                        print(f"    {subkey}: shape={subvalue.shape}, dtype={subvalue.dtype}")
    except Exception as e:
        print(f"Error accessing problematic dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Debug RLDS dataset issues")
    parser.add_argument("--working_data_dir", type=str, required=True, 
                       help="Path to working dataset directory (PATH1)")
    parser.add_argument("--working_task", type=str, required=True,
                       help="Name of working RLDS task")
    parser.add_argument("--problem_data_dir", type=str, required=True,
                       help="Path to problematic dataset directory (PATH2)")
    parser.add_argument("--problem_task", type=str, required=True,
                       help="Name of problematic RLDS task")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start checking from this episode number")
    parser.add_argument("--analyze_structure", action="store_true",
                       help="Analyze dataset structure in detail")
    
    args = parser.parse_args()
    
    print("🔍 RLDS Dataset Debugging Tool")
    print(f"Working dataset: {args.working_task} in {args.working_data_dir}")
    print(f"Problematic dataset: {args.problem_task} in {args.problem_data_dir}")
    
    try:
        # Load datasets using OpenVLA pipeline
        print("\n📂 Loading datasets...")
        working_dataset, working_stats, working_kwargs = load_openvla_dataset(
            args.working_data_dir, 
            args.working_task, 
            train=True
        )
        if working_dataset is None:
            print("❌ Failed to load working dataset")
            return
        print("✅ Working dataset loaded successfully")
        
        problematic_dataset, problem_stats, problem_kwargs = load_openvla_dataset(
            args.problem_data_dir,
            args.problem_task,
            train=True
        )
        if problematic_dataset is None:
            print("❌ Failed to load problematic dataset, attempting to load without statistics...")
            problematic_dataset, problem_stats, problem_kwargs = load_openvla_dataset(
                args.problem_data_dir,
                args.problem_task,
                train=True,
                skip_statistic_computation=True
            )
            if problematic_dataset is None:
                print("❌ Failed to load problematic dataset even without statistics. The issue is severe.")
                return

        print("✅ Problematic dataset loaded successfully")
        
        # Analyze structure if requested
        if args.analyze_structure:
            analyze_dataset_structure(working_dataset, "WORKING DATASET")
            analyze_dataset_structure(problematic_dataset, "PROBLEMATIC DATASET")
        
        # Compare datasets
        compare_datasets(working_dataset, working_stats, problematic_dataset, problem_stats)
        
        # Search for problematic record
        problematic_records = find_problematic_record(
            problematic_dataset, 
            "PROBLEMATIC DATASET",
            start_from=args.start_from
        )
        
        if problematic_records:
            print(f"\n🎯 SUMMARY:")
            print(f"Found {len(problematic_records)} problematic trajectories.")
            for i, (idx, _, error) in enumerate(problematic_records):
                print(f"  {i+1}. Trajectory {idx}: Error: {type(error).__name__}")
            print(f"\nDataset kwargs: {problem_kwargs}")
        else:
            print(f"\n🤔 No issues found in the dataset.")
            print("The dataset may be working correctly, or the issue might be in the processing pipeline.")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 