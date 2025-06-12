#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import os

def inspect_libero_lm_90():
    """Inspect the libero_lm_90 dataset to understand its structure."""
    
    print("=== Inspecting libero_lm_90 Dataset ===\n")
    
    # Try to load from tfrecord files directly since the high-level API had issues
    try:
        tfrecord_pattern = "datasets/libero_data/libero_lm_90/1.0.0/libero_lm_90-train.tfrecord-*"
        tfrecord_files = tf.io.gfile.glob(tfrecord_pattern)
        
        if not tfrecord_files:
            print(f"No tfrecord files found matching: {tfrecord_pattern}")
            return
        
        print(f"Found {len(tfrecord_files)} tfrecord files")
        print("Loading first tfrecord file for inspection...\n")
        
        # Load from first tfrecord file
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files[0])
        
        for i, raw_record in enumerate(raw_dataset.take(2)):  # Look at 2 records
            print(f"=== Raw Record {i+1} ===")
            
            # Parse the tfrecord
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            print("\n--- Episode Metadata ---")
            metadata_keys = [k for k in example.features.feature.keys() if k.startswith('episode_metadata/')]
            for key in sorted(metadata_keys):
                feature = example.features.feature[key]
                if feature.HasField('bytes_list'):
                    values = feature.bytes_list.value
                    for j, val in enumerate(values):
                        try:
                            decoded = val.decode('utf-8')
                            print(f"  {key}: '{decoded}'")
                        except:
                            print(f"  {key}: (binary data, length={len(val)})")
                elif feature.HasField('int64_list'):
                    values = list(feature.int64_list.value)
                    print(f"  {key}: {values}")
            
            print("\n--- Steps Structure ---")
            step_keys = [k for k in example.features.feature.keys() if k.startswith('steps/')]
            
            # Get the number of steps
            num_steps = None
            for key in step_keys:
                feature = example.features.feature[key]
                if feature.HasField('bytes_list'):
                    num_steps = len(feature.bytes_list.value)
                    break
                elif feature.HasField('float_list'):
                    # For action and other fields, we need to figure out steps vs feature dimension
                    if 'action' in key:
                        # Actions are 7D, so divide by 7
                        num_steps = len(feature.float_list.value) // 7
                        break
            
            print(f"Number of steps: {num_steps}")
            
            # Show first few steps in detail
            print("\n--- First 3 Steps Content ---")
            for step_idx in range(min(3, num_steps)):
                print(f"\nStep {step_idx}:")
                
                # Language instruction
                if 'steps/language_instruction' in example.features.feature:
                    feature = example.features.feature['steps/language_instruction']
                    if feature.HasField('bytes_list') and step_idx < len(feature.bytes_list.value):
                        try:
                            decoded = feature.bytes_list.value[step_idx].decode('utf-8')
                            print(f"  language_instruction: '{decoded}'")
                        except:
                            print(f"  language_instruction: (binary data)")
                
                # Language motions
                if 'steps/language_motions' in example.features.feature:
                    feature = example.features.feature['steps/language_motions']
                    if feature.HasField('bytes_list') and step_idx < len(feature.bytes_list.value):
                        try:
                            decoded = feature.bytes_list.value[step_idx].decode('utf-8')
                            print(f"  language_motions: '{decoded}'")
                        except:
                            print(f"  language_motions: (binary data)")
                
                # Language motions future
                if 'steps/language_motions_future' in example.features.feature:
                    feature = example.features.feature['steps/language_motions_future']
                    if feature.HasField('bytes_list') and step_idx < len(feature.bytes_list.value):
                        try:
                            decoded = feature.bytes_list.value[step_idx].decode('utf-8')
                            print(f"  language_motions_future: '{decoded}'")
                        except:
                            print(f"  language_motions_future: (binary data)")
                
                # Action
                if 'steps/action' in example.features.feature:
                    feature = example.features.feature['steps/action']
                    if feature.HasField('float_list'):
                        start_idx = step_idx * 7
                        end_idx = start_idx + 7
                        if end_idx <= len(feature.float_list.value):
                            action_vals = list(feature.float_list.value[start_idx:end_idx])
                            print(f"  action: {action_vals}")
                
                # State
                if 'steps/observation/state' in example.features.feature:
                    feature = example.features.feature['steps/observation/state']
                    if feature.HasField('float_list'):
                        start_idx = step_idx * 8  # Assuming 8D state
                        end_idx = start_idx + 8
                        if end_idx <= len(feature.float_list.value):
                            state_vals = list(feature.float_list.value[start_idx:end_idx])
                            print(f"  state: {state_vals}")
                
                # Joint state
                if 'steps/observation/joint_state' in example.features.feature:
                    feature = example.features.feature['steps/observation/joint_state']
                    if feature.HasField('float_list'):
                        start_idx = step_idx * 7  # Assuming 7D joint state
                        end_idx = start_idx + 7
                        if end_idx <= len(feature.float_list.value):
                            joint_vals = list(feature.float_list.value[start_idx:end_idx])
                            print(f"  joint_state: {joint_vals}")
                
                # Flags
                for flag_key in ['is_first', 'is_last', 'is_terminal']:
                    full_key = f'steps/{flag_key}'
                    if full_key in example.features.feature:
                        feature = example.features.feature[full_key]
                        if feature.HasField('int64_list') and step_idx < len(feature.int64_list.value):
                            val = feature.int64_list.value[step_idx]
                            print(f"  {flag_key}: {bool(val)}")
                
                # Reward and discount
                for float_key in ['reward', 'discount']:
                    full_key = f'steps/{float_key}'
                    if full_key in example.features.feature:
                        feature = example.features.feature[full_key]
                        if feature.HasField('float_list') and step_idx < len(feature.float_list.value):
                            val = feature.float_list.value[step_idx]
                            print(f"  {float_key}: {val}")
            
            print("\n" + "="*60 + "\n")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("Inspecting libero_lm_90 dataset structure...\n")
    inspect_libero_lm_90()

if __name__ == "__main__":
    main() 