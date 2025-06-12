#!/usr/bin/env python3
"""
Example usage script for the Libero No-Noops RLDS dataset with multi-config support.

This script demonstrates how to load and iterate through different LIBERO suites
using the config system.
"""

import os
import tensorflow as tf
import tensorflow_datasets as tfds
from libero_no_noops_dataset_dataset_builder import LiberoNoNoopsDataset

def main():
    # Set the environment variable to point to the parent directory containing all suite folders
    # Expected structure:
    # /path/to/libero_no_noops_enhanced/
    # ├── libero_10_no_noops/
    # ├── libero_90_no_noops/
    # ├── libero_spatial_no_noops/
    # ├── libero_goal_no_noops/
    # └── libero_object_no_noops/
    os.environ['LIBERO_DATA_DIR'] = '/home/gaofeng/arm_ws/openvla/experiments/robot/libero/libero_no_noops_enhanced'
    
    print("🚀 Multi-Config LIBERO Dataset Builder")
    print("="*60)
    
    # Available configs
    builder = LiberoNoNoopsDataset()
    configs = [config.name for config in builder.BUILDER_CONFIGS]
    print(f"Available configs: {configs}")
    print()
    
    # Example 1: Load a specific config
    config_name = "libero_10"  # Change this to test different configs
    print(f"📋 Loading config: {config_name}")
    
    try:
        # Method 1: Using tfds.load with config
        ds = tfds.load(
            'libero_no_noops_dataset',
            split='train',
            builder_kwargs={'config': config_name}
        )
        print(f"✅ Successfully loaded {config_name} using tfds.load")
        
        # Show dataset info
        builder_with_config = LiberoNoNoopsDataset(config=config_name)
        info = builder_with_config.info
        print(f"Description: {info.description}")
        print(f"Features: {list(info.features['steps'].keys())}")
        
    except Exception as e:
        print(f"❌ Failed to load {config_name}: {e}")
        print("\n💡 Try building the dataset first:")
        print(f"   tfds build libero_no_noops_dataset --config {config_name}")
        return
    
    # Example 2: Iterate through a few examples
    print(f"\n📊 Sample data from {config_name}:")
    print("-" * 40)
    
    for i, example in enumerate(ds.take(2)):
        print(f"\nExample {i + 1}:")
        
        # Episode metadata
        metadata = example['episode_metadata']
        suite = metadata['suite'].numpy().decode()
        demo_id = metadata['demo_id'].numpy()
        file_path = metadata['file_path'].numpy().decode()
        seg_labels = metadata['seg_labels'].numpy().decode()
        
        print(f"  Suite: {suite}")
        print(f"  Demo ID: {demo_id}")
        print(f"  File: {os.path.basename(file_path)}")
        print(f"  Seg labels: {seg_labels[:80]}{'...' if len(seg_labels) > 80 else ''}")
        
        steps = example['steps']
        print(f"  Number of steps: {len(steps)}")
        
        # Look at first step
        first_step = steps[0]
        print(f"  Task: {first_step['language_instruction'].numpy().decode()}")
        
        # Show motion sequence sample
        if len(steps) >= 3:
            print(f"  Sample motions:")
            for j in range(min(3, len(steps))):
                step = steps[j]
                future_motions = step['language_motions_future'].numpy().decode()
                current_motion = future_motions.split('|')[0] if future_motions else "end of sequence"
                print(f"    Step {j + 1}: {current_motion}")

def build_all_configs():
    """Example function to build all available configs."""
    print("\n🏗️ Building all configs:")
    print("="*60)
    
    # Get all available configs
    builder = LiberoNoNoopsDataset()
    configs = [config.name for config in builder.BUILDER_CONFIGS]
    
    for config_name in configs:
        print(f"\n📦 Building {config_name}...")
        try:
            # Check if data directory exists
            base_dir = os.environ.get('LIBERO_DATA_DIR')
            data_dir = os.path.join(base_dir, f"{config_name}_no_noops")
            
            if not os.path.exists(data_dir):
                print(f"⚠️  Data directory not found: {data_dir}")
                continue
                
            # Build the dataset
            builder_with_config = LiberoNoNoopsDataset(config=config_name)
            builder_with_config.download_and_prepare()
            print(f"✅ Successfully built {config_name}")
            
        except Exception as e:
            print(f"❌ Failed to build {config_name}: {e}")
            
def load_and_compare_configs():
    """Load and compare different LIBERO suite configs."""
    print("\n🔍 Comparing different LIBERO suites:")
    print("="*60)
    
    base_dir = os.environ.get('LIBERO_DATA_DIR')
    available_configs = []
    
    # Check which configs have data available
    builder = LiberoNoNoopsDataset()
    for config in builder.BUILDER_CONFIGS:
        data_dir = os.path.join(base_dir, f"{config.name}_no_noops")
        if os.path.exists(data_dir):
            available_configs.append(config.name)
    
    print(f"Available configs with data: {available_configs}")
    
    for config_name in available_configs[:3]:  # Compare first 3 available configs
        print(f"\n📋 {config_name.upper()}:")
        print("-" * 30)
        
        try:
            # Load dataset
            ds = tfds.load(
                'libero_no_noops_dataset',
                split='train',
                builder_kwargs={'config': config_name}
            )
            
            # Get basic stats
            total_episodes = 0
            total_steps = 0
            unique_tasks = set()
            
            for example in ds.take(10):  # Sample 10 episodes
                total_episodes += 1
                steps = example['steps']
                total_steps += len(steps)
                
                # Get task description
                first_step = steps[0]
                task = first_step['language_instruction'].numpy().decode()
                unique_tasks.add(task)
            
            print(f"  Sample episodes: {total_episodes}")
            print(f"  Total steps: {total_steps}")
            print(f"  Average steps per episode: {total_steps/total_episodes:.1f}")
            print(f"  Unique tasks sampled: {len(unique_tasks)}")
            
        except Exception as e:
            print(f"  ❌ Error loading {config_name}: {e}")

def show_usage_examples():
    """Show different ways to use the multi-config dataset."""
    print("\n💡 Usage Examples:")
    print("="*60)
    
    print("""
# Method 1: Using tfds.load with config parameter
ds = tfds.load(
    'libero_no_noops_dataset',
    split='train', 
    builder_kwargs={'config': 'libero_10'}
)

# Method 2: Using builder directly with config
builder = LiberoNoNoopsDataset(config='libero_90')
ds = builder.as_dataset(split='train')

# Method 3: Build specific config from command line
# tfds build libero_no_noops_dataset --config libero_spatial

# Method 4: Build all configs
for config in ['libero_10', 'libero_90', 'libero_spatial', 'libero_goal', 'libero_object']:
    tfds build libero_no_noops_dataset --config {config}
    
# Method 5: List available configs
builder = LiberoNoNoopsDataset()
configs = [config.name for config in builder.BUILDER_CONFIGS]
print(f"Available: {configs}")
    """)

def check_data_structure():
    """Check the expected data structure."""
    print("\n📁 Expected Data Structure:")
    print("="*60)
    
    base_dir = os.environ.get('LIBERO_DATA_DIR', '[LIBERO_DATA_DIR]')
    print(f"""
{base_dir}/
├── libero_10_no_noops/
│   ├── task1_demo.hdf5
│   ├── task2_demo.hdf5
│   └── ...
├── libero_90_no_noops/
│   ├── task1_demo.hdf5
│   └── ...
├── libero_spatial_no_noops/
│   ├── task1_demo.hdf5
│   └── ...
├── libero_goal_no_noops/
│   ├── task1_demo.hdf5
│   └── ...
└── libero_object_no_noops/
    ├── task1_demo.hdf5
    └── ...
    """)
    
    # Check actual structure
    if base_dir != '[LIBERO_DATA_DIR]':
        print(f"\n🔍 Checking actual structure at: {base_dir}")
        if os.path.exists(base_dir):
            subdirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('_no_noops')]
            print(f"Found directories: {subdirs}")
            
            for subdir in subdirs:
                full_path = os.path.join(base_dir, subdir)
                hdf5_files = [f for f in os.listdir(full_path) if f.endswith('_demo.hdf5')]
                print(f"  {subdir}: {len(hdf5_files)} HDF5 files")
        else:
            print(f"❌ Directory does not exist: {base_dir}")

if __name__ == "__main__":
    check_data_structure()
    show_usage_examples() 
    main()
    # Uncomment these to run additional examples:
    # load_and_compare_configs()
    # build_all_configs() 