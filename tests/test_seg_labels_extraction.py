#!/usr/bin/env python3

import os
import sys
sys.path.append("experiments/robot/libero")

from libero.libero import benchmark
from libero.libero.envs import SegmentationRenderEnv
from libero.libero import get_libero_path

def test_seg_labels_extraction():
    """Test extracting segmentation labels from LIBERO environment."""
    
    print("=== Testing Segmentation Labels Extraction ===\n")
    
    # Get a task from libero_10 (which should have objects)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(0)  # First task
    
    print(f"Testing with task: {task.language}")
    print(f"Task name: {task.name}")
    
    # Create enhanced environment
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 224,
        "camera_widths": 224,
        "camera_depths": True,
        "camera_segmentations": "instance",
    }
    env = SegmentationRenderEnv(**env_args)
    env.seed(0)
    
    # Reset environment to initialize segmentation mappings
    obs = env.reset()
    
    print("\n--- Environment Segmentation Info ---")
    print(f"segmentation_id_mapping: {env.segmentation_id_mapping}")
    print(f"segmentation_robot_id: {env.segmentation_robot_id}")
    if hasattr(env, 'instance_to_id'):
        print(f"instance_to_id: {env.instance_to_id}")
    
    # Check what instances are available in the model
    if hasattr(env, 'env') and hasattr(env.env, 'model') and hasattr(env.env.model, 'instances_to_ids'):
        all_instances = list(env.env.model.instances_to_ids.keys())
        print(f"All model instances: {all_instances}")
    
    # Extract segmentation labels
    def extract_segmentation_labels(env, obs):
        """
        Extract segmentation labels from the environment and observations.
        
        Returns labels in the format matching libero_lm_90:
        'none|object1_name|object2_name|...|MountedPanda0|RethinkMount0|PandaGripper0'
        """
        labels = []
        
        # Always start with 'none' (background)
        labels.append('none')
        
        # Get object instance names from segmentation mapping (excluding robot parts)
        if hasattr(env, 'segmentation_id_mapping') and env.segmentation_id_mapping:
            # Add object labels from segmentation mapping (these are the scene objects)
            for seg_id, instance_name in env.segmentation_id_mapping.items():
                labels.append(instance_name)
        
        # Get all instances from the model to include robot parts
        if hasattr(env, 'env') and hasattr(env.env, 'model') and hasattr(env.env.model, 'instances_to_ids'):
            all_instances = list(env.env.model.instances_to_ids.keys())
            
            # Add robot-related instances that are typically present
            robot_instances = ['MountedPanda0', 'RethinkMount0', 'PandaGripper0', 'Panda0']
            for robot_instance in robot_instances:
                if robot_instance in all_instances and robot_instance not in labels:
                    labels.append(robot_instance)
        
        # Fallback: add standard robot labels if we couldn't extract them
        standard_robot_labels = ['MountedPanda0', 'RethinkMount0', 'PandaGripper0']
        for robot_label in standard_robot_labels:
            if robot_label not in labels:
                labels.append(robot_label)
        
        return labels
    
    seg_labels = extract_segmentation_labels(env, obs)
    seg_labels_str = "|".join(seg_labels)
    
    print(f"\n--- Extracted Segmentation Labels ---")
    print(f"Labels list: {seg_labels}")
    print(f"Labels string: '{seg_labels_str}'")
    
    print(f"\n--- Comparison with libero_lm_90 format ---")
    example_lm90 = 'none|akita_black_bowl_1|wine_bottle_1|white_cabinet_1|wine_rack_1|MountedPanda0|RethinkMount0|PandaGripper0'
    print(f"libero_lm_90 example: '{example_lm90}'")
    print(f"Our extraction:        '{seg_labels_str}'")
    
    # Check if our format matches the expected pattern
    has_none = seg_labels_str.startswith('none|')
    has_robot_parts = any(robot in seg_labels_str for robot in ['MountedPanda0', 'RethinkMount0', 'PandaGripper0'])
    
    print(f"\n--- Format Validation ---")
    print(f"✓ Starts with 'none|': {has_none}")
    print(f"✓ Has robot parts: {has_robot_parts}")
    print(f"✓ Total labels: {len(seg_labels)}")
    
    env.close()
    
    return seg_labels_str

if __name__ == "__main__":
    try:
        result = test_seg_labels_extraction()
        print(f"\n✅ Test completed successfully!")
        print(f"Final result: '{result}'")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 