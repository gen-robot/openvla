from typing import Iterator, Tuple, Any
import warnings

import cv2
import glob
import numpy as np
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds

import h5py


class LiberoNoNoopsConfig(tfds.core.BuilderConfig):
    """BuilderConfig for LiberoNoNoopsDataset."""

    def __init__(self, *, suite_name: str, description: str = "", **kwargs):
        """Constructs a LiberoNoNoopsConfig.
        
        Args:
            suite_name: Name of the LIBERO suite (e.g., 'libero_10', 'libero_90').
            description: Description of this config.
            **kwargs: keyword arguments forwarded to super.
        """
        super(LiberoNoNoopsConfig, self).__init__(
            name=suite_name,
            description=description,
            **kwargs,
        )
        self.suite_name = suite_name


class LiberoNoNoopsDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Libero no-noops dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release with multi-config support for different LIBERO suites.',
    }

    # Define configs for different LIBERO suites
    BUILDER_CONFIGS = [
        LiberoNoNoopsConfig(
            suite_name="libero_10",
            description="LIBERO-10: 10 diverse manipulation tasks across different scenes",
        ),
        LiberoNoNoopsConfig(
            suite_name="libero_90", 
            description="LIBERO-90: 90 manipulation tasks with procedural generation",
        ),
        LiberoNoNoopsConfig(
            suite_name="libero_spatial",
            description="LIBERO-Spatial: Tasks focusing on spatial reasoning and relationships",
        ),
        LiberoNoNoopsConfig(
            suite_name="libero_object",
            description="LIBERO-Object: Tasks focusing on object manipulation and interaction",
        ),
        LiberoNoNoopsConfig(
            suite_name="libero_goal",
            description="LIBERO-Goal: Tasks with complex goal specifications and constraints",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        config_description = ""
        if self.builder_config:
            config_description = f" - {self.builder_config.description}"
        
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, encoding_format='jpeg',
                            doc='Main camera RGB observation.'
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(224, 224, 3), dtype=np.uint8, encoding_format='jpeg',
                            doc='Wrist camera RGB observation.'
                        ),
                        'depth': tfds.features.Image(
                            shape=(224, 224, 1), dtype=np.uint8, encoding_format='png',
                            doc='Main camera depth observation.'
                        ),
                        'wrist_depth': tfds.features.Image(
                            shape=(224, 224, 1), dtype=np.uint8, encoding_format='png',
                            doc='Wrist camera depth observation.'
                        ),
                        'seg': tfds.features.Image(
                            shape=(224, 224, 1), dtype=np.uint8, encoding_format='png',
                            doc='Main camera segmentation observation.'
                        ),
                        'wrist_seg': tfds.features.Image(
                            shape=(224, 224, 1), dtype=np.uint8, encoding_format='png',
                            doc='Wrist camera segmentation observation.'
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,), dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).'
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,), dtype=np.float32,
                            doc='Robot joint angles.'
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc='Robot EEF action.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_motions': tfds.features.Text(
                        doc='Previous motions separated by \'|\'.'
                    ),
                    'language_motions_future': tfds.features.Text(
                        doc='Language Motions in Future steps, separated by \'|\''
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'demo_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Original demo ID number within a LIBERO \'episode.\'',
                    ),
                    'seg_labels': tfds.features.Text(
                        doc='Segmentation labels, separated by \'|\''
                    ),
                    'suite': tfds.features.Text(
                        doc=f'LIBERO suite name{config_description}'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # Get base data directory
        base_data_dir = os.environ.get('LIBERO_DATA_DIR', None)
        if base_data_dir is None:
            raise ValueError(
                "LIBERO_DATA_DIR environment variable is not set. "
                "Please set it to point to the parent directory containing your LIBERO suite folders."
            )
        
        # Determine data directory based on config
        if self.builder_config:
            suite_name = self.builder_config.suite_name
            data_dir = os.path.join(base_data_dir, f"{suite_name}_no_noops")
        else:
            raise ValueError(
                "No config specified. Please use one of the available configs: "
                f"{[config.name for config in self.BUILDER_CONFIGS]}"
            )
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Expected structure: {base_data_dir}/{suite_name}_no_noops/\n"
                f"Available configs: {[config.name for config in self.BUILDER_CONFIGS]}"
            )
        
        print(f"Using config: {suite_name}")
        print(f"Data directory: {data_dir}")
        
        return {
            'train': self._generate_examples(
                path=os.path.join(data_dir, '*_demo.hdf5'),
                suite_name=suite_name
            ),
        }

    def _normalize_depth(self, depth_img):
        """Normalize depth image to 0-255 range."""
        if depth_img.dtype != np.uint8:
            # Normalize depth to 0-255 range
            depth_min = np.min(depth_img)
            depth_max = np.max(depth_img)
            if depth_max > depth_min:
                depth_img = ((depth_img - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_img = np.zeros_like(depth_img, dtype=np.uint8)
        return depth_img

    def _validate_enhanced_features(self, hdf5_path, episode_data, episode_key):
        """Validate that the HDF5 file contains required enhanced features."""
        missing_features = []
        
        # Check for motion descriptions
        if 'motion_descriptions' not in episode_data:
            missing_features.append('motion_descriptions')
        
        # Check for segmentation labels  
        if 'seg_labels' not in episode_data:
            missing_features.append('seg_labels')
        
        if missing_features:
            suite_name = self.builder_config.suite_name if self.builder_config else "unknown"
            error_msg = f"""
❌ MISSING ENHANCED FEATURES in {os.path.basename(hdf5_path)} - {episode_key}
Missing: {missing_features}
Suite: {suite_name}

This dataset builder requires enhanced HDF5 files with real motion descriptions and segmentation labels.

🔧 SOLUTION: Regenerate your dataset using the enhanced regeneration script:

python experiments/robot/libero/regenerate_libero_dataset_enhanced.py \\
    --libero_task_suite {suite_name} \\
    --libero_raw_data_dir [raw_data_path] \\
    --libero_target_dir [target_path]

📋 INSPECT YOUR DATA: Use the inspection script to check your files:
python rlds_dataset_builder/libero_no_noops_dataset/inspect_enhanced_hdf5.py {os.path.dirname(hdf5_path)}

Enhanced features provide:
✓ Real motion descriptions: "move forward, close gripper", "rotate counterclockwise"
✓ Real segmentation labels: "none|yellow_cube|blue_bowl|wooden_table|MountedPanda0|..."
"""
            raise AssertionError(error_msg)

    def _generate_examples(self, path, suite_name) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(hdf5_path):
            """Parse a single HDF5 file containing multiple episodes."""
            
            # Extract task description from filename
            # e.g., "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5"
            filename = os.path.basename(hdf5_path)
            task_description = filename.replace('_demo.hdf5', '').replace('_', ' ')
            # Clean up the task description to make it more readable
            parts = task_description.split(' ')
            # Remove scene identifiers like "KITCHEN SCENE6"
            if len(parts) > 2 and 'SCENE' in parts[1]:
                task_description = ' '.join(parts[2:])
            else:
                task_description = ' '.join(parts)
            
            # Load HDF5 file
            with h5py.File(hdf5_path, 'r') as f:
                data_group = f['data']
                episode_keys = [key for key in data_group.keys() if key.startswith('demo_')]
                
                for episode_key in episode_keys:
                    episode_data = data_group[episode_key]
                    
                    # Validate that enhanced features are present - this will raise AssertionError if missing
                    self._validate_enhanced_features(hdf5_path, episode_data, episode_key)
                    
                    # Extract data arrays
                    actions = episode_data['actions'][:]
                    rewards = episode_data['rewards'][:]
                    dones = episode_data['dones'][:]
                    
                    # Extract observations
                    obs_group = episode_data['obs']
                    agentview_images = obs_group['agentview_rgb'][:]
                    eye_in_hand_images = obs_group['eye_in_hand_rgb'][:]
                    agentview_depths = obs_group['agentview_depth'][:]
                    eye_in_hand_depths = obs_group['eye_in_hand_depth'][:]
                    agentview_segmentations = obs_group['agentview_segmentation'][:]
                    eye_in_hand_segmentations = obs_group['eye_in_hand_segmentation'][:]
                    joint_states = obs_group['joint_states'][:]
                    ee_states = obs_group['ee_states'][:]
                    gripper_states = obs_group['gripper_states'][:]
                    
                    # Extract enhanced motion descriptions (guaranteed to be present after validation)
                    motion_desc_bytes = episode_data['motion_descriptions'][:]
                    motion_descriptions = [desc.decode('utf-8') if isinstance(desc, bytes) else str(desc) 
                                         for desc in motion_desc_bytes]
                    print(f"  ✅ Loaded enhanced motion descriptions: {len(motion_descriptions)} steps")
                    
                    # Extract enhanced segmentation labels (guaranteed to be present after validation)
                    seg_labels_data = episode_data['seg_labels'][()]
                    if isinstance(seg_labels_data, bytes):
                        seg_labels = seg_labels_data.decode('utf-8')
                    else:
                        seg_labels = str(seg_labels_data)
                    print(f"  ✅ Loaded enhanced seg labels: {seg_labels}")
                    
                    # Build episode steps
                    episode_steps = []
                    num_steps = len(actions)
                    
                    for step_idx in range(num_steps):
                        # Resize images to 224x224 if needed
                        agentview_img = agentview_images[step_idx]
                        eye_in_hand_img = eye_in_hand_images[step_idx]
                        agentview_depth = agentview_depths[step_idx]
                        eye_in_hand_depth = eye_in_hand_depths[step_idx]
                        agentview_seg = agentview_segmentations[step_idx]
                        eye_in_hand_seg = eye_in_hand_segmentations[step_idx]
                        
                        if agentview_img.shape[:2] != (224, 224):
                            agentview_img = cv2.resize(agentview_img, (224, 224))
                        if eye_in_hand_img.shape[:2] != (224, 224):
                            eye_in_hand_img = cv2.resize(eye_in_hand_img, (224, 224))
                        if agentview_depth.shape[:2] != (224, 224):
                            agentview_depth = cv2.resize(agentview_depth, (224, 224))
                        if eye_in_hand_depth.shape[:2] != (224, 224):
                            eye_in_hand_depth = cv2.resize(eye_in_hand_depth, (224, 224))
                        if agentview_seg.shape[:2] != (224, 224):
                            agentview_seg = cv2.resize(agentview_seg, (224, 224))
                        if eye_in_hand_seg.shape[:2] != (224, 224):
                            eye_in_hand_seg = cv2.resize(eye_in_hand_seg, (224, 224))
                        
                        # Note: Images from LIBERO are upside down, so we rotate them 180 degrees
                        agentview_img = cv2.rotate(agentview_img, cv2.ROTATE_180)
                        eye_in_hand_img = cv2.rotate(eye_in_hand_img, cv2.ROTATE_180)
                        agentview_depth = cv2.rotate(agentview_depth, cv2.ROTATE_180)
                        eye_in_hand_depth = cv2.rotate(eye_in_hand_depth, cv2.ROTATE_180)
                        agentview_seg = cv2.rotate(agentview_seg, cv2.ROTATE_180)
                        eye_in_hand_seg = cv2.rotate(eye_in_hand_seg, cv2.ROTATE_180)
                        
                        # Process depth images
                        agentview_depth = self._normalize_depth(agentview_depth)
                        eye_in_hand_depth = self._normalize_depth(eye_in_hand_depth)
                        
                        # Ensure single channel for depth and segmentation
                        if len(agentview_depth.shape) == 2:
                            agentview_depth = np.expand_dims(agentview_depth, axis=2)
                        if len(eye_in_hand_depth.shape) == 2:
                            eye_in_hand_depth = np.expand_dims(eye_in_hand_depth, axis=2)
                        if len(agentview_seg.shape) == 3 and agentview_seg.shape[2] == 1:
                            agentview_seg = agentview_seg[:, :, 0]
                        if len(agentview_seg.shape) == 2:
                            agentview_seg = np.expand_dims(agentview_seg, axis=2)
                        if len(eye_in_hand_seg.shape) == 3 and eye_in_hand_seg.shape[2] == 1:
                            eye_in_hand_seg = eye_in_hand_seg[:, :, 0]
                        if len(eye_in_hand_seg.shape) == 2:
                            eye_in_hand_seg = np.expand_dims(eye_in_hand_seg, axis=2)
                        
                        # Construct state vector (EEF pose + gripper state)
                        # ee_states contains [pos(3) + axis_angle(3)] = 6D pose
                        # gripper_states contains 2D gripper position
                        ee_pose = ee_states[step_idx]  # 6D: position + axis angle
                        gripper_pos = gripper_states[step_idx]  # 2D: gripper positions
                        state = np.concatenate([ee_pose, gripper_pos]).astype(np.float32)
                        
                        # Generate language motions (previous motions) - improved windowing
                        if step_idx == 0:
                            language_motions = ""
                        else:
                            prev_motions = motion_descriptions[:step_idx]
                            # Use last 3-5 motions for better context
                            window_size = min(5, max(3, len(prev_motions)))
                            language_motions = "|".join(prev_motions[-window_size:])
                        
                        # Generate future language motions - improved windowing
                        if step_idx == num_steps - 1:
                            language_motions_future = ""
                        else:
                            future_motions = motion_descriptions[step_idx+1:]
                            # Use next 3-5 motions for better context
                            window_size = min(5, max(3, len(future_motions)))
                            language_motions_future = "|".join(future_motions[:window_size])
                        
                        step_data = {
                            'observation': {
                                'image': agentview_img,
                                'wrist_image': eye_in_hand_img,
                                'depth': agentview_depth.astype(np.uint8),
                                'wrist_depth': eye_in_hand_depth.astype(np.uint8),
                                'seg': agentview_seg.astype(np.uint8),
                                'wrist_seg': eye_in_hand_seg.astype(np.uint8),
                                'state': state,
                                'joint_state': joint_states[step_idx].astype(np.float32),
                            },
                            'action': actions[step_idx].astype(np.float32),
                            'language_instruction': task_description,
                            'language_motions': language_motions,
                            'language_motions_future': language_motions_future,
                            'reward': float(rewards[step_idx]),
                            'discount': 1.0,  # Default discount
                            'is_first': step_idx == 0,
                            'is_last': step_idx == num_steps - 1,
                            'is_terminal': bool(dones[step_idx]),  # True on terminal steps
                        }
                        episode_steps.append(step_data)
                    
                    # Extract demo ID from episode key (e.g., "demo_0" -> 0)
                    demo_id = int(episode_key.split('_')[1])
                    
                    # Create sample for this episode
                    sample = {
                        'steps': episode_steps,
                        'episode_metadata': {
                            'file_path': hdf5_path,
                            'demo_id': demo_id,
                            'seg_labels': seg_labels,
                            'suite': suite_name,
                        }
                    }
                    
                    yield f"{suite_name}_{os.path.basename(hdf5_path)}_{episode_key}", sample

        # Get all HDF5 files matching the pattern
        hdf5_files = glob.glob(path)
        print(f"Found {len(hdf5_files)} HDF5 files to process for {suite_name}")
        
        if not hdf5_files:
            error_msg = f"""
❌ NO HDF5 FILES FOUND at pattern: {path}

Make sure:
1. LIBERO_DATA_DIR environment variable points to the parent directory: {os.environ.get('LIBERO_DATA_DIR', 'NOT SET')}
2. The directory contains {suite_name}_no_noops/ subdirectory  
3. The subdirectory contains *_demo.hdf5 files
4. Files were generated using regenerate_libero_dataset_enhanced.py

Expected structure:
{os.environ.get('LIBERO_DATA_DIR', '[LIBERO_DATA_DIR]')}/
├── {suite_name}_no_noops/
│   ├── task1_demo.hdf5
│   ├── task2_demo.hdf5
│   └── ...
"""
            raise FileNotFoundError(error_msg)
        
        # Issue warning about dataset builder requirements
        warnings.warn(f"""
🔥 ENHANCED FEATURES REQUIRED for {suite_name}
This dataset builder requires HDF5 files with enhanced features:
- Real motion descriptions from LIBERO environment  
- Real segmentation labels with object instance names

Processing {len(hdf5_files)} files for {suite_name}. If any file lacks enhanced features, the build will fail.
Use inspect_enhanced_hdf5.py to verify your data quality first.
""", UserWarning)
        
        # Process each HDF5 file
        for hdf5_file in hdf5_files:
            print(f"Processing {hdf5_file}")
            yield from _parse_example(hdf5_file) 