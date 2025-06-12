"""
Enhanced version of regenerate_libero_dataset.py that also captures segmentation labels and motion descriptions.

Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments,
and additionally captures:
- Segmentation instance mappings for proper seg_labels
- Motion descriptions for language_motions and language_motions_future

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset_enhanced.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR> \
        [--num-jobs <NUM_JOBS>] [--job-id <JOB_ID>]

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset_enhanced.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

    Example (Multiprocess):
        python experiments/robot/libero/regenerate_libero_dataset_enhanced.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops \
            --num-jobs 4 --job-id 0
"""

import argparse
import json
import os
import time

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from libero.libero.envs import SegmentationRenderEnv

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
)


IMAGE_RESOLUTION = 224


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def get_enhanced_libero_env(task, model_family, resolution=256, depth=True, segmentation="instance"):
    """
    Initializes and returns the enhanced LIBERO environment with segmentation capabilities.
    """
    from libero.libero import get_libero_path
    
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": depth,
        "camera_segmentations": segmentation,
    }
    env = SegmentationRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def describe_motion_from_action(action, prev_pos=None, prev_gripper=None, threshold=0.01):
    """
    Generate motion description from robot action.
    
    LIBERO Action Space:
    - [0:3]: XYZ position delta in meters (typical range ±0.01-0.1m per step)
    - [3:6]: Orientation delta as axis-angle in radians (typical range ±0.1-0.5 rad per step)  
    - [6]: Gripper action in [-1,+1] where -1=close, +1=open, 0=maintain
    
    All actions are DELTA actions (changes), not absolute positions.
    Uses appropriate thresholds: 5mm for position, ~3° for orientation, 0.1 for gripper.
    Allows multiple simultaneous movements across different axes.
    """
    # Mapping similar to primitive_movements.py but adapted for LIBERO action space
    names = [
        {-1: "backward", 0: None, 1: "forward"},      # X axis
        {-1: "right", 0: None, 1: "left"},            # Y axis  
        {-1: "down", 0: None, 1: "up"},               # Z axis
        {-1: "tilt down", 0: None, 1: "tilt up"},     # Pitch (around Y axis)
        {},  # Roll (around X axis) - disabled
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},  # Yaw (around Z axis)
        {-1: "close gripper", 0: None, 1: "open gripper"},  # Gripper
    ]
    
    # Store raw action for fallback logic
    raw_action = action.copy()
    
    # Discretize action into -1, 0, 1 based on threshold
    move_vec = np.zeros(7, dtype=int)
    
    # Position components (0-2) - allow multiple simultaneous movements
    pos_threshold = 0.005  # 5mm movement threshold for position (meters)
    for i in range(3):
        if action[i] > pos_threshold:
            move_vec[i] = 1
        elif action[i] < -pos_threshold:
            move_vec[i] = -1
    
    # Orientation components (3-5) - allow multiple simultaneous rotations
    ori_change = action[3:6]
    ori_threshold = 0.05  # ~3 degree threshold for orientation (radians)
    for i, ori_val in enumerate(ori_change):
        if abs(ori_val) > ori_threshold:
            move_vec[3 + i] = 1 if ori_val > 0 else -1
    
    # Gripper component (6)
    gripper_threshold = 0.1  # Threshold for gripper state change
    if prev_gripper is not None:
        gripper_change = action[6] - prev_gripper
        if abs(gripper_change) > gripper_threshold:
            move_vec[6] = 1 if gripper_change > 0 else -1
    else:
        # First step, use absolute gripper value
        if action[6] > gripper_threshold:
            move_vec[6] = 1
        elif action[6] < -gripper_threshold:
            move_vec[6] = -1
    
    # Generate description using the structured approach
    description = ""
    
    # Handle XYZ movements first - collect all simultaneous movements
    xyz_moves = []
    for i in range(3):
        if move_vec[i] != 0 and i < len(names) and move_vec[i] in names[i] and names[i][move_vec[i]] is not None:
            xyz_moves.append(names[i][move_vec[i]])
    
    if xyz_moves:
        description = "move " + " and ".join(xyz_moves)
    
    # Handle orientation movements - allow multiple simultaneous rotations
    for i in range(3, 6):
        if move_vec[i] != 0 and i < len(names) and names[i] and move_vec[i] in names[i]:
            ori_desc = names[i][move_vec[i]]
            if ori_desc is not None:
                if description:
                    description += " and "
                description += ori_desc
    
    # Handle gripper movement
    if move_vec[6] != 0 and 6 < len(names) and move_vec[6] in names[6]:
        gripper_desc = names[6][move_vec[6]]
        if gripper_desc is not None:
            if description:
                description += ", "
            description += gripper_desc
    
    # Fallback logic when no significant movement is detected
    if not description:
        # Find the most significant movement even if below threshold
        # Use relative thresholds for comparison
        normalized_action = np.abs(raw_action.copy())
        # Normalize each component by its threshold for fair comparison
        normalized_action[0:3] /= pos_threshold      # Position components
        normalized_action[3:6] /= ori_threshold      # Orientation components  
        normalized_action[6] /= gripper_threshold    # Gripper component
        
        max_idx = np.argmax(normalized_action)
        if normalized_action[max_idx] > 0.1:  # At least 10% of threshold
            direction = 1 if raw_action[max_idx] > 0 else -1
            
            if max_idx < len(names) and names[max_idx] and direction in names[max_idx] and names[max_idx][direction] is not None:
                action_name = names[max_idx][direction]
                if max_idx < 3:
                    description = "move " + action_name
                else:
                    description = action_name
            else:
                description = "hold position"
        else:
            description = "hold position"
    
    return description


def generate_motion_descriptions(actions, ee_states, gripper_states):
    """
    Generate motion descriptions for all actions in a trajectory.
    Improved to better track gripper state changes across the trajectory.
    """
    descriptions = []
    
    for i, action in enumerate(actions):
        # Get previous gripper state (average of gripper joint positions)
        prev_gripper = gripper_states[i-1].mean() if i > 0 else None
        
        # For more robust gripper tracking, also consider the actual gripper action from previous timestep
        prev_gripper_action = actions[i-1][6] if i > 0 else None
        
        # Use the previous gripper action if available, otherwise fall back to state-based tracking
        effective_prev_gripper = prev_gripper_action if prev_gripper_action is not None else prev_gripper
        
        description = describe_motion_from_action(
            action, 
            prev_pos=ee_states[i-1][:3] if i > 0 else None, 
            prev_gripper=effective_prev_gripper
        )
        descriptions.append(description)
    
    return descriptions


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


def create_annotated_video(rgb_frames, depth_frames, seg_frames, motion_descriptions, seg_labels, 
                          save_path, task_description, actions=None, fps=10):
    """
    Create an annotated video showing RGB, depth, segmentation and motion descriptions.
    
    Args:
        rgb_frames: List of RGB images (agentview)
        depth_frames: List of depth images 
        seg_frames: List of segmentation images
        motion_descriptions: List of motion description strings
        seg_labels: Segmentation labels string (pipe-separated)
        save_path: Path to save the video
        task_description: Task description for title
        actions: Optional list of action arrays to display
        fps: Frames per second for the video
    """
    import cv2
    
    if not rgb_frames:
        return
        
    # Parse segmentation labels for legend
    labels_list = seg_labels.split('|') if isinstance(seg_labels, str) else ['none']
    
    # Rotate images by 180 degrees (as done in libero_utils.py)
    rgb_frames = [frame[::-1, ::-1] for frame in rgb_frames]
    depth_frames = [frame[::-1, ::-1] for frame in depth_frames]
    seg_frames = [frame[::-1, ::-1] for frame in seg_frames]
    
    # Get frame dimensions
    h, w = rgb_frames[0].shape[:2]
    
    # Calculate required height for text (more space for action values)
    text_height = 240  # Increased height for better spacing
    video_width = w * 3
    video_height = h + text_height
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (video_width, video_height))
    
    try:
        for i, (rgb, depth, seg, motion_desc) in enumerate(zip(rgb_frames, depth_frames, seg_frames, motion_descriptions)):
            # Convert RGB to BGR for OpenCV
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Process depth image
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = np.squeeze(depth, axis=-1)
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
            depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            
            # Process segmentation image
            if seg.ndim == 3 and seg.shape[-1] == 1:
                seg = np.squeeze(seg, axis=-1)
            
            # Create colored segmentation with colormap
            if seg.max() > 0:
                seg_colored = (plt.cm.viridis(seg / seg.max()) * 255).astype(np.uint8)[:, :, :3]
                seg_bgr = cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR)
            else:
                seg_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Resize all images to same size
            rgb_bgr = cv2.resize(rgb_bgr, (w, h))
            depth_bgr = cv2.resize(depth_bgr, (w, h))
            seg_bgr = cv2.resize(seg_bgr, (w, h))
            
            # Create combined frame
            combined_frame = np.hstack([rgb_bgr, depth_bgr, seg_bgr])
            
            # Add padding for text
            padded_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            padded_frame[:h, :, :] = combined_frame
            
            # Text settings - smaller font to fit more text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35  # Smaller font for better fit and clarity
            thickness = 1
            line_height = 14   # Reduced line height for smaller font
            
            # Colors for different text elements
            white = (255, 255, 255)
            yellow = (0, 255, 255)
            green = (0, 255, 0)
            cyan = (255, 255, 0)
            
            # Add panel labels
            cv2.putText(padded_frame, "RGB", (w//2 - 20, h + 15), font, font_scale, white, thickness)
            cv2.putText(padded_frame, "Depth", (w + w//2 - 25, h + 15), font, font_scale, white, thickness)
            cv2.putText(padded_frame, "Segmentation", (2*w + w//2 - 50, h + 15), font, font_scale, white, thickness)
            
            # Current line position for text
            y_pos = h + 35
            
            # Add frame number 
            frame_text = f"Frame {i+1}/{len(rgb_frames)}"
            cv2.putText(padded_frame, frame_text, (10, y_pos), font, font_scale, white, thickness)
            y_pos += line_height + 2  # Extra spacing
            
            # Add motion description (wrap text properly)
            motion_text = f"Motion: {motion_desc}"
            max_chars_per_line = 110  # Increased for smaller font
            
            if len(motion_text) > max_chars_per_line:
                # Split long text into multiple lines at word boundaries
                words = motion_text.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= max_chars_per_line:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                for line in lines[:2]:  # Max 2 lines for motion to save space
                    cv2.putText(padded_frame, line, (10, y_pos), font, font_scale, cyan, thickness)
                    y_pos += line_height
            else:
                cv2.putText(padded_frame, motion_text, (10, y_pos), font, font_scale, cyan, thickness)
                y_pos += line_height
            
            y_pos += 3  # Extra spacing before actions
            
            # Add action values if available (fix overlapping)
            if actions is not None and i < len(actions):
                action = actions[i]
                # Format action values nicely with proper spacing
                pos_str = f"Pos: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]"
                ori_str = f"Ori: [{action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}]" 
                grip_str = f"Gripper: {action[6]:.3f}"
                
                # Put each action component on its own line to avoid overlap
                cv2.putText(padded_frame, pos_str, (10, y_pos), font, font_scale, green, thickness)
                y_pos += line_height
                cv2.putText(padded_frame, ori_str, (10, y_pos), font, font_scale, green, thickness)
                y_pos += line_height
                cv2.putText(padded_frame, grip_str, (10, y_pos), font, font_scale, green, thickness)
                y_pos += line_height
            
            y_pos += 3  # Extra spacing before task
            
            # Add task description (wrap if needed)
            task_text = f"Task: {task_description}"
            if len(task_text) > max_chars_per_line:
                task_text = task_text[:max_chars_per_line-3] + "..."
            cv2.putText(padded_frame, task_text, (10, y_pos), font, font_scale, yellow, thickness)
            y_pos += line_height + 2
            
            # Add segmentation legend (wrap across multiple lines instead of truncating)
            legend_prefix = "Objects: "
            all_labels_text = ", ".join(labels_list)
            full_legend_text = legend_prefix + all_labels_text

            max_chars_per_line = 110  # Same as motion descriptions

            if len(full_legend_text) > max_chars_per_line:
                # Split into multiple lines at word boundaries (treating each label as a "word")
                lines = []
                current_line = legend_prefix
                current_length = len(legend_prefix)
                
                for label in labels_list:
                    addition = ", " + label if not current_line.endswith(legend_prefix) else label
                    
                    if current_length + len(addition) <= max_chars_per_line:
                        current_line += addition
                        current_length += len(addition)
                    else:
                        if current_line != legend_prefix:  # Don't add empty lines
                            lines.append(current_line)
                        current_line = "         " + label  # Indent continuation lines
                        current_length = len(current_line)
                
                if current_line and current_line.strip():  # Add the last line if not empty
                    lines.append(current_line)
                
                # Display multiple lines
                for line in lines[:3]:  # Max 3 lines for objects to save space
                    cv2.putText(padded_frame, line, (10, y_pos), font, font_scale, green, thickness)
                    y_pos += line_height
            else:
                cv2.putText(padded_frame, full_legend_text, (10, y_pos), font, font_scale, green, thickness)
            
            # Write frame to video
            out.write(padded_frame)
            
    finally:
        out.release()
    
    print(f"Saved annotated video: {save_path}")


def save_visualizations(obs, save_dir, prefix=""):
    """
    Saves RGB, depth, and segmentation images from an observation dictionary.
    """
    os.makedirs(save_dir, exist_ok=True)

    cameras = ["agentview", "robot0_eye_in_hand"]

    for camera in cameras:
        # Save RGB
        rgb_key = f"{camera}_image"
        if rgb_key in obs and obs[rgb_key] is not None:
            imageio.imwrite(os.path.join(save_dir, f"{prefix}_{camera}_rgb.png"), obs[rgb_key])

        # Save Depth
        depth_key = f"{camera}_depth"
        if depth_key in obs and obs[depth_key] is not None:
            depth_img = obs[depth_key]
            if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                depth_img = np.squeeze(depth_img, axis=-1)

            min_val, max_val = np.min(depth_img), np.max(depth_img)
            if max_val > min_val:
                depth_img = (depth_img - min_val) / (max_val - min_val) * 255.0
            depth_img = depth_img.astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, f"{prefix}_{camera}_depth.png"), depth_img)

        # Save Segmentation
        seg_key = f"{camera}_segmentation_instance"
        if seg_key in obs and obs[seg_key] is not None:
            seg_img = obs[seg_key]
            if seg_img.ndim == 3 and seg_img.shape[-1] == 1:
                seg_img = np.squeeze(seg_img, axis=-1)

            max_val = np.max(seg_img)
            if max_val == 0:
                colored_seg = np.zeros(seg_img.shape + (3,), dtype=np.uint8)
            else:
                colored_seg = (plt.cm.viridis(seg_img / max_val) * 255).astype(np.uint8)

            imageio.imwrite(os.path.join(save_dir, f"{prefix}_{camera}_segmentation.png"), colored_seg[:, :, :3])


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset with enhanced features!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        if args.job_id == 0:
            print(f"Warning: Target directory {args.libero_target_dir} already exists. Files may be overwritten.")
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    if args.num_jobs > 1:
        metainfo_json_out_path = f"./{args.libero_task_suite}_metainfo_enhanced.job_{args.job_id}.json"
    else:
        metainfo_json_out_path = f"./{args.libero_task_suite}_metainfo_enhanced.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Determine task range for this job
    if args.num_jobs > 1:
        tasks_per_job = int(np.ceil(num_tasks_in_suite / args.num_jobs))
        start_task_id = args.job_id * tasks_per_job
        end_task_id = min((args.job_id + 1) * tasks_per_job, num_tasks_in_suite)
        task_ids_to_process = range(start_task_id, end_task_id)
        if not task_ids_to_process:
            print(f"Job {args.job_id} has no tasks to process. Exiting.")
            return
        print(f"Job {args.job_id} of {args.num_jobs}: processing tasks from {start_task_id} to {end_task_id - 1}...")
    else:
        task_ids_to_process = range(num_tasks_in_suite)

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in tqdm.tqdm(task_ids_to_process):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_enhanced_libero_env(
            task, "llava", resolution=IMAGE_RESOLUTION, depth=True, segmentation="instance")

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for i in range(len(orig_data.keys())):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Extract segmentation labels from first observation
            seg_labels = extract_segmentation_labels(env, obs)
            seg_labels_str = "|".join(seg_labels)

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []
            agentview_depths = []
            eye_in_hand_depths = []
            agentview_segmentations = []
            eye_in_hand_segmentations = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    robot_states.append(demo_data["robot_states"][0])
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )
                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                agentview_depths.append(obs["agentview_depth"])
                eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])
                agentview_segmentations.append(obs["agentview_segmentation_instance"])
                eye_in_hand_segmentations.append(obs["robot0_eye_in_hand_segmentation_instance"])

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            # At end of episode, generate motion descriptions and save trajectories to new HDF5 files (only keep successes)
            if done:
                # Generate motion descriptions
                motion_descriptions = generate_motion_descriptions(
                    np.array(actions), 
                    np.array(ee_states), 
                    np.array(gripper_states)
                )
                
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                obs_grp.create_dataset("agentview_depth", data=np.stack(agentview_depths, axis=0))
                obs_grp.create_dataset("eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0))
                obs_grp.create_dataset("agentview_segmentation", data=np.stack(agentview_segmentations, axis=0))
                obs_grp.create_dataset("eye_in_hand_segmentation", data=np.stack(eye_in_hand_segmentations, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)
                
                # Add enhanced features
                motion_descriptions_bytes = [desc.encode('utf-8') for desc in motion_descriptions]
                ep_data_grp.create_dataset("motion_descriptions", data=motion_descriptions_bytes)
                ep_data_grp.create_dataset("seg_labels", data=seg_labels_str.encode('utf-8'))

                # Create sample verification videos for first few successful episodes per task
                if args.create_videos and num_success <= args.max_videos_per_task:  # Save videos for first max_videos_per_task successful episodes per task
                    video_dir = os.path.join(args.libero_target_dir, "verification_videos")
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # Clean task name for filename
                    clean_task_name = task_description.replace(" ", "_").replace("/", "_")[:50]
                    video_path = os.path.join(video_dir, f"{clean_task_name}_demo_{i}_success_{num_success}.mp4")
                    
                    try:
                        create_annotated_video(
                            rgb_frames=agentview_images,
                            depth_frames=agentview_depths, 
                            seg_frames=agentview_segmentations,
                            motion_descriptions=motion_descriptions,
                            seg_labels=seg_labels_str,
                            save_path=video_path,
                            task_description=task_description,
                            actions=np.array(actions),
                            fps=8  # Slower playback for better inspection
                        )
                    except Exception as e:
                        print(f"Warning: Could not create verification video: {e}")

                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()
            metainfo_json_dict[task_key][episode_key]["seg_labels"] = seg_labels_str
            metainfo_json_dict[task_key][episode_key]["num_motion_descriptions"] = len(motion_descriptions) if done else 0

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved enhanced regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Enhanced dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "all"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    parser.add_argument("--num-jobs", type=int, default=1, help="Number of parallel jobs to split the dataset regeneration into.")
    parser.add_argument("--job-id", type=int, default=0, help="The 0-indexed ID of this job.")
    parser.add_argument("--create-videos", action="store_true", default=False, help="Create verification videos for first few episodes of each task.")
    parser.add_argument("--max-videos-per-task", type=int, default=3, help="Maximum number of verification videos to create per task.")
    args = parser.parse_args()

    # Start data regeneration
    if args.libero_task_suite == "all":
        task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
        import copy
        original_args = copy.deepcopy(args)

        raw_dir_suite_found = None
        for suite in task_suites:
            if suite in original_args.libero_raw_data_dir:
                raw_dir_suite_found = suite
                break
        
        target_dir_suite_found = None
        for suite in task_suites:
            if suite in original_args.libero_target_dir:
                target_dir_suite_found = suite
                break
        
        for suite in task_suites:
            print(f"\nProcessing suite: {suite}")
            suite_args = copy.deepcopy(original_args)
            suite_args.libero_task_suite = suite
            
            if raw_dir_suite_found:
                suite_args.libero_raw_data_dir = original_args.libero_raw_data_dir.replace(raw_dir_suite_found, suite)
            
            if target_dir_suite_found:
                suite_args.libero_target_dir = original_args.libero_target_dir.replace(target_dir_suite_found, suite)
            
            print(f"  Raw data dir: {suite_args.libero_raw_data_dir}")
            print(f"  Target dir: {suite_args.libero_target_dir}")

            main(suite_args)
    else:
        main(args) 