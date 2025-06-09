"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR> \
        [--num-jobs <NUM_JOBS>] [--job-id <JOB_ID>]

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

    Example (Multiprocess):
        python experiments/robot/libero/regenerate_libero_dataset.py \
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

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
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
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        if args.job_id == 0:
            print(f"Warning: Target directory {args.libero_target_dir} already exists. Files may be overwritten.")
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    if args.num_jobs > 1:
        metainfo_json_out_path = f"./{args.libero_task_suite}_metainfo.job_{args.job_id}.json"
    else:
        metainfo_json_out_path = f"./{args.libero_task_suite}_metainfo.json"
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
        env, task_description = get_libero_env(
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

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
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
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
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
