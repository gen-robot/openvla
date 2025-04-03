"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import robosuite.utils.transform_utils as T
from robosuite.utils.observables import Observable
from scipy.spatial.transform import Rotation as R

import wandb

# Plot the action curve for check
import matplotlib.pyplot as plt

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_local_vla: bool = True                      # If True, uses local VLA model

    window_size: Optional[int] = None                # If provided, uses a sliding window of this size to chunk the past observations and actions
    num_actions_chunk: Optional[int] = None          # If provided, uses a action chunk of this size to chunk the future actions

    use_parallel_decoding: bool = True               # If True, uses parallel decoding inside LLaMa model's sdpa attention, i.e., replacing causal mask with bidirectional mask
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 250                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim, action_dim=ACTION_DIM, num_actions_chunk=cfg.num_actions_chunk)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    cfg.run_id = run_id

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


import h5py
import cv2

def save_data(save_path, count, obs_image_array, wrist_image_array, state_array, joint_state_array, action_array, is_correction_array):
    file_path = os.path.join(save_path, 'motionplanning', 'data.h5')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    count -= 1

    if count == 0:
        with h5py.File(file_path, 'w') as f:
            traj = f.create_group(f'traj_{count}')

            traj.create_group("obs").create_group("agent").create_dataset('state', data=np.array(state_array))
            traj["obs"]["agent"].create_dataset('joint_state', data=np.array(joint_state_array))
            traj.create_dataset('is_correction', data=np.array(is_correction_array))
            traj.create_dataset('actions', data=np.array(action_array))
    else:
        with h5py.File(file_path, 'a') as f:
            traj = f.create_group(f'traj_{count}')

            traj.create_group("obs").create_group("agent").create_dataset('state', data=np.array(state_array))
            traj["obs"]["agent"].create_dataset('joint_state', data=np.array(joint_state_array))
            traj.create_dataset('is_correction', data=np.array(is_correction_array))
            traj.create_dataset('actions', data=np.array(action_array))
    
    count_head = count // 100
    count_tail = count % 100
    image_save_path = os.path.join(save_path, 'full', str(count_head), str(count_tail))

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    else:
        for f in os.listdir(image_save_path):
            file_path = os.path.join(image_save_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for i, img in enumerate(obs_image_array):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_save_path, f"{i}.png"), img_rgb)

    image_save_path = os.path.join(save_path, 'wrist', str(count_head), str(count_tail))
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    else:
        for f in os.listdir(image_save_path):
            file_path = os.path.join(image_save_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for i, img in enumerate(wrist_image_array):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_save_path, f"{i}.png"), img_rgb)

def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    total_count=0,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != cfg.num_actions_chunk:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the cfg.num_actions_chunk "
               "{cfg.num_actions_chunk} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    goal_obj_pose = env.env.object_states_dict["wooden_cabinet_1_middle_region"].get_geom_state()
    
    goal_pos = np.array([goal_obj_pose["pos"][0], goal_obj_pose["pos"][1] + 0.2, goal_obj_pose["pos"][2]]).astype(np.float32)
    goal_quat = np.array([0.707, 0, 0, 0.707]).astype(np.float32)

    goal_pos_2 = np.array([goal_obj_pose["pos"][0], goal_obj_pose["pos"][1] + 0.05, goal_obj_pose["pos"][2]]).astype(np.float32)
    goal_pos_3 = np.array([goal_obj_pose["pos"][0], goal_obj_pose["pos"][1] + 0.3, goal_obj_pose["pos"][2]]).astype(np.float32)

    stage_1_step = 120
    stage_2_step = 55
    stage_3_step = 10
    stage_4_step = 55

    correction_t = 0

    # Run episode
    success = False
    do_correction = False

    max_steps = max_steps * 0.6

    action_array = []
    action_correction_array = []

    save_action_array = []
    save_state_array = []
    save_joint_state_array = []
    save_obs_image_array = []
    save_wrist_image_array = []
    is_correction_array = []

    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            save_joint_state_array.append(obs["robot0_joint_pos"])
            save_obs_image_array.append(obs["agentview_image"])
            save_wrist_image_array.append(obs["robot0_eye_in_hand_image"])
            save_state_array.append(observation["state"])
            is_correction_array.append(do_correction)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            if do_correction:
                if correction_t < stage_1_step + cfg.num_steps_wait:
                    # Execute action in environment
                    action[-1] = -1
                    action[0] = (goal_pos[0] - obs["robot0_eef_pos"][0]) * 3
                    action[1] = (goal_pos[1] - obs["robot0_eef_pos"][1]) * 3
                    action[2] = (goal_pos[2] - obs["robot0_eef_pos"][2]) * 3

                    current_pose = T.pose2mat((obs["robot0_eef_pos"], obs["robot0_eef_quat"]))
                    target_pose = T.pose2mat((goal_pos, goal_quat))

                    rot0 = R.from_matrix(current_pose[:3, :3])
                    rot1 = R.from_matrix(target_pose[:3, :3])

                    delta_rot = rot1 * rot0.inv()
                    delta_euler_angle = delta_rot.as_euler('xyz', degrees=False)

                    action[3] = delta_euler_angle[0] * 0.1
                    action[4] = delta_euler_angle[1] * 0.1
                    action[5] = delta_euler_angle[2] * 0.1
                elif correction_t < stage_1_step + stage_2_step + cfg.num_steps_wait:
                    action[-1] = -1
                    action[0] = (goal_pos_2[0] - obs["robot0_eef_pos"][0]) * 3
                    action[1] = (goal_pos_2[1] - obs["robot0_eef_pos"][1]) * 3
                    action[2] = (goal_pos_2[2] - obs["robot0_eef_pos"][2]) * 3
                    action[3] = 0
                    action[4] = 0
                    action[5] = 0
                elif correction_t < stage_1_step + stage_2_step + stage_3_step + cfg.num_steps_wait:
                    action[-1] = 1
                    action[0] = 0
                    action[1] = 0
                    action[2] = 0
                    action[3] = 0
                    action[4] = 0
                    action[5] = 0
                else:
                    action[-1] = 1
                    action[0] = (goal_pos_3[0] - obs["robot0_eef_pos"][0]) * 3
                    action[1] = (goal_pos_3[1] - obs["robot0_eef_pos"][1]) * 3
                    action[2] = (goal_pos_3[2] - obs["robot0_eef_pos"][2]) * 3
                    action[3] = 0
                    action[4] = 0
                    action[5] = 0
                
                correction_t += 1

            if do_correction:
                action_correction_array.append(action)
            else:
                action_array.append(action)

            save_action_array.append(action)

            obs, reward, done, info = env.step(action.tolist())
            # print("test:", env.env.object_states_dict["wooden_cabinet_1_middle_region"].get_geom_state())
            # print("OBS?:", obs["robot0_joint_pos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"])
            if done:
                success = True
                break
            if t + 1 < max_steps + cfg.num_steps_wait:
                t += 1
            else:
                do_correction = True
            
            if do_correction and correction_t > stage_1_step + stage_2_step + stage_3_step + stage_4_step:
                break

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    print("task_description:", task_description)

    # # Save the traj data
    # save_dir = "/nvme_data/liangzhi/dataset/libero_correction/goal/"
    # os.makedirs(save_dir, exist_ok=True)

    # if do_correction and success:
    #     total_count += 1
    #     save_data(save_dir, 
    #             total_count, 
    #             save_obs_image_array, 
    #             save_wrist_image_array, 
    #             save_state_array, 
    #             save_joint_state_array, 
    #             save_action_array, 
    #             is_correction_array)

    if total_count >= 100:
        exit(0)

    # Plot the action
    # save_dir = "./rollouts/image_test/"
    # os.makedirs(save_dir, exist_ok=True)

    # action_array = np.array(action_array)
    # action_correction_array = np.array(action_correction_array)
    # num_fig = action_array.shape[-1]
    
    # if len(action_array) != 0 and len(action_correction_array) != 0:

    
    #     fig, axs = plt.subplots(num_fig, 1, figsize=(10, 10))

    #     for i in range(num_fig):
    #         axs[i].plot(np.arange(len(action_array)), action_array[:, i], label="Action")
    #         axs[i].plot(np.arange(len(action_array), len(action_array) + len(action_correction_array)), action_correction_array[:, i], label="Action Correction")
    #         axs[i].set_title(f"Action {i}")
    #         axs[i].legend()

    #     num_files = len(os.listdir(save_dir))

    #     plt.savefig(os.path.join(save_dir, f"{num_files}.png"))

    return success, replay_images, total_count


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    total_count=0,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx // 5]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, total_count = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            total_count,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, run_id=cfg.run_id
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes, total_count


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    if cfg.num_actions_chunk is not None:
        cfg.future_action_window_size = cfg.num_actions_chunk - 1
        num_actions_chunk = cfg.num_actions_chunk
    else:
        cfg.future_action_window_size = None
        num_actions_chunk = cfg.num_actions_chunk = NUM_ACTIONS_CHUNK

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    total_count = 0

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes, total_count = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            total_count
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
