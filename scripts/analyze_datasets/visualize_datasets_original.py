import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tyro
from prismatic.util.cot_utils import visualize_reasoning
from prismatic.util.img_utils import images_to_video
from prismatic.vla.datasets import EpisodicRLDSDataset
from prismatic.vla.datasets.datasets import RLDSBatchTransform
from prismatic.vla.datasets.rlds.dataset import make_single_dataset
from tqdm import tqdm
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action
)
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, STATE_DIM_MAP, ACTION_DIM_MAP

@dataclass
class Config:
    name: str = "libero_object_no_noops"
    data_dir: str = "/nvme_data/embodied_agent/libero_data/modified_libero_rlds_episode_id"
    original_dataset_dir: str = "/nvme_data/liangzhi/franka-dataset/process/pick/"

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_local_vla: bool = True
    cot_tags: str = None

    window_size: Optional[int] = 1                    # If provided, uses a sliding window of this size to chunk the past observations and actions
    future_action_window_size: Optional[int] = 0      # If provided, uses a future action window of this size to chunk the future actions

    use_parallel_decoding: bool = False               # If True, uses parallel decoding inside LLaMa model's sdpa attention, i.e., replacing causal mask with bidirectional mask
    use_l1_regression: bool = False                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = False                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 1                    # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    

def create_vla_and_processor(cfg: Config):
    if cfg.pretrained_checkpoint == "" or cfg.pretrained_checkpoint is None:
        return None, None
    
    ACTION_DIM = ACTION_DIM_MAP[OXE_DATASET_CONFIGS[cfg.name]["action_encoding"]]

    vla = get_vla(cfg, action_dim=ACTION_DIM)
    processor = get_processor(cfg)

    return vla, processor

def main(cfg: Config):
    assert cfg.name in OXE_DATASET_CONFIGS, f"Dataset {cfg.name} not found in OXE_DATASET_CONFIGS!"
    if 'ecot-openvla-7b-oxe' in cfg.pretrained_checkpoint and cfg.name == 'bridge_orig':
        cfg.unnorm_key = 'bridge_reasoning'
    else:
        cfg.unnorm_key = cfg.name
    vla, processor = create_vla_and_processor(cfg)
    PROPRIO_DIM = STATE_DIM = STATE_DIM_MAP[OXE_DATASET_CONFIGS[cfg.name]["state_encoding"]]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.pretrained_checkpoint is not None:
        timestamp = cfg.pretrained_checkpoint.rstrip('/').split('/')[-1] + "_" + timestamp

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)

    # Load continuous action head
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, vla.llm_dim)

    if vla is not None:
        assert cfg.unnorm_key in vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    os.makedirs(f"outputs/{cfg.name}/{timestamp}", exist_ok=True)

    for episode_dir in os.listdir(cfg.original_dataset_dir):
        episode_id = episode_dir.split("_")[-1]

        file_path = os.path.join(cfg.original_dataset_dir, episode_dir, "data.npy")
        traj_data = np.load(file_path, allow_pickle=True).item()

        traj_num_step = len(traj_data["actions"])
        action_list = []
        vla_action_list = []
        vla_action_list_plot = []
        input_image_list = []

        for i in tqdm(range(traj_num_step)):
            image_primary = traj_data["front_rgb"][i].astype(np.uint8)
            # (480, 480, 3) -> (256, 256, 3)
            image_primary = cv2.resize(image_primary, (256, 256), interpolation=cv2.INTER_AREA)
            input_image_list.append(image_primary)

            action_list.append(traj_data["actions"][i].astype(np.float32))
            if vla is not None:
                observation = {
                    "full_image": image_primary,
                }

                vla_action_chunk, generated_ids = get_vla_action(
                    cfg, vla, processor, observation, traj_data["insruction"], 
                    action_head=action_head, 
                    proprio_projector=proprio_projector, 
                    use_film=cfg.use_film, 
                    do_sample=False,
                    enable_cot=False,
                    gt_reasoning_text=""
                )

                if len(vla_action_list) == 0:
                    for action in vla_action_chunk:
                        vla_action_list.append(action)
                vla_action = vla_action_list.pop(0)
                vla_action_list_plot.append(vla_action)

        video_path = None
        if len(input_image_list) > 10:
            short_instruction = "_".join(traj_data["insruction"].rstrip(".").split(" "))
            video_path = f"{short_instruction.rstrip('.').lower()}_ep{episode_id}"
            images_to_video(input_image_list, f"outputs/{cfg.name}/{timestamp}/videos", video_name=video_path, fps=20)
        
        if len(input_image_list) > 10:
            actions = np.array(action_list)
            if len(vla_action_list_plot) > 0:
                vla_actions = np.array(vla_action_list_plot)
            fig, axs = plt.subplots(len(actions[0]), 1, figsize=(10, len(actions)))
            for i in range(len(actions[0])):
                axs[i].plot(actions[:, i], label="label")
                if len(vla_actions) > 0:
                    axs[i].plot(vla_actions[:, i], label="pred")
                axs[i].legend()
            plt.tight_layout()
            plt.savefig(f"outputs/{cfg.name}/{timestamp}/{video_path}_actions.png")
            plt.close()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)