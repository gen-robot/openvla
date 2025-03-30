import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

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
    name: str = "bridge_orig"
    data_dir: str = "/nvme_data/embodied_agent/oxe_data/rlds"

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_local_vla: bool = True

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

    dataset = EpisodicRLDSDataset(
        data_root_dir=cfg.data_dir,
        data_mix=cfg.name,
        batch_transform=lambda x: x,
        resize_resolution=(256, 256),
        shuffle_buffer_size=0,
        image_aug=False,
        window_size=1,
        future_action_window_size=0,
        enable_cot=True,
    )

    os.makedirs(f"outputs/{cfg.name}/{timestamp}", exist_ok=True)
    f = open(f"outputs/{cfg.name}/{timestamp}/datasets.csv", "w")
    f.write("file_path, episode_id, episode_length, language_instruction, has_reasoning, video_path\n")

    for ep_idx, ep_data in tqdm(enumerate(dataset), desc="Visualizing dataset"):
        file_name = None
        episode_id = None
        language_instruction = None

        images = []
        actions = []
        has_reasoning = False

        vla_actions = []
        vla_images = []
        action_list = []

        for i, step_data in enumerate(ep_data):
            if file_name is None:
                file_name = step_data.get("file_name", b"").decode()
            if episode_id is None:
                episode_id = step_data.get("episode_id", str(ep_idx))
            if language_instruction is None:
                language_instruction = step_data["task"]["language_instruction"].decode()
            image_primary = step_data["observation"]["image_primary"][0]
            action = step_data["action"][0]
            reasoning = step_data["reasoning"].decode()
            if i == 0:
                has_reasoning = len(reasoning) > 0

            print(f">>>> Episode {episode_id} Step {i} / {len(ep_data)}")

            print("Instruction: ", language_instruction)
            print("Reasoning: ", reasoning)

            if len(reasoning) > 0:
                reasoning_parts = reasoning.split("@")
                tags = [(reasoning_parts[i], reasoning_parts[i + 1].rstrip()) for i in range(0, len(reasoning_parts), 2)]
                reasoning_text = " ".join([f" {tag[0]} {tag[1]}" for tag in tags])
            else:
                reasoning_text = ""

            merged_image = visualize_reasoning(image_primary, language_instruction, reasoning_text)
            merged_image_arr = merged_image #.numpy()
            images.append(merged_image_arr)
            actions.append(action)

            if vla is not None:
                observation = {
                    "full_image": image_primary,
                }

                if cfg.use_proprio:
                    observation["state"] = step_data["observation"]["proprio"][0]

                vla_action_chunk, generated_ids = get_vla_action(
                    cfg, vla, processor, observation, language_instruction, 
                    action_head=action_head, 
                    proprio_projector=proprio_projector, 
                    use_film=cfg.use_film, 
                    do_sample=False)

                if "cot" in cfg.pretrained_checkpoint:
                    generated_text = processor.batch_decode(generated_ids)[0]
                    print("Generated text: ", generated_text)
                    vla_image = visualize_reasoning(image_primary, language_instruction, generated_text)
                    vla_images.append(vla_image)

                if len(action_list) == 0:
                    for action in vla_action_chunk:
                        action_list.append(action)
                vla_action = action_list.pop(0)
                vla_actions.append(vla_action)

        short_instruction = "_".join(language_instruction.rstrip(".").split(" "))
        video_path = f"{short_instruction.rstrip('.').lower()}_ep{episode_id}"
        images_to_video(images, f"outputs/{cfg.name}/{timestamp}/videos", video_name=video_path, fps=5)
        if len(vla_images) > 0:
            images_to_video(vla_images, f"outputs/{cfg.name}/{timestamp}/vla_videos", video_name=video_path, fps=5)

        f.write(f"{file_name},{episode_id},{len(images)},{language_instruction},{has_reasoning},{video_path}\n")

        actions = np.array(actions)
        if len(vla_actions) > 0:
            vla_actions = np.array(vla_actions)
        fig, axs = plt.subplots(len(actions[0]), 1, figsize=(10, len(actions)))
        for i in range(len(actions[0])):
            axs[i].plot(actions[:, i], label="label")
            if len(vla_actions) > 0:
                axs[i].plot(vla_actions[:, i], label="pred")
            axs[i].legend()
        plt.tight_layout()
        plt.savefig(f"outputs/{cfg.name}/{timestamp}/{video_path}_actions.png")
        plt.close()

    f.close()

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)