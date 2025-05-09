import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
import json

import torch
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
    print("???:", cfg.unnorm_key)
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
        window_size=cfg.window_size if cfg.window_size is not None else 1,
        future_action_window_size=cfg.future_action_window_size if cfg.future_action_window_size is not None else 0,
        enable_cot=True,
    )

    os.makedirs(f"outputs/{cfg.name}/{timestamp}", exist_ok=True)
    with open(f"outputs/{cfg.name}/{timestamp}/datasets.csv", "w") as f:
        f.write("file_path, episode_id, episode_length, language_instruction, has_reasoning, video_path\n")

    # Liangzhi: TODO, load json save in training, check token ids
    with open("/nvme_data/arm_ws/EmbodiedAgent/quick_jump/openvla/outputs/debug/save.json", "r") as f:
        load_data = json.load(f)

    check_data = load_data[0]

    for ep_idx, ep_data in tqdm(enumerate(dataset), desc="Visualizing dataset", total=len(dataset)):
        if ep_data[0]["episode_id"].decode('utf-8') != check_data["metadata"]["episode_ids"][0]:
            continue
        print(f"Find traj {check_data['metadata']['episode_ids'][0]}!")
        print("Timestep:", check_data["metadata"]["timesteps"][0])

        # for i, step_data in enumerate(ep_data):
        #     # Liangzhi: speed up test
        #     if i != check_data["metadata"]["timesteps"][0]:
        #         continue
        for idx in range(9):
            idx = 8
            file_name = None
            episode_id = None
            language_instruction = None

            images = []
            actions = []
            has_reasoning = False
            has_reasoning_check = False

            vla_actions = []
            vla_images = []
            action_list = []
            for i, step_data in enumerate(ep_data):
                # i = check_data["metadata"]["timesteps"][0]
                # step_data = ep_data[i]
                if i % 10 != 0:
                    continue

                if file_name is None:
                    file_name = step_data.get("file_name", b"").decode()
                if episode_id is None:
                    episode_id = step_data.get("episode_id", str(ep_idx))
                    # if isinstance(episode_id, bytes):
                    #     episode_id = episode_id.decode()
                if language_instruction is None:
                    language_instruction = step_data["task"]["language_instruction"].decode()
                image_primary = step_data["observation"]["image_primary"][0]
                action = step_data["action"][0]
                reasoning = step_data["reasoning"].decode()
                if not has_reasoning_check:
                    has_reasoning = len(reasoning) > 0
                    has_reasoning_check = True
                if not has_reasoning:
                    break

                if len(reasoning) > 0:
                    reasoning_parts = reasoning.split("@")
                    tags = [(reasoning_parts[i], reasoning_parts[i + 1].rstrip()) for i in range(0, len(reasoning_parts), 2)]
                    if idx == len(tags):
                        end_str = " ACTION: "
                    else:
                        end_str = f" {tags[idx][0]}"
                    tags = tags[:idx]
                    # print("?????:", len(tags))
                    # exit(0)
                    reasoning_text = "".join([f" {tag[0]} {tag[1]}" for tag in tags]) + end_str
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

                    # Liangzhi: Give ground truth reasoning text for test
                    vla_action_chunk, generated_ids = get_vla_action(
                        cfg, vla, processor, observation, language_instruction, 
                        action_head=action_head, 
                        proprio_projector=proprio_projector, 
                        use_film=cfg.use_film, 
                        do_sample=False,
                        enable_cot="cot" in cfg.pretrained_checkpoint,
                        gt_reasoning_text=reasoning_text,
                        is_debug=True)

                    # print("ID check(generated):", generated_ids[0])
                    # print("ID check(labels):", check_data["labels"][0])
                    # output_generated_ids = generated_ids[0].cpu().numpy().tolist()[30:]
                    # check_data_ids = check_data["labels"][0][29:]
                    # l1 = 0
                    # l2 = 0
                    # check_success = True
                    # for check_idx in range(500):
                    #     if not check_success:
                    #         print(check_idx, ":", output_generated_ids[l1], check_data_ids[l2])
                    #         l1 += 1
                    #         l2 += 1
                    #         continue
                    #     if output_generated_ids[l1] == check_data_ids[l2]:
                    #         print(check_idx, ":", output_generated_ids[l1], check_data_ids[l2])
                    #         l1 += 1
                    #         l2 += 1
                    #         continue
                    #     if output_generated_ids[l1] == check_data_ids[l2 + 1]:
                    #         print(check_idx, ":", "Skip", check_data_ids[l2])
                    #         l2 += 1
                    #         continue
                    #     if output_generated_ids[l1 + 1] == check_data_ids[l2]:
                    #         print(check_idx, ":", output_generated_ids[l1], "Skip")
                    #         l1 += 1
                    #         continue
                    #     print(check_idx, ":", output_generated_ids[l1], check_data_ids[l2])
                    #     l1 += 1
                    #     l2 += 1
                    #     check_success = False

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

            video_path = None
            if len(images) > 5:
                short_instruction = "_".join(language_instruction.rstrip(".").split(" "))
                video_path = f"{short_instruction.rstrip('.').lower()}_ep{episode_id}_lv{idx}"
                images_to_video(images, f"outputs/{cfg.name}/{timestamp}/videos", video_name=video_path, fps=20)
            if len(vla_images) > 5:
                images_to_video(vla_images, f"outputs/{cfg.name}/{timestamp}/vla_videos", video_name=video_path, fps=5)

            with open(f"outputs/{cfg.name}/{timestamp}/datasets.csv", "a") as f:
                f.write(f"{file_name},{episode_id},{len(images)},{language_instruction},{has_reasoning},{video_path}\n")

            if len(images) > 10:
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
                plt.savefig(f"outputs/{cfg.name}/{timestamp}/{video_path}_actions_lv{idx}.png")
                plt.close()

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)