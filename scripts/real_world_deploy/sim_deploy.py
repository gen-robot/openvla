"""
Real world deployment script for simulation testing
"""
import os
import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union, Annotated, List

import cv2
import tyro
import datetime

import gymnasium as gym
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import RecordEpisode

import numpy as np
import argparse
import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action
)
from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, STATE_DIM_MAP, ACTION_DIM_MAP

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


@dataclass
class Config:
    name: str = "panda_rlds_dataset"

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_local_vla: bool = True
    cot_tags: str = None

    window_size: Optional[int] = 1                    # If provided, uses a sliding window of this size to chunk the past observations and actions
    future_action_window_size: Optional[int] = 0      # If provided, uses a future action window of this size to chunk the future actions
    directly_resize: bool = False

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
    num_traj: int = 100

    #################################################################################################################
    # Real world
    #################################################################################################################
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 9876                                                    # Host Port

    #################################################################################################################
    # Maniskill environment parameters
    #################################################################################################################

    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    reward_mode: Optional[str] = None
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_ee_target_delta_pose"
    render_mode: str = "rgb_array"
    shader: str = "default"
    record_dir: Optional[str] = None
    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    quiet: bool = False
    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 0


object_name_dict = {
    0: "001_carrot_simpler",
    1: "002_kitchen shovel_1",
    2: "003_bread_1",
    3: "004_plastic bottle_1",
    4: "005_7up can_1",
    5: "006_zuchinni_1",
    6: "007_ketchup bottle_1",
    7: "008_watering can_1",
    8: "009_pipe_1",
    9: "010_toy bear_1",
    10: "011_fast food cup_1",
    11: "012_plant_1",
    12: "013_banana_1",
    13: "014_hamburger_1",
    14: "015_golf ball_1",
    15: "016_BBQ sauce_1",
    16: "017_travel cup_1",
    17: "018_pepper_1",
    18: "019_nonstop can_1",
    19: "020_potato_1",
    20: "021_baguette_1",
    21: "022_champagne glass_1",
    22: "023_kitchen spoon_1",
    23: "024_onion_1",
    24: "025_cup_1",
}

class OpenVLAServer:
    def __init__(self, cfg: Config) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        assert cfg.name in OXE_DATASET_CONFIGS, f"Dataset {cfg.name} not found in OXE_DATASET_CONFIGS!"
        if 'ecot-openvla-7b-oxe' in cfg.pretrained_checkpoint and cfg.name == 'bridge_orig':
            cfg.unnorm_key = 'bridge_reasoning'
        else:
            cfg.unnorm_key = cfg.name
        vla, processor = self.create_vla_and_processor(cfg)
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
            action_head = get_action_head(cfg, vla.llm_dim, num_actions_chunk=cfg.future_action_window_size + 1)

        if vla is not None:
            assert cfg.unnorm_key in vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        self.cfg = cfg
        self.vla = vla
        self.processor = processor
        self.action_head = action_head
        self.timestamp = timestamp
        self.proprio_projector = proprio_projector
        self.vla_action_list = []

    def create_vla_and_processor(self, cfg: Config):
        if cfg.pretrained_checkpoint == "" or cfg.pretrained_checkpoint is None:
            return None, None
        
        ACTION_DIM = ACTION_DIM_MAP[OXE_DATASET_CONFIGS[cfg.name]["action_encoding"]]

        vla = get_vla(cfg, action_dim=ACTION_DIM)
        processor = get_processor(cfg)

        return vla, processor

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["images"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            if not self.cfg.directly_resize:
                image_full_original = image[1, 40:520, :, :]
                image_wrist_original = image[0, 80:560, :, :]
            else:
                image_full_original = image[1, :, :, :]
                image_wrist_original = image[0, :, :, :]
            
            image_primary = cv2.resize(image_full_original, (256, 256), interpolation=cv2.INTER_AREA)
            image_wrist = cv2.resize(image_wrist_original, (256, 256), interpolation=cv2.INTER_AREA)
            instruction = "Pick up the object on the table and place it into the white tray."
            unnorm_key = "panda_rlds_dataset"

            observation = {
                "full_image": image_primary,
                "image_wrist": image_wrist,
            }

            if len(self.vla_action_list) == 0:
                vla_action_chunk, generated_ids = get_vla_action(
                    self.cfg, self.vla, self.processor, observation, instruction, 
                    action_head=self.action_head, 
                    proprio_projector=self.proprio_projector, 
                    use_film=self.cfg.use_film, 
                    do_sample=True,
                    enable_cot=False,
                    gt_reasoning_text=""
                )
                # print("check:", vla_action_chunk)
                for action in vla_action_chunk:
                    self.vla_action_list.append(action)
            
            vla_actions = []
            while len(self.vla_action_list) > 0:
                vla_action = self.vla_action_list.pop(0)
                vla_action[-1] = 1 - vla_action[-1]
                vla_action[-1] = 2 * vla_action[-1] - 1
                vla_actions.append(vla_action)

            vla_actions = np.array(vla_actions, dtype=np.float32)

            if double_encode:
                return json_numpy.dumps(vla_actions)
            else:
                return vla_actions
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self) -> None:
        env_kwargs = dict(
            obs_mode=self.cfg.obs_mode,
            reward_mode=self.cfg.reward_mode,
            control_mode=self.cfg.control_mode,
            render_mode=self.cfg.render_mode,
            sensor_configs=dict(shader_pack=self.cfg.shader),
            human_render_camera_configs=dict(shader_pack=self.cfg.shader),
            viewer_camera_configs=dict(shader_pack=self.cfg.shader),
            sim_config=dict(control_freq=5), # currently is 20, align with data, should be carefully
            num_envs=self.cfg.num_envs,
            sim_backend=self.cfg.sim_backend,
            render_backend="gpu",
            enable_shadow=True,
            parallel_in_single_scene=False,
        )

        if self.cfg.robot_uids is not None:
            env_kwargs["robot_uids"] = tuple(self.cfg.robot_uids.split(","))
        env: BaseEnv = gym.make(
            self.cfg.env_id,
            **env_kwargs
        )

        run_dir = f"./results/{self.cfg.env_id}/{self.timestamp}"
        os.makedirs(run_dir, exist_ok=True)

        record_dir = run_dir

        record_dir = record_dir.format(env_id=self.cfg.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False,
                            max_steps_per_video=200)

        def clip(x, max_value):
            """Clip the value to the range [-max_value, max_value]"""
            return max(-max_value, min(max_value, x))

        success_num = 0

        for idx in range(self.cfg.num_traj):
            episode_id = torch.randint(10000000000000, (env.num_envs,), device=env.device)
            obs, _ = env.reset(seed=self.cfg.seed, options=dict(episode_id=episode_id, obj_set="test"))

            action_list = []
            success_check = False

            max_steps = 200
            for i in range(max_steps):
                if len(action_list) == 0: 
                    img_tensor = obs["sensor_data"]["c19_front_view"]["rgb"][0].to(torch.uint8).cpu().numpy()
                    lang = "Pick up the object on the table and place it into the white tray."

                    image = np.stack([img_tensor, img_tensor], axis=0)
                    payload = {
                        "images": image,
                        "instruction": lang,
                        "unnorm_key": "panda_rlds_dataset"
                    }

                    action = self.predict_action(payload)

                    # save all actions into list
                    if len(action.shape) == 1:
                        action_list.append(action)
                    else:
                        for a in action:
                            action_list.append(a)
                
                action = action_list.pop(0)
                obs, reward, terminated, truncated, info = env.step(action)
                
                obj_name = object_name_dict[env.select_carrot_ids[0].item()]

                delta_pos = (env.objs_plate["001_plate_simpler"].pose.p - env.objs_carrot[obj_name].pose.p)[0]
                success_check = (np.linalg.norm(delta_pos[:2]) < 0.05 and np.abs(delta_pos[2]) < 0.05)

                if success_check:
                    break
            
            if success_check:
                success_num += 1
            
            print("Episode:", idx, "Success:", success_check, "Success Rate:", success_num / (idx + 1))




def deploy(cfg: Config) -> None:
    server = OpenVLAServer(cfg)
    server.run()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    deploy(cfg)