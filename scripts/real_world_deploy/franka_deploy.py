"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import tyro
import datetime

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
    name: str = "panda_rlds_dataset_real"

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_local_vla: bool = True
    cot_tags: str = None

    window_size: Optional[int] = 1                    # If provided, uses a sliding window of this size to chunk the past observations and actions
    future_action_window_size: Optional[int] = 0      # If provided, uses a future action window of this size to chunk the future actions
    pic_process: str = "cut"
    sim_normalize: bool = False
    do_sample: bool = False

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

    #################################################################################################################
    # Real world
    #################################################################################################################
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 9876                                                    # Host Port


# === Server Interface ===
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
            # print("norm_stats:", vla.norm_stats)
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

            if self.cfg.pic_process == "cut":
                image_full_original = image[1, 40:520, :, :]
                image_wrist_original = image[0, 80:560, :, :]
            elif self.cfg.pic_process == "origin":
                image_full_original = image[1, :, :, :]
                image_wrist_original = image[0, :, :, :]
            elif self.cfg.pic_process == "pad":
                image_full_original = np.zeros((640, 640, 3), dtype=np.uint8)
                image_wrist_original = np.zeros((640, 640, 3), dtype=np.uint8)
                image_full_original[80:560, :, :] = image[1, :, :, :]
                image_wrist_original[80:560, :, :] = image[0, :, :, :]

            image_primary = cv2.resize(image_full_original, (256, 256), interpolation=cv2.INTER_AREA)
            image_wrist = cv2.resize(image_wrist_original, (256, 256), interpolation=cv2.INTER_AREA)
            # instruction = "put carrot on plate"
            # unnorm_key = "bridge_orig" #"pmc16384"
            instruction = "Pick up the object on the table and place it into the white tray."
            unnorm_key = "panda_rlds_dataset_real"

            # print("image:", image_primary)

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
                    do_sample=self.cfg.do_sample,
                    enable_cot=False,
                    gt_reasoning_text=""
                )
                # print("check:", vla_action_chunk)
                for action in vla_action_chunk:
                    self.vla_action_list.append(action)
            
            vla_actions = []
            while len(self.vla_action_list) > 0:
                vla_action = self.vla_action_list.pop(0)
                if self.cfg.sim_normalize:
                    vla_action[-1] = (vla_action[-1] + 1) / 2
                else:
                    vla_action[-1] = 1 - vla_action[-1]
                if vla_action[-1] < 0.5:
                    vla_action[-1] = 0
                if vla_action[-1] >= 0.5:
                    vla_action[-1] = 1
                vla_actions.append(vla_action)

            vla_actions = np.array(vla_actions, dtype=np.float32)

            if double_encode:
                return JSONResponse(json_numpy.dumps(vla_actions))
            else:
                return JSONResponse(vla_actions)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

def deploy(cfg: Config) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    deploy(cfg)
