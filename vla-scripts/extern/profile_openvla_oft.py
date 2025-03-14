import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_vla,
    get_action_head,
    get_processor,
    get_proprio_projector,
    prepare_images_for_vla,
    normalize_proprio
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


INSTRUCTION = "put spoon on towel"
PROPRIO_DIM = 14
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}

def get_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


@dataclass
class OpenVLAConfig:
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_local_vla: bool = True

    window_size: Optional[int] = None                # If provided, uses a sliding window of this size to chunk the past observations and actions
    future_action_window_size: Optional[int] = None  # If provided, uses a future action window of this size to chunk the future actions

    use_parallel_decoding: bool = True               # If True, uses parallel decoding inside LLaMa model's sdpa attention, i.e., replacing causal mask with bidirectional mask
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25                    # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    
    # fmt: on


@torch.inference_mode()
@draccus.wrap()
def profile(cfg: OpenVLAConfig) -> None:

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Convert relative checkpoint path to absolute path
    if cfg.pretrained_checkpoint:
        assert os.path.exists(cfg.pretrained_checkpoint), f"Wrong path: {cfg.pretrained_checkpoint}"
        cfg.pretrained_checkpoint = os.path.abspath(cfg.pretrained_checkpoint)
        print(f"Using absolute checkpoint path: {cfg.pretrained_checkpoint}")

    vla = get_vla(cfg)
    processor = get_processor(cfg)

    resize_size = get_image_resize_size(cfg) # openvla's image size is 224x224

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)

    # Load continuous action head
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, vla.llm_dim)

    # Check that the model contains the action un-normalization key
    assert cfg.unnorm_key in vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    total_time = 0
    for _ in range(10):
        prompt = get_openvla_prompt(INSTRUCTION)
       
        def create_random_image(size: tuple[int, int]) -> Image.Image:
            return np.asarray(np.random.rand(*size) * 255, dtype=np.uint8)

        all_images = [create_random_image((224, 224, 3)) for _ in range(cfg.num_images_in_input)]

        t0 = time.time()
        prepared_images = prepare_images_for_vla(all_images, cfg)

        # process primary image
        primary_image = prepared_images.pop(0)
        inputs = processor(prompt, primary_image).to(device, dtype=torch.bfloat16)

        # process additional wrist images if any
        if prepared_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(device, dtype=torch.bfloat16) for image_wrist in prepared_images
            ]
            # concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        

        # process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = np.asarray(np.random.rand(PROPRIO_DIM), dtype=np.float32)
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio, proprio_norm_stats)

        # generate action
        if action_head is None:
            action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                action_head=action_head,
                use_film=cfg.use_film,
            )

        effective_action = [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]

        # Simulate applying actions to a real robot
        for i, act in enumerate(effective_action):
            # Simulate robot execution time (e.g., 100ms per action)
            robot_execution_time = 0.05  # 100ms
            time.sleep(robot_execution_time)
            
            # Print action being applied (optional)
            print(f"Applied action {i+1}/{len(effective_action)}: {act.shape}")
        
        # Record total time including robot execution
        total_time += time.time() - t0
        
    # Calculate and print average time
    avg_time = total_time / 10
    print(f"Average execution time (including simulated robot actions): {avg_time:.4f} seconds")
    print(f"Average time per action: {avg_time / len(effective_action):.4f} seconds")
    # print the max GPU memory allocated
    print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    profile()

    