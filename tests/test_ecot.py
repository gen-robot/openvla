import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import time
import numpy as np
import cv2
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont
import enum
import json
import h5py

import requests
from io import BytesIO
from PIL import Image

import os

from prismatic.util.cot_utils import (
    CotTag, split_reasoning, get_cot_tags_list, get_metadata
)
from prismatic.util.img_utils import draw_2d_points, draw_bboxes
from prismatic.util.rlds_utils import get_data_from_rlds


device = "cuda:0"

# Load Processor & VLA
# NOTE: Requires ~15 GB of GPU memory, but can enable load_in_4bit to reduce
# model memory usage to ~5 GB
# path_to_converted_ckpt = "/home/gaofeng/arm_ws/openvla/checkpoints/bridge_rt_1/oft+g2tb8+openvla-7b+bridge_rt_1+b4+lr-0.0005+lora-r64+dropout-0.0+chunk-1+cot--30000_chkpt"
path_to_converted_ckpt = "/nvme_data/embodied_agent/pretrained/ecot-openvla-7b-oxe"
processor = AutoProcessor.from_pretrained(path_to_converted_ckpt, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    path_to_converted_ckpt,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # [Optional] Set `load_in_4bit=True` if not enough GPU memory + run `pip install bitsandbytes`
    # Then, comment out `.to(device)`
).to(device)
# if os.path.isdir(path_to_converted_ckpt):
#     with open(Path(path_to_converted_ckpt) / "dataset_statistics.json", "r") as f:
#         vla.norm_stats = json.load(f)

# Create prompt
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def get_openvla_prompt(instruction: str, old_version: bool = False) -> str:
    if old_version:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut: TASK:"

rlds_data = get_data_from_rlds(
    "/home/gaofeng/arm_ws/openvla/datasets", 
    "bridge_orig", 
    split="train",)

for i, (key, value) in enumerate(rlds_data.items()):
    print(f"Processing {i} / {len(rlds_data)}: {key}")
    instruction = value["language_instruction"]
    images = value["observation"]["image_0"]
    image = images[len(images) // 2]

    prompt = get_openvla_prompt(instruction, old_version=False)

    # Run inference
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # import pdb; pdb.set_trace()
    # Run OpenVLA Inference
    start_time = time.time()

    torch.manual_seed(0)
    action, generated_ids = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False, max_new_tokens=1024)
    generated_text = processor.batch_decode(generated_ids)[0]
    print(f"Time: {time.time() - start_time:.4f} || Action: {action}")

    #@title Visualize reasoning and image
    tags = [f" {tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(generated_text, tags)
    text = ['Instruction: ' + instruction]
    text += [tag + reasoning[tag] for tag in [' TASK:',' PLAN:',' SUBTASK REASONING:',' SUBTASK:',
                                            ' MOVE REASONING:',' MOVE:', ' VISIBLE OBJECTS:', ' GRIPPER POSITION:'] if tag in reasoning]
    metadata = get_metadata(reasoning)
    bboxes = {}
    for k, v in metadata["bboxes"].items():
        if k[0] == ",":
            k = k[1:]
        bboxes[k.lstrip().rstrip()] = v

    caption = ""
    for t in text:
        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
        word_list = wrapper.wrap(text=t)
        caption_new = ''
        for ii in word_list[:-1]:
            caption_new = caption_new + ii + '\n      '
        caption_new += word_list[-1]

        caption += caption_new.lstrip() + "\n\n"

    base = Image.fromarray(np.ones((512, 640, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(base)
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=14)
    color = (0,0,0) # RGB
    draw.text((30, 30), caption, color, font=font)

    # import pdb; pdb.set_trace()
    img_arr = np.array(image)
    draw_2d_points(img_arr, metadata["gripper"], img_size=(256, 256))
    draw_bboxes(img_arr, bboxes, img_size=(256, 256))

    text_arr = np.array(base)
    # resize text_arr to make it can be concatenated with img_arr at the same height, keep the aspect ratio
    target_height = img_arr.shape[0]
    text_arr = cv2.resize(text_arr, (int(text_arr.shape[1] * target_height / text_arr.shape[0]), target_height))

    # import pdb; pdb.set_trace()
    reasoning_img = Image.fromarray(np.concatenate([img_arr, text_arr], axis=1))

    # save reasoning_img
    reasoning_img.save(f"own_reasoning_img_{i}.png")

    for k, v in reasoning.items():
        print(k, v)