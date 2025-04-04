import argparse
import json
import os
import time
import warnings
import tqdm
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils import NumpyFloatValuesEncoder, post_process_caption
from prismatic import load


# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int, default=0)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--dataset_name", type=str)

args = parser.parse_args()
result_path = f"./outputs/{args.dataset_name}/bboxes"
os.makedirs(result_path, exist_ok=True)
bbox_json_path = os.path.join(result_path, f"results_bboxes_{args.id}.json")

print("Loading data...")
split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=f"train[{start}%:{end}%]")
print("Done.")

# Load Prismatic VLM
if args.gpu is not None:
    device = f"cuda:{args.gpu}"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

hf_token = os.environ["HF_TOKEN"]
vlm_model_id = "prism-dinosiglip+7b"

print(f"Loading Prismatic VLM ({vlm_model_id})...")
vlm = load(vlm_model_id, hf_token=hf_token)
vlm = vlm.to(device, dtype=torch.bfloat16)
print("Done.")

# Load gDINO model
model_id = "IDEA-Research/grounding-dino-base"
print(f"Loading gDINO to device {device}...")
processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Done.")

BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3

def create_user_prompt(lang_instruction):
    user_prompt = "List all the objects you can see in this image, especially including any objects mentioned in the language instruction. Format your response as a simple list of object names separated by periods (e.g., 'cup. table. robot gripper.'). Be specific and comprehensive, but avoid using commas or other punctuation."
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt

def post_process_object_list(caption):
    """
    Process the VLM output to create a clean list of objects separated by periods.
    This format works better for gDINO object detection.
    """
    # Remove any explanatory text or prefixes
    if ":" in caption:
        caption = caption.split(":", 1)[1]
    
    # Replace commas with periods
    caption = caption.replace(",", ".")
    
    # Replace other list markers and clean up
    caption = caption.replace("-", "").replace("•", "").replace("\n", " ")
    
    # Split by periods, clean each item, and rejoin
    items = [item.strip() for item in caption.split(".") if item.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_items = [item for item in items if not (item in seen or seen.add(item))]
    
    # Join with periods
    result = ". ".join(unique_items)
    
    # Ensure it ends with a period
    if not result.endswith("."):
        result += "."
        
    return result

bbox_results_json = {}
for ep_idx, episode in tqdm.tqdm(
    enumerate(ds),
    total=len(ds),
    desc=f"Generating bounding boxes [{start}%:{end}%]",
):

    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    if isinstance(episode_id, bytes):
        episode_id = episode_id.decode()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"ID {args.id} starting ep: {episode_id}, {file_path}")

    if file_path not in bbox_results_json.keys():
        bbox_results_json[file_path] = {}

    start = time.time()
    bboxes_list = []
    for step_idx, step in tqdm.tqdm(
        enumerate(episode["steps"]),
        total=len(episode["steps"]),
        desc=f"Generating bounding boxes for Episode {episode_id}",
        leave=False,
    ):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())
        
        # Generate object list using Prismatic VLM
        user_prompt = create_user_prompt(lang_instruction)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        torch.manual_seed(0)
        object_list = vlm.generate(image, prompt_text, do_sample=True, temperature=0.4, max_new_tokens=64, min_length=1)
        # Post-process the object list for gDINO
        object_list = post_process_object_list(object_list)
        # print(f"Objects detected: {object_list}")
        
        # Use the object list for gDINO
        inputs = processor(
            images=image,
            text=object_list,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[image.size[::-1]])[0]

        logits, phrases, boxes = (
            results["scores"].cpu().numpy(),
            results["labels"],
            results["boxes"].cpu().numpy(),
        )

        bboxes = []
        for lg, p, b in zip(logits, phrases, boxes):
            b = list(b.astype(int))
            lg = round(lg, 5)
            bboxes.append((lg, p, b))

        bboxes_list.append(bboxes)
        # break
    end = time.time()
    bbox_results_json[file_path][str(ep_idx)] = {
        "episode_id": str(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)
    print(f"ID {args.id} finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
