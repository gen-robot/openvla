import argparse
import json
import os
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder

from prismatic import load

import tensorflow as tf
# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--splits", default=1, type=int)
parser.add_argument("--results_path", default=None, type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--data_dir", type=str)

args = parser.parse_args()

if args.gpu is not None:
    device = f"cuda:{args.gpu}"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

hf_token = os.environ["HF_TOKEN"]
vlm_model_id = "prism-dinosiglip+7b"

warnings.filterwarnings("ignore")

split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

# Load Bridge V2
ds = tfds.load(
    args.dataset_name,
    data_dir=args.data_dir,
    split=f"train[{start}%:{end}%]",
)

# Load Prismatic VLM
print(f"Loading Prismatic VLM ({vlm_model_id})...")
vlm = load(vlm_model_id, hf_token=hf_token)
vlm = vlm.to(device, dtype=torch.bfloat16)

if args.results_path is None:
    args.results_path = f"./outputs/{args.dataset_name}/descriptions"
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
results_json_path = os.path.join(args.results_path, f"results_{args.id}.json")

def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other. Make sure to describe the gripper and its interactions with objects in the scene."
    # user_prompt = "Briefly describe the objects in this scene."]
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt

results_json = {}
for idx, episode in tqdm(enumerate(ds), total=len(ds), desc=f"Generating descriptions [{start}%:{end}%]"):
    if "episode_id" not in episode["episode_metadata"].keys():
        episode_id = idx
    else:
        episode_id = episode["episode_metadata"]["episode_id"].numpy()
        if isinstance(episode_id, bytes):
            episode_id = episode_id.decode()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    
    # Initialize episode entry if it doesn't exist
    if file_path not in results_json.keys():
        results_json[file_path] = {}
    
    if str(episode_id) not in results_json[file_path]:
        results_json[file_path][str(episode_id)] = {
            "episode_id": str(episode_id),
            "file_path": file_path,
            "steps": []
        }
    
    for step_idx, step in enumerate(episode["steps"]):
        lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())

        user_prompt = create_user_prompt(lang_instruction)
        prompt_builder = vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        torch.manual_seed(0)
        caption = vlm.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=64,
            min_length=1,
        )
        
        # Store step info
        step_data = {
            "step_idx": step_idx,
            "caption": caption,
        }
        results_json[file_path][str(episode_id)]["steps"].append(step_data)
        
        # Save after each step to preserve progress
        with open(results_json_path, "w") as f:
            json.dump(results_json, f, indent=2, cls=NumpyFloatValuesEncoder)