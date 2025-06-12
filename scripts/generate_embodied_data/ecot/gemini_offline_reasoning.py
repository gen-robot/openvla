import json
import os
import re
import time
import tqdm

import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode
from scripts.generate_embodied_data.online_annotator import OnlineAnnotator

import tensorflow as tf
# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")


def build_prompt(features, language_instruction, step_index=None, caption=None, list_only_moves=False):
    # Read the VLM prompt2 template
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts/vlm_prompt2.md")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    # Create structured features representation
    structured_features = "{\n"

    keys = list(features.keys())
    # Remove image from keys if present, as we'll handle images separately
    if "image" in keys:
        keys.remove("image")

    # If step_index is specified, only include that step's features
    if step_index is not None:
        if list_only_moves:
            structured_features = structured_features + f'    0: "{features["move_primitive"][step_index]}"\n'
        else:
            structured_features = structured_features + f'    0: {"{"}\n'

            for key in keys:
                feature_value = features[key][step_index]
                if isinstance(feature_value, str):
                    feature_value = f'"{feature_value}"'
                elif isinstance(feature_value, np.ndarray):
                    feature_value = feature_value.tolist()

                structured_features = structured_features + f'        "{key}": {feature_value},\n'

            structured_features = structured_features + "    },\n"
    else:
        # Include all steps if step_index is not specified
        for i in range(len(features[keys[0]])):
            if list_only_moves:
                structured_features = structured_features + f'    {i}: "{features["move_primitive"][i]}"\n'
            else:
                structured_features = structured_features + f'    {i}: {"{"}\n'

                for key in keys:
                    feature_value = features[key][i]
                    if isinstance(feature_value, str):
                        feature_value = f'"{feature_value}"'
                    elif isinstance(feature_value, np.ndarray):
                        feature_value = feature_value.tolist()

                    structured_features = structured_features + f'        "{key}": {feature_value},\n'

                structured_features = structured_features + "    },\n"

    structured_features = structured_features + "}"

    # Create features description based on what's included
    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "euler" represents the orientation of the end effector in Euler angles (roll, pitch, yaw),\n'
            '- "gripper_openness" indicates how open the gripper is, with higher values meaning more open,\n'
            '- "move_primitive" describes the move that is about to be executed,'
        )

    # Handle caption
    caption_text = ""
    if caption is not None:
        caption_text = f"The robot is operating in the following environment. {caption}"

    # Replace placeholders in the template
    prompt = prompt_template.replace("LANGUAGE_INSTRUCTION", language_instruction)
    prompt = prompt.replace("TRAJECTORY_FEATURES", structured_features)
    prompt = prompt.replace("FEATURES_DESCRIPTION", features_desc)
    prompt = prompt.replace("CAPTION", caption_text)
    
    return prompt


def find_task_occurrences(input_string, tags):
    # Initialize an empty list to store all matches
    all_matches = []
    # Use a regex pattern to extract each entry's index and content.
    # This pattern assumes entries are formatted as: number: "text" (optionally followed by a comma)
    pattern = r'(\d+):\s*"(.*?)"(?:,|$)'
    entries = re.findall(pattern, input_string, re.DOTALL)
    
    for entry_num, entry_content in entries:
        # Create a dictionary to store this entry's data, starting with its index.
        entry_data = {'index': entry_num}
        
        # For each tag, find all occurrences in the entry's content.
        for tag in tags:
            tag_pattern = r'<{0}>(.*?)</{0}>'.format(tag)
            matches = re.findall(tag_pattern, entry_content, re.DOTALL)
            if matches:
                # If only one occurrence was found, store it as a string.
                # Otherwise, store a list of all occurrences (each stripped of leading/trailing whitespace).
                if len(matches) == 1:
                    entry_data[tag] = matches[0].strip()
                else:
                    entry_data[tag] = [match.strip() for match in matches]
            else:
                entry_data[tag] = None
        
        all_matches.append(entry_data)
    
    return all_matches


def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason",
                                                   "relevant_objects", "primitive_actions", "action_reason", )):
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    matches = find_task_occurrences(reasoning_output, tags)

    for match in matches:
        # First element is the step number, rest are the tag contents
        step_num = int(match['index'])
        tag_contents = {tag: match[tag] for tag in tags if match[tag] is not None}
        trajectory[step_num] = tag_contents

    return trajectory


def get_reasoning_dict(features, metadata, lm, step_index, logging_name=None):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    # Build prompt for this specific step
    prompt = build_prompt(features, language_instruction, step_index=step_index, caption=caption, list_only_moves=False)
    
    # Get the image for this step
    if "images" in features and step_index < len(features["images"]):
        image = features["images"][step_index]
    else:
        print(f"Warning: No image available for step {step_index}")
        image = None
    
    print(f"Processing step {step_index} - metadata:", metadata, "\nprompt:", prompt)

    # add a time bar to log the time taken to generate the reasoning
    with tqdm.tqdm(total=100, desc=f"Generating reasoning for {logging_name} step {step_index}") as pbar:
        max_attempts = 3
        reasoning_output = None
        for attempt in range(max_attempts):
            try:
                # Generate with image if available
                if image is not None:
                    reasoning_output = lm.generate_with_image(prompt, image)
                else:
                    reasoning_output = lm.generate(prompt)
                    
                if reasoning_output is not None:
                    break
            except Exception as e:
                print(f"Attempt {attempt+1}/{max_attempts} failed with error: {str(e)}. Retrying...")
                time.sleep(2)  # Add a small delay before retrying
                
        pbar.update(100)
        
    if reasoning_output is None:
        print("reasoning output is None.")
        import pdb; pdb.set_trace()
    print(f"Step {step_index} reasoning:", reasoning_output)

    # Extract JSON data from the response
    try:
        # Look for JSON data in the response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, reasoning_output, re.DOTALL)
        
        if json_match:
            json_text = json_match.group(1)
            reasoning_data = json.loads(json_text)
            return reasoning_data
        else:
            # Fall back to the original extraction method if no JSON is found
            return extract_reasoning_dict(reasoning_output)
    except json.JSONDecodeError:
        # If JSON parsing fails, fall back to the original extraction method
        return extract_reasoning_dict(reasoning_output)


def build_single_reasoning(episode_index, builder, lm):
    ds = builder.as_dataset(split=f"train[{episode_index}:{episode_index + 1}]")
    episode = next(iter(ds))
    total_episode_num = builder.info.splits["train"].num_examples

    ft = dict()

    # Collect images for all steps
    ft["images"] = [step["observation"]["image"] for step in episode["steps"]]
    ft["state_3d"] = [list(step["observation"]["state"][:3].numpy()) for step in episode["steps"]]
    ft["euler"] = [list(step["observation"]["state"][3:6].numpy()) for step in episode["steps"]]
    ft["gripper_openness"] = [step["observation"]["state"][-1].numpy() for step in episode["steps"]]

    move_primitives = get_move_primitives_episode(episode)
    ft["move_primitive"] = [move[0] for move in move_primitives]

    mt = {
        "episode_id": episode["episode_metadata"]["episode_id"].numpy(),
        "file_path": str(episode["episode_metadata"]["file_path"].numpy())[2:-1],
        "n_steps": len(episode["steps"]),
        "language_instruction": str(next(iter(episode["steps"]))["language_instruction"].numpy().decode()),
    }

    if isinstance(mt["episode_id"], bytes):
        mt["episode_id"] = mt["episode_id"].decode()
    
    # Generate reasoning for each step
    logging_name = f"Episode {episode_index} / {total_episode_num}"
    all_reasonings = {}
    
    # Process each step individually
    for step_idx in range(len(episode["steps"])):
        step_reasoning = get_reasoning_dict(ft, mt, lm, step_idx, f"{logging_name}")
        all_reasonings[step_idx] = step_reasoning
    
    entry = {"reasoning": all_reasonings, "features": ft, "metadata": mt}
    return entry


def jsonify(data):
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {key: jsonify(value) for key, value in data.items()}
    if isinstance(data, list):
        return [jsonify(item) for item in data]
    if isinstance(data, tuple):
        return [jsonify(item) for item in data]
    return data


def generate_reasonings(builder, episode_indexes, save_path="reasonings.json"):
    reasonings = dict()
    lm = OnlineAnnotator("gemini-2.5-pro-preview-03-25")

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    for i in episode_indexes:
        entry = build_single_reasoning(i, builder, lm)

        if entry["metadata"]["file_path"] in reasonings.keys():
            reasonings[entry["metadata"]["file_path"]][entry["metadata"]["episode_id"]] = entry
        else:
            reasonings[entry["metadata"]["file_path"]] = {entry["metadata"]["episode_id"]: entry}

        print("computed reasoning:", entry)

        with open(save_path, "w") as out_f:
            json.dump(jsonify(reasonings), out_f)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default="cobot_rlds")
    parser.add_argument("--data_dir", type=str, default="datasets")
    args = parser.parse_args()

    result_dir = f"./outputs/{args.dataset_name}"

    builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)
    total_num_episodes = builder.info.splits["train"].num_examples
    print("num_episodes in dataset:", total_num_episodes)

    def get_id_range(id, splits, total_num_episodes):
        split_percents = 100 // splits
        start = id * split_percents
        end = (id + 1) * split_percents
        start_episode_id = int(total_num_episodes * start / 100)
        end_episode_id = int(total_num_episodes * end / 100)
        if id == splits - 1:  # Last split should include the final episode
            end_episode_id = total_num_episodes
        return start_episode_id, end_episode_id
    
    # Check if all episodes will be covered by the splits
    all_episodes = set(range(total_num_episodes))
    covered_episodes = set()
    
    for id in range(args.splits):
        start_id, end_id = get_id_range(id, args.splits, total_num_episodes)
        episodes_in_split = set(range(start_id, end_id))
        covered_episodes.update(episodes_in_split)
        print(f"Split {id}: Episodes {start_id} to {end_id-1} ({len(episodes_in_split)} episodes)")
    
    missing_episodes = all_episodes - covered_episodes
    if missing_episodes:
        print(f"WARNING: {len(missing_episodes)} episodes will not be processed by any split!")
        print(f"Missing episodes: {sorted(missing_episodes)}")
    else:
        print(f"All {total_num_episodes} episodes will be covered by the splits.")
    
    # Get the range for the current split
    start_episode_id, end_episode_id = get_id_range(args.id, args.splits, total_num_episodes)
    print(f"This process (ID {args.id}) will handle episodes {start_episode_id} to {end_episode_id-1}")
    
    episode_indexes = list(range(start_episode_id, end_episode_id))

    save_path = os.path.join(result_dir, f"reasonings/reasonings_{args.id}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    generate_reasonings(builder, episode_indexes, save_path=save_path)

