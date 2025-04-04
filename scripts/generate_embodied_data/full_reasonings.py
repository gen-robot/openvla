import json
import os
import re
import time
import tqdm

import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode
from scripts.generate_embodied_data.utils import Gemini

import tensorflow as tf
# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")


def build_prompt(features, language_instruction, caption=None, list_only_moves=False):
    structured_features = "{\n"

    keys = list(features.keys())

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

    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    break_line = ""  # for line formatting

    return f"""# Annotate the training trajectory with reasoning

## Specification of the experimental setup

You're an expert reinforcement learning researcher. You've trained an optimal policy for controlling a robotic arm. The
robot successfully completed a task specified by the instruction: "{language_instruction}". For that purpose, the
robotic arm executed a sequence of actions. Consecutive moves that were executed are the following:


```python
trajectory_features = {structured_features}
```

{features_desc}

{caption}## Your objective

I want you to annotate the given trajectory with reasoning. That is, for each step, I need to know not only {
break_line}which action should be chosen, but importantly what reasoning justifies that action choice. I want you to {
break_line}be descriptive and include all the relevant information available. The reasoning should include the task {
break_line}to complete, the remaining high-level steps, the high-level movements that should be executed and why they {
break_line}are required, the premises that allow inferring the direction of each move, including the locations of {
break_line}relevant objects, possible obstacles or difficulties to avoid, and any other relevant justification.

### Begin by describing the task

Start by giving an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, {
break_line}the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe {
break_line}the high-level movements that were most likely executed, based on the task that was completed and the {
break_line}primitive movements that were executed. Then, for each high-level movement write the interval of steps that {
break_line}movement consists of. Also, for each high-level movement write a justification for why it should be {
break_line}executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight {
break_line}all the relevant details, but ensure that your description is consistent with the trajectory that was {
break_line}executed, specified by the features listed above in the `trajectory_features` dictionary.

### List the reasonings for each step

Finally, for each step describe the reasoning that allows to determine the correct action. For each step describe the {
break_line}remaining part of the objective, the current progress, the objects that are still relevant for determining {
break_line}the plan, and the plan for the next steps, based on the available features. Start the reasoning from a high {
break_line}level and gradually add finer features. I need you to be descriptive and very precise. Ensure that the {
break_line}reasoning is consistent with the task and the executed trajectory. Write the answer for this part as a {
break_line}Python-executable dictionary. For every step in the initial trajectory there should be exactly one separate {
break_line}item of the form <step id>:<reasoning>. Do not group the answers. The final dictionary should have exactly {
break_line}the same set of integer keys as the dictionary of features provided in the `trajectory_features` dictionary {
break_line}above. The reasoning should be a single string that describes the reasoning in natural language and {
break_line}includes all the required features.

Each reasoning string should have the following form:
- Describe the full task that remains to be completed (but only describe what remains), and place it inside a {
break_line}tag <task>.
- Describe the complete high-level plan for completing the remaining task (the list of remaining high-level steps), {
break_line}and place it inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it {
break_line}inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence {
break_line}that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Identify and describe the key objects that are relevant for the current subtask, and place them in a list of object names inside a tag <relevant_objects>.
- Describe the current primitive movement of the arm that needs to be executed, and place it inside a tag <move>.
- Describe why the chosen movement should be executed now and which features of the current environment influence that {
break_line}decision. Place it inside a tag <move_reason>.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- For each step on the trajectory:
  1. Describe the reasoning that leads to determining the correct action
  2. Identify the relevant objects for the current subtask
  3. Provide justification for the movement
- The reasoning should be descriptive and precise. You should provide exactly one reasoning string for each step on the {
break_line}trajectory specified by `trajectory_features`.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""


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


def get_reasoning_dict(features, metadata, lm, logging_name=None):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    prompt = build_prompt(features, language_instruction, caption=caption, list_only_moves=False)
    print("metadata:", metadata, "\nprompt:", prompt)

    # add a time bar to log the time taken to generate the reasoning
    with tqdm.tqdm(total=100, desc=f"Generating reasoning for {logging_name}") as pbar:
        max_attempts = 3
        reasoning_output = None
        for attempt in range(max_attempts):
            reasoning_output = lm.generate(prompt)
            if reasoning_output is not None:
                break
            print(f"Attempt {attempt+1}/{max_attempts} failed. Retrying...")
        pbar.update(100)
    if reasoning_output is None:
        print("reasoning output is None.")
        import pdb; pdb.set_trace()
    print("reasoning:", reasoning_output)

    # save reasoning output to file
    # with open("reasoning_output.txt", "r") as f:
    #     # f.write(reasoning_output)
    #     # read the content into a string
    #     reasoning_output = ""
    #     for line in f:
    #         reasoning_output += line

    return extract_reasoning_dict(reasoning_output)


def build_single_reasoning(episode_index, builder, lm, captions):
    ds = builder.as_dataset(split=f"train[{episode_index}:{episode_index + 1}]")
    episode = next(iter(ds))
    total_episode_num = builder.info.splits["train"].num_examples

    ft = dict()

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
    mt["caption"] = captions[mt["file_path"]][mt["episode_id"]]["caption"]

    logging_name = f"Episode {episode_index} / {total_episode_num}"
    reasoning = get_reasoning_dict(ft, mt, lm, logging_name)
    entry = {"reasoning": reasoning, "features": ft, "metadata": mt}

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


def generate_reasonings(builder, episode_indexes, captions_dict, save_path="reasonings.json"):
    reasonings = dict()
    lm = Gemini()

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    # with open("captions.json", "r") as captions_file:
    #     captions_dict = json.load(captions_file)

    for i in episode_indexes:
        entry = build_single_reasoning(i, builder, lm, captions_dict)

        if entry["metadata"]["file_path"] in reasonings.keys():
            reasonings[entry["metadata"]["file_path"]][entry["metadata"]["episode_id"]] = entry
            # reasonings[entry["metadata"]["file_path"]][i] = entry
        else:
            reasonings[entry["metadata"]["file_path"]] = {entry["metadata"]["episode_id"]: entry}
            # reasonings[entry["metadata"]["file_path"]] = {i: entry}

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

    with open(os.path.join(result_dir, "captions.json"), "r") as captions_file:
        captions_dict = json.load(captions_file)

    num_episodes = sum([len(v) for v in captions_dict.values()])
    print("num_episodes in captions:", num_episodes)

    builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)
    total_num_episodes = builder.info.splits["train"].num_examples
    print("num_episodes in dataset:", total_num_episodes)

    if num_episodes != total_num_episodes:
        print("[WARNING] num_episodes in captions and dataset are not the same")

    def get_id_range(id, splits, total_num_episodes):
        split_percents = 100 // splits
        start = id * split_percents
        end = (id + 1) * split_percents
        start_episode_id = int(total_num_episodes * start / 100)
        end_episode_id = int(total_num_episodes * end / 100)
        if id == splits - 1:  # Last split should include the final episode
            end_episode_id = total_num_episodes
        return start_episode_id, end_episode_id
    
    # run over id to check if no id is ignored
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

    generate_reasonings(builder, episode_indexes, captions_dict, save_path=save_path)

