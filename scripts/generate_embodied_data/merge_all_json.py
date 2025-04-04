"""
file_path:
    episode_id:
        features:
            move_primitive: list (N) of str (move primitive)
            gripper_position: list (N) of [int, int] (gripper position, in pixels, [x, y])
            bboxes: list (N) of list (K objects) of [float, str, list of int] (bounding boxes, in pixels, [confidence, object_name, [x1, y1, x2, y2]])
        metadata:
            episode_id: episode id
            file_path: file path
            n_steps: number of steps
            language_instruction: language instruction
        reasoning:
            {0: {task: str, plan: str, subtask: str, subtask_reason: str, move: str, move_reason: str}}
            ...
"""

import json
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, required=True)
args = parser.parse_args()

results_path = args.results_path

# Find all split reasoning, gripper, and bboxes files
reasoning_files = glob.glob(os.path.join(results_path, "reasonings", "reasonings*.json"))
gripper_files = glob.glob(os.path.join(results_path, "gripper_positions", "gripper_positions*.json"))
bboxes_files = glob.glob(os.path.join(results_path, "bboxes", "results_*.json"))

# If no split files found, use the original file paths
if not reasoning_files:
    assert os.path.exists(os.path.join(results_path, "full_reasonings.json")), "Reasoning file not found"
    reasoning_files = [os.path.join(results_path, "reasonings", "full_reasonings.json")]
if not gripper_files:
    assert os.path.exists(os.path.join(results_path, "gripper_positions", "full_gripper_positions.json")), "Gripper file not found"
    gripper_files = [os.path.join(results_path, "gripper_positions", "full_gripper_positions.json")]
if not bboxes_files:
    assert os.path.exists(os.path.join(results_path, "bboxes", "full_bboxes.json")), "Bboxes file not found"
    bboxes_files = [os.path.join(results_path, "bboxes", "full_bboxes.json")]

print(f"Found {len(reasoning_files)} reasoning files, {len(gripper_files)} gripper files, and {len(bboxes_files)} bboxes files")

# Merge reasoning files
reasoning_count = 0
reasoning_json = {}
for file_path in reasoning_files:
    print(f"Reading reasoning file: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        for file_path_key, episodes in data.items():
            if file_path_key not in reasoning_json:
                reasoning_json[file_path_key] = {}
            for episode_id_key, episode in episodes.items():
                assert episode_id_key not in reasoning_json[file_path_key], f"Episode {episode_id_key} already exists in {file_path_key}"
                reasoning_json[file_path_key][episode_id_key] = episode
                reasoning_count += 1

print(f"Merged {reasoning_count} episodes from {len(reasoning_files)} reasoning files")

# Merge gripper files
gripper_count = 0
gripper_json = {}
for file_path in gripper_files:
    print(f"Reading gripper file: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        for file_path_key, episodes in data.items():
            if file_path_key not in gripper_json:
                gripper_json[file_path_key] = {}
            for episode_id_key, episode in episodes.items():
                assert episode_id_key not in gripper_json[file_path_key], f"Episode {episode_id_key} already exists in {file_path_key}"
                gripper_json[file_path_key][episode_id_key] = episode
                gripper_count += 1

print(f"Merged {gripper_count} episodes from {len(gripper_files)} gripper files")

# Merge bboxes files
bboxes_count = 0
bboxes_json = {}
for file_path in bboxes_files:
    print(f"Reading bboxes file: {file_path}")
    with open(file_path, "r") as f:
        bbox_json = json.load(f)
        for file_name, file_name_json in bbox_json.items():
            if file_name not in bboxes_json:
                bboxes_json[file_name] = {}
            for _, ep_json in file_name_json.items():
                ep_id = ep_json["episode_id"]
                if ep_id in bboxes_json[file_name]:
                    print(ep_id, ep_json)
                    print(bboxes_json[file_name][ep_id])
                    raise ValueError(f"Duplicate episode id {ep_id} in {file_name}")
                bboxes_json[file_name][ep_id] = ep_json
                bboxes_count += 1

print(f"Merged {bboxes_count} episodes from {len(bboxes_files)} bboxes files")

# save merged json
with open(os.path.join(results_path, "reasonings", "full_reasonings.json"), "w") as f:
    json.dump(reasoning_json, f, indent=4)
with open(os.path.join(results_path, "gripper_positions", "full_gripper_positions.json"), "w") as f:
    json.dump(gripper_json, f, indent=4)
with open(os.path.join(results_path, "bboxes", "full_bboxes.json"), "w") as f:
    json.dump(bboxes_json, f, indent=4)

no_merged_keys = []
merged_json = {}
full_count = 0
for file_path in reasoning_json.keys():
    for episode_id in reasoning_json[file_path].keys():
        flag = False
        if (
            file_path not in gripper_json.keys()
            or episode_id not in gripper_json[file_path].keys()
            or file_path not in bboxes_json.keys()
            or episode_id not in bboxes_json[file_path].keys()
        ):
            logging_str = f"File path {file_path} or episode id {episode_id} not found in gripper json or bboxes json."
            no_merged_keys.append(logging_str)
            flag = True

        if flag:
            continue

        reasoning = reasoning_json[file_path][episode_id]
        gripper = gripper_json[file_path][episode_id]
        bboxes = bboxes_json[file_path][episode_id]

        if file_path not in merged_json.keys():
            merged_json[file_path] = {}
        merged_json[file_path][episode_id] = {
            "features": {
                "move_primitive": reasoning["features"]["move_primitive"],
                "gripper_position": gripper["gripper_positions"],
                "bboxes": bboxes["bboxes"],
            },
            "metadata": {
                "episode_id": episode_id,
                "file_path": file_path,
            },
            "reasoning": reasoning["reasoning"],
        }
        full_count += 1

print(f"Merged {full_count} episodes with {len(no_merged_keys)} episodes not merged")

with open(os.path.join(results_path, "merged_json.json"), "w") as f:
    json.dump(merged_json, f, indent=4)
