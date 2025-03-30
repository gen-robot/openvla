import json


reasoning_data_path = "/nvme_data/embodied_agent/oxe_data/rlds/reasonings_dataset.json"

with open(reasoning_data_path, "r") as f:
    reasoning_data = json.load(f)

f_out = open("reasoning_datasets.csv", "w")

# title line: file_path, episode_id, episode_length
f_out.write("file_path, episode_id, episode_length, language_instruction, has_reasoning, reasoning_length, has_gripper_pos, has_bboxes\n")
from tqdm import tqdm

total_files = len(reasoning_data.keys())
for file_name in tqdm(reasoning_data.keys(), desc="Processing files"):
    episodes = reasoning_data[file_name].keys()
    for episode_id in tqdm(episodes, desc=f"Processing episodes in {file_name}", leave=False):
        has_reasoning = "reasoning" in reasoning_data[file_name][episode_id].keys()
        episode_length = reasoning_data[file_name][episode_id]["metadata"]["n_steps"]
        language_instruction = reasoning_data[file_name][episode_id]["metadata"]["language_instruction"]
        if 'features' in reasoning_data[file_name][episode_id].keys():
            has_gripper_pos = "gripper_position" in reasoning_data[file_name][episode_id]["features"].keys()
            has_bboxes = "bboxes" in reasoning_data[file_name][episode_id]["features"].keys()
        else:
            has_gripper_pos = False
            has_bboxes = False
        if has_reasoning:
            reasoning_length = len(reasoning_data[file_name][episode_id]["reasoning"])
        else:
            reasoning_length = 0

        f_out.write(f"{file_name}, {episode_id}, {episode_length}, {language_instruction}, {has_reasoning}, {reasoning_length}, {has_gripper_pos}, {has_bboxes}\n")

f_out.close()
