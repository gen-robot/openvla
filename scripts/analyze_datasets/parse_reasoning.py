import json

def print_dict_structure(d, indent=0, max_depth=None, current_depth=0):
    """
    Print the structure of a nested dictionary, showing keys and value types.
    
    Args:
        d: The dictionary to analyze
        indent: Current indentation level (default: 0)
        max_depth: Maximum depth to traverse (default: None, meaning no limit)
        current_depth: Current depth in the traversal (default: 0)
    """
    if max_depth is not None and current_depth >= max_depth:
        print(" " * indent + "...")
        return
    
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}: dict")
            print_dict_structure(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list):
            if len(value) > 0:
                print(" " * indent + f"{key}: list of {type(value[0]).__name__}")
                if isinstance(value[0], dict) and (max_depth is None or current_depth < max_depth):
                    print(" " * (indent + 4) + "Sample item:")
                    print_dict_structure(value[0], indent + 8, max_depth, current_depth + 1)
            else:
                print(" " * indent + f"{key}: empty list")
        else:
            print(" " * indent + f"{key}: {type(value).__name__}")

reasoning_data_path = "/nvme_data/embodied_agent/oxe_data/rlds/reasonings_dataset.json"

with open(reasoning_data_path, "r") as f:
    reasoning_data = json.load(f)

f_out = open("reasoning_datasets.csv", "w")
example_out = open("example_reasoning.json", "w")

# title line: file_path, episode_id, episode_length
f_out.write("file_path, episode_id, episode_length, language_instruction, has_reasoning, reasoning_length, has_gripper_pos, has_bboxes, reasoning_content\n")
from tqdm import tqdm

total_files = len(reasoning_data.keys())
for file_name in tqdm(reasoning_data.keys(), desc="Processing files"):
    episodes = reasoning_data[file_name].keys()
    for episode_id in tqdm(episodes, desc=f"Processing episodes in {file_name}", leave=False):
        if example_out is not None:
            example_out.write(f"{file_name}, {episode_id}, {reasoning_data[file_name][episode_id]}\n")
            example_out.close()
            example_out = None
        import pdb; pdb.set_trace()
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
            import pdb; pdb.set_trace()
            reasoning_length = len(reasoning_data[file_name][episode_id]["reasoning"])
            reasoning_content = reasoning_data[file_name][episode_id]["reasoning"]['0'].keys()
        else:
            reasoning_length = 0
            reasoning_content = []


        f_out.write(f"{file_name}, {episode_id}, {episode_length}, {language_instruction}, {has_reasoning}, {reasoning_length}, {has_gripper_pos}, {has_bboxes}, {reasoning_content}\n")

f_out.close()
