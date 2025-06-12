import json
import os
import argparse
from utils import NumpyFloatValuesEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--results_path", type=str, required=True)
args = parser.parse_args()

# Merge object lists
full_obj_json = {}
obj_count = 0

path_to_object_lists = os.path.join(args.results_path, "object_lists_gemini")
path_to_full_obj = os.path.join(path_to_object_lists, "full_object_lists.json")
os.makedirs(path_to_object_lists, exist_ok=True)

for json_f in os.listdir(path_to_object_lists):
    if not json_f.startswith("results_object_lists_") or not json_f.endswith(".json"):
        continue
    with open(os.path.join(path_to_object_lists, json_f), "r") as f:
        obj_json = json.load(f)

    for file_path, file_json in obj_json.items():
        if file_path not in full_obj_json.keys():
            full_obj_json[file_path] = {}
        for ep_id, obj_dict in file_json.items():
            assert ep_id not in full_obj_json[file_path].keys(), f"Duplicate episode ID {ep_id} found in {file_path}"
            full_obj_json[file_path][ep_id] = obj_dict
            obj_count += 1

print("Inserted", obj_count, "object lists into combined json")
print("Saving to:", path_to_full_obj)

with open(path_to_full_obj, "w") as f:
    json.dump(full_obj_json, f, indent=2, cls=NumpyFloatValuesEncoder)
