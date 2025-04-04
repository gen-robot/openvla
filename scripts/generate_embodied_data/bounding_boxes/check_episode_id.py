import json

# Load the JSON file
with open('libero_object_no_noops/results_0.json', 'r') as f:
    data = json.load(f)

# Extract all episode IDs
episode_ids = []
for file_path in data:
    for episode_id in data[file_path]:
        episode_ids.append(episode_id)

# Check for duplicates
unique_ids = set(episode_ids)
if len(unique_ids) < len(episode_ids):
    print(f"Found duplicates! {len(episode_ids) - len(unique_ids)} duplicated episode IDs")
    
    # Find the specific duplicates
    from collections import Counter
    duplicates = [item for item, count in Counter(episode_ids).items() if count > 1]
    print(f"Duplicated IDs: {duplicates}")
else:
    print("No duplicates found among the episode IDs")