import json
import os
import re
import time
import tqdm

import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode


class Gemini:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", None)
        assert api_key is not None, "GEMINI_API_KEY is not set"
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def safe_call(self, f):
        while True:
            try:
                res = f()
                return res
            except ResourceExhausted:
                time.sleep(5)

    def generate(self, prompt, images=None):
        chat = self.safe_call(lambda: self.model.start_chat(history=[]))
        
        # If images are provided, send them along with the prompt
        if images:
            # For Gemini, you need to format the images properly
            # This might vary depending on the specific VLM API
            content = [prompt]
            for image in tqdm.tqdm(images, desc="Processing images"):
                # Convert numpy array to the format expected by the API
                # This might be base64 encoding or another format
                # For Gemini, you can use the following:
                content.append({"mime_type": "image/jpeg", "data": self._process_image(image)})

            response = self.safe_call(lambda: chat.send_message(content).text)
        else:
            # Text-only prompt
            response = self.safe_call(lambda: chat.send_message(prompt).text)

        for i in range(8):
            if response is None:
                print(f"n_retries: {i}")
                return None
            if "FINISHED" in response:
                print(f"n_retries: {i}")
                return response
            else:
                print("FINISHED not found in response")
            response = response + self.safe_call(lambda: chat.send_message("Truncated, please continue.").text)

        print(f"n_retries: {i}")

        return None
    
    def _process_image(self, image_array):
        """
        Process a numpy array image for the VLM API.
        For Gemini, you need to convert the image to bytes.
        """
        import io
        from PIL import Image
        
        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            # Normalize if needed
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        
        # Ensure the image has the right shape (H, W, 3) for RGB
        if len(image_array.shape) == 2:  # Grayscale
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[-1] == 1:  # Single channel
            image_array = np.concatenate([image_array] * 3, axis=-1)
        elif image_array.shape[-1] == 4:  # RGBA
            image_array = image_array[:, :, :3]  # Drop alpha channel
            
        pil_image = Image.fromarray(image_array)
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        
        return image_bytes


def build_prompt(features, language_instruction, caption=None, list_only_moves=False):

    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    break_line = ""  # for line formatting

    # NOTE: Simplified prompt to only request gripper position
    return f"""# Annotate the training trajectory with gripper positions

## Task Context

You are analyzing a trajectory of a robotic arm performing a task specified by the instruction: "{language_instruction}". You will be given a sequence of images showing the robot's execution.

{caption}## Your objective

For each step (corresponding to each image provided), identify the 2D pixel coordinates of the gripper's center point (e.g., the midpoint between the fingertips).

**Output Format:**
Write the answer as a Python-executable dictionary. For every step in the trajectory, there should be exactly one item of the form `<step_id>: "<gripper_position_tag>"`. The final dictionary should have exactly the same set of integer keys as the number of images provided (e.g., if 10 images are provided, the keys should be 0, 1, ..., 9).

**Gripper Position Tag Content:**
Each value string MUST contain the following information encapsulated in an XML-like tag:

1.  **Gripper Position:** Detect the current gripper position (e.g., the center point between the fingertips) in the image. Provide the 2D pixel coordinates `[x, y]`. Output this information in a tag `<gripper_position>` as a Python list string: `[x, y]`. If and only if the gripper is not clearly visible, provide `None`.

**Example Output String for one step (using pixel coordinates):**
```python
# Example for step 5:
5: "<gripper_position>[318, 150]</gripper_position>"
```
```python
# Example for step 6 (gripper not visible):
6: "<gripper_position>[100, 100]</gripper_position>"
```

Ensure that the coordinates are precise. Provide the complete dictionary containing entries for all steps.

At the very end of the entire response, write a single label `FINISHED` to indicate that the answer is complete.
"""


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


def extract_reasoning_dict(reasoning_output, tags=("gripper_position",)):
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    matches = find_task_occurrences(reasoning_output, tags)

    for match in matches:
        # First element is the step number, rest are the tag contents
        step_num = int(match['index'])
        tag_contents = {tag: match[tag] for tag in tags if match[tag] is not None}

        # Attempt to parse the gripper position string into a list or None
        if 'gripper_position' in tag_contents:
            pos_str = tag_contents['gripper_position']
            try:
                # Handle potential 'None' string or list string
                if pos_str.strip().lower() == 'none':
                    tag_contents['gripper_position'] = None
                else:
                    # Use ast.literal_eval for safe evaluation of list string
                    import ast
                    parsed_pos = ast.literal_eval(pos_str)
                    if isinstance(parsed_pos, list) and len(parsed_pos) == 2 and all(isinstance(c, (int, float)) for c in parsed_pos):
                         tag_contents['gripper_position'] = [int(c) for c in parsed_pos] # Convert to int
                    else:
                         # Invalid format, treat as None or keep original string? Let's default to None.
                         print(f"Warning: Step {step_num}: Could not parse gripper_position '{pos_str}' as [x, y]. Setting to None.")
                         tag_contents['gripper_position'] = None
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Warning: Step {step_num}: Error parsing gripper_position '{pos_str}': {e}. Setting to None.")
                tag_contents['gripper_position'] = None

        trajectory[step_num] = tag_contents

    return trajectory


def get_reasoning_dict(features, metadata, lm, logging_name=None):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    prompt = build_prompt(features, language_instruction, caption=caption, list_only_moves=True)
    print("metadata:", metadata, "\nprompt:", prompt)

    # Extract images if they exist in features
    images = features.get("image", None)

    # write the prompt to a file
    with open("prompt.txt", "w") as f:
        f.write(prompt)

    # add a time bar to log the time taken to generate the reasoning
    with tqdm.tqdm(total=100, desc=f"Generating reasoning for {logging_name}") as pbar:
        reasoning_output = lm.generate(prompt, images=images)
        pbar.update(100)
    # import pdb; pdb.set_trace()
    print("reasoning:", reasoning_output)
    import pdb; pdb.set_trace()

    # save reasoning output to file
    # with open("reasoning_output.txt", "r") as f:
    #     # f.write(reasoning_output)
    #     # read the content into a string
    #     reasoning_output = ""
    #     for line in f:
    #         reasoning_output += line

    return extract_reasoning_dict(reasoning_output)


def build_single_reasoning(episode_id, builder, lm, captions):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    total_episode_num = builder.info.splits["train"].num_examples

    ft = dict()
    ft["image"] = [step["observation"]["image"].numpy() for step in episode["steps"]]
    ft["state_3d"] = [list(step["observation"]["state"][:3].numpy()) for step in episode["steps"]]
    # ft["euler"] = [list(step["observation"]["state"][3:6].numpy()) for step in episode["steps"]]
    # ft["gripper_openness"] = [step["observation"]["state"][-1].numpy() for step in episode["steps"]]

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

    logging_name = f"Episode {episode_id} / {total_episode_num}"
    reasoning = get_reasoning_dict(ft, mt, lm, logging_name)
    entry = {"reasoning": reasoning, "features": {k: v for k, v in ft.items() if k != "image"}, "metadata": mt}

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


def generate_reasonings(builder, episode_ids, save_path="reasonings.json"):
    reasonings = dict()
    lm = Gemini()

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    with open("captions.json", "r") as captions_file:
        captions_dict = json.load(captions_file)

    for i in episode_ids:
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
    parser.add_argument("--episode_id", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="reasonings.json")
    parser.add_argument("--dataset_name", type=str, default="cobot_rlds")
    parser.add_argument("--data_dir", type=str, default="datasets")
    args = parser.parse_args()

    with open("captions.json", "r") as captions_file:
        captions_dict = json.load(captions_file)

    num_episodes = sum([len(v) for v in captions_dict.values()])
    print("num_episodes:", num_episodes)

    if args.episode_id is None:
        episode_ids = range(num_episodes)
    else:
        episode_ids = [args.episode_id]

    builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)
    generate_reasonings(builder, episode_ids, save_path=args.save_path)

