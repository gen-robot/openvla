import json
import os
import re
import time
import tqdm

import numpy as np
# import google.generativeai as genai
from google import genai
from google.api_core.exceptions import ResourceExhausted
# from google.generativeai import types
from google.genai import types

from scripts.generate_embodied_data.primitive_movements import get_move_primitives_episode
from scripts.generate_embodied_data.utils import Gemini

import tensorflow as tf
# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

# Add constants for system instructions
REASONING_SYSTEM_INSTRUCTIONS = """
You are an expert reinforcement learning researcher analyzing robotic trajectories. 
Your task is to provide detailed reasoning for each step in a robot's execution path.

Your analysis should be thorough, precise, and grounded in the observable data. 
Focus on explaining the robot's decision-making process based on its current state, 
the task requirements, and the physical environment.

IMPORTANT: In your responses, do not reference the internal feature names such as 
state_3d, euler, gripper_openness, or move_primitive. Instead, use natural language 
descriptions of the robot's position, orientation, gripper state, and movements.

Format your response according to the tags requested in the prompt.
"""

def get_safety_settings():
    """Return safety settings for Gemini API"""
    return [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]

# Add a Gemini client initialization function
def initialize_gemini_client(api_key, model_id="gemini-2.0-flash"):
    """Initialize and return the Gemini client"""
    client = genai.Client(api_key=api_key)
    print(f"Using Gemini model: {model_id}")
    return client

def build_prompt(features, language_instruction, caption=None, list_only_moves=False, task_objects=None):
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

    # Add task-relevant objects section if provided
    task_objects_section = ""
    if task_objects and len(task_objects) > 0:
        task_objects_str = ", ".join([f'"{obj}"' for obj in task_objects])
        task_objects_section = f"""## Task-relevant objects

The following objects are relevant for this task: [{task_objects_str}]

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

{caption}{task_objects_section}## Your objective

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
- Describe the full task to be completed and use consistent words with the task instruction, and place it inside a {
break_line}tag <task>.
- Describe the complete high-level plan for completing the full task (the list of high-level steps). Number each step {
break_line}with "1./2./3." format and separate with dots (Example: "1. Pick up the cup. 2. Move to the table"). {
break_line}Place this inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it {
break_line}inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence {
break_line}that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Identify and describe the key objects that are relevant for the current subtask, and place them as a comma-separated {
break_line}list of object names without square brackets (Example: "cup, table, plate") inside a tag <relevant_objects>.
- Include ONLY the raw move primitive string itself inside the <move> tag, without any description or additional text. {
break_line}For example: <move>move forward down</move> or <move>close gripper</move>. Important: Do not use "stop" as a {
break_line}movement - this can cause the robot to lock up during execution. If you see a "stop" in the trajectory, {
break_line}interpret it as a pause before the next meaningful action.
- Describe why the chosen movement should be executed now and which features of the current environment influence that {
break_line}decision. Don't just explain the move - analyze the current state and environment, then explain what {
break_line}motivates this specific movement choice. Use qualitative descriptions instead of detailed numerical values - {
break_line}for example, say "The gripper is positioned to the left of the cup, so it needs to move right to align with it" or {
break_line}"The gripper is too high above the object, so it needs to move down to reach a proper grasping position". {
break_line}Place it inside a tag <move_reason>.

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
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete.
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

def build_step_prompt(features, step_idx, window_size, language_instruction, caption=None, task_objects=None):
    """
    Build a prompt for generating reasoning for a single step with context window.
    
    Args:
        features: Dictionary containing trajectory features
        step_idx: Index of the current step to focus on
        window_size: Number of steps to include before and after the current step
        language_instruction: The natural language instruction for the task
        caption: Optional caption describing the scene
        task_objects: Optional list of task-relevant objects
        
    Returns:
        A prompt string for the LLM to generate reasoning for the specific step
    """
    # Calculate window boundaries
    total_steps = len(features["move_primitive"])
    start_idx = max(0, step_idx - window_size)
    end_idx = min(total_steps, step_idx + window_size + 1)  # Include context after the current step
    
    # Create a subset of features for the window
    windowed_features = {key: features[key][start_idx:end_idx] for key in features if key in ["move_primitive", "state_3d", "euler", "gripper_openness"]}
    
    # Find the relative position of the current step in the window
    current_relative_idx = step_idx - start_idx
    
    # Build a structured representation of the features in the window
    structured_features = "{\n"
    keys = list(windowed_features.keys())
    
    for i in range(len(windowed_features[keys[0]])):
        # Add a marker to highlight the current step
        step_marker = " (CURRENT STEP) " if i == current_relative_idx else ""
        
        structured_features = structured_features + f'    {start_idx + i}{step_marker}: {"{"}\n'
        
        for key in keys:
            feature_value = windowed_features[key][i]
            if isinstance(feature_value, str):
                feature_value = f'"{feature_value}"'
            elif isinstance(feature_value, np.ndarray):
                feature_value = feature_value.tolist()
            
            structured_features = structured_features + f'        "{key}": {feature_value},\n'
        
        structured_features = structured_features + "    },\n"
    
    structured_features = structured_features + "}"
    
    # Create features description
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
    
    # Process caption and task objects
    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    task_objects_section = ""
    if task_objects and len(task_objects) > 0:
        task_objects_str = ", ".join([f'"{obj}"' for obj in task_objects])
        task_objects_section = f"""## Task-relevant objects

The following objects are relevant for this task: [{task_objects_str}]

"""

    break_line = ""  # for line formatting
    
    return f"""# Analyze a single step in the robot trajectory

## Step-specific analysis

You're focusing on step {step_idx} of a robot trajectory (marked as CURRENT STEP in the data below). The robot is executing a task specified by the instruction: "{language_instruction}". 

I've provided a window of context including some steps before and after the current step to help you understand the full sequence. You can use this FULL CONTEXT to understand the trajectory, but your reasoning must be written as if you only have access to the CURRENT AND PREVIOUS information.

```python
trajectory_window = {structured_features}
```

{features_desc}

{caption}{task_objects_section}## Your objective

I want you to analyze the current step {step_idx} with detailed reasoning. 

IMPORTANT REQUIREMENT: While you have access to the full trajectory data (including future steps) to understand the complete context, your reasoning output MUST be written as if you only have access to the current observation and past information. The model that will be trained on your output will NOT have access to future states during inference.

Focus on explaining:
1. What action should be executed at the current step based on what's currently observable
2. Why this action makes sense in the context of the overall task given the current state
3. How this specific movement contributes to completing the task

### Analyze the current step in the context of the full task

First, describe the overall task based on the language instruction and visual scene. Then focus specifically on step {step_idx}, explaining what the robot is trying to accomplish at this moment and why this particular primitive movement makes sense given what the robot can observe right now.

### Provide structured reasoning for the current step

For step {step_idx}, the reasoning string should have the following form:
- Describe the complete task to be accomplished and place it inside a <task> tag
- Describe the full high-level plan with numbered steps and place it inside a <plan> tag
- Explain why a specific high-level step should be executed now based on what can be observed at the current moment (place within <subtask_reason> tag)
- Identify the specific high-level step that should be executed now and place it inside a <subtask> tag
- List the objects relevant to the current subtask as visible in the current image (place in <relevant_objects> tag)
- Explain why a specific movement makes sense right now based ONLY on what can be observed at the current moment (place in <move_reason> tag)
- Include ONLY the raw move primitive string inside the <move> tag

CRITICAL: Your reasoning in the <subtask_reason> and <move_reason> tags MUST NOT reference future states or outcomes. It should be written as if you only have access to the current observation and state information. This is essential because the model being trained won't have access to future information during deployment.

Format your reasoning with the following structured tags:
- <task>Complete task description</task>
- <plan>Numbered high-level plan (e.g., "1. Approach the cup. 2. Grasp the cup. 3. Lift the cup.")</plan>
- <subtask_reason>Explanation of why a specific subtask needs to be executed now based on current observations</subtask_reason>
- <subtask>Current high-level step being executed</subtask>
- <relevant_objects>List of objects important for the current subtask (comma-separated)</relevant_objects>
- <move_reason>Explanation of why a specific movement is appropriate based on current observations only</move_reason>
- <move>Raw move primitive string only</move>

IMPORTANT: In your explanations, do not reference the internal feature names (state_3d, euler, gripper_openness, move_primitive). 
Instead, use natural language to describe the robot's position, orientation, gripper state, and movements. For example, 
say "The robot arm is positioned above the cup" rather than "The state_3d value shows the arm is above the cup."

Your response should be formatted as:
{step_idx}: "Your full structured reasoning with all tags here"

## Task summary

Your key responsibility is to generate reasoning that explains the robot's actions in a way that ONLY uses information available at the current moment - as if you cannot see the future steps.
"""


class StepByStepGemini:
    """A wrapper around the Gemini client that processes reasoning step by step with corresponding images."""
    
    def __init__(self, client, model_id="gemini-2.0-flash"):
        self.client = client
        self.model_id = model_id
        self.safety_settings = get_safety_settings()
        
    def generate_step_by_step(self, prompt, features, language_instruction, images, task_objects=None):
        """Generate reasoning for each step using the corresponding image."""
        step_count = len(features["move_primitive"])
        
        # Process each step with its corresponding image
        step_reasonings = {}
        window_size = 5  # Number of steps before and after to include for context
        max_retries = 5  # Maximum number of retry attempts per step
        
        for step_idx in tqdm.tqdm(range(step_count), desc="Processing individual steps"):
            # Generate step-specific prompt using the new function
            current_step_prompt = build_step_prompt(
                features=features,
                step_idx=step_idx,
                window_size=window_size,
                language_instruction=language_instruction,
                caption=None if not hasattr(prompt, 'caption') else prompt.caption,
                task_objects=task_objects
            )
        
            # Add retry loop for each step
            success = False
            for attempt in range(max_retries):
                try:
                    step_image = images[step_idx] if step_idx < len(images) else images[-1]
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[current_step_prompt, step_image],
                        config=types.GenerateContentConfig(
                            system_instruction=REASONING_SYSTEM_INSTRUCTIONS,
                            temperature=0.2,
                            safety_settings=self.safety_settings,
                        )
                    )
                    step_reasoning = response.text

                    # Check if the response contains all required tags
                    required_tags = ["<task>", "<plan>", "<subtask>", "<subtask_reason>", "<move>", "<move_reason>", "<relevant_objects>"]
                    missing_tags = [tag for tag in required_tags if tag not in step_reasoning]
                    
                    if not step_reasoning or len(step_reasoning.strip()) < 20 or missing_tags:
                        error_msg = f"Generated reasoning for step {step_idx} is missing tags: {missing_tags}" if missing_tags else f"Generated reasoning for step {step_idx} is empty or insufficient"
                        print(error_msg)
                        raise ValueError(error_msg)
                    
                    # print(f"Step {step_idx} reasoning: {step_reasoning}")
                    
                    step_reasonings[str(step_idx)] = step_reasoning
                    success = True
                    break
                except Exception as e:
                    print(f"Error generating reasoning for step {step_idx}: {e}")
                    print(f"Attempt {attempt+1}/{max_retries} failed. Retrying...")
                    time.sleep(5)
                    # Check if we're on the last retry attempt
                    if attempt == max_retries - 1:
                        import pdb; pdb.set_trace()
        
        # Format the output as a dictionary string
        output = "{\n"
        for step_idx, reasoning in step_reasonings.items():
            cleaned_reasoning = reasoning.replace('\n', ' ').replace('"', '')
            output += f'  {step_idx}: "{cleaned_reasoning}",\n'
        output += "}\n\nFINISHED"
        
        return output
    
    def _fallback_step_by_step_generation(self, features, images, task_objects=None, base_prompt=None):
        """Fallback method that processes each step individually if the full generation fails."""
        # This method is now redundant with the improved step-by-step approach
        # Keeping for compatibility, but simply calling the main method
        return self.generate_step_by_step(base_prompt, features, images, task_objects)


def get_reasoning_dict(features, metadata, client, model_id="gemini-2.0-flash", images=None, task_objects=None, logging_name=None):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    # Generate the main prompt using the existing function
    prompt = build_prompt(features, language_instruction, caption=caption, list_only_moves=False, task_objects=task_objects)
    # print("metadata:", metadata, "\nprompt:", prompt)

    # Set up safety settings
    safety_settings = get_safety_settings()

    # Add a time bar to log the time taken to generate the reasoning
    with tqdm.tqdm(total=100, desc=f"Generating reasoning for {logging_name}") as pbar:
        max_attempts = 3
        reasoning_output = None
        
        for attempt in range(max_attempts):
            try:
                # If images are provided, use step-by-step processing
                if images and len(images) > 0:
                    # Create a wrapper that processes step by step
                    step_by_step_lm = StepByStepGemini(client, model_id)
                    
                    # Generate reasoning step by step
                    reasoning_output = step_by_step_lm.generate_step_by_step(
                        prompt=prompt,
                        features=features,
                        language_instruction=language_instruction,
                        images=images,
                        task_objects=task_objects
                    )
                else:
                    # No images, use the standard prompt-based generation
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            system_instruction=REASONING_SYSTEM_INSTRUCTIONS,
                            temperature=0.2,
                            safety_settings=safety_settings,
                        )
                    )
                    reasoning_output = response.text
                
                if reasoning_output is not None:
                    break
            except Exception as e:
                print(f"Error in attempt {attempt+1}: {str(e)}")
                # import pdb; pdb.set_trace()
            
            print(f"Attempt {attempt+1}/{max_attempts} failed. Retrying...")
            time.sleep(1)  # Small delay before retrying
            
        pbar.update(100)
    
    if reasoning_output is None:
        print("reasoning output is None.")
        import pdb; pdb.set_trace()
    
    print("reasoning:", reasoning_output)
    return extract_reasoning_dict(reasoning_output)


def build_single_reasoning(episode_index, builder, client, model_id, captions, object_lists=None):
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
    
    # Get caption from the provided captions dictionary
    mt["caption"] = captions[mt["file_path"]][mt["episode_id"]]["description"]
    
    # Get task-relevant objects if provided
    task_objects = None
    if object_lists and mt["file_path"] in object_lists and mt["episode_id"] in object_lists[mt["file_path"]]:
        task_objects = object_lists[mt["file_path"]][mt["episode_id"]].get("task_relevant_objects", [])
    
    # Extract images from each step
    images = [Image.fromarray(step["observation"]["image"].numpy()) for step in episode["steps"]]

    logging_name = f"Episode {episode_index} / {total_episode_num}"
    reasoning = get_reasoning_dict(ft, mt, client, model_id, images=images, task_objects=task_objects, logging_name=logging_name)
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


def generate_reasonings(builder, episode_indexes, captions_dict, save_path="reasonings.json", object_lists_path=None, model_id="gemini-2.0-flash"):
    reasonings = dict()
    
    # Initialize Gemini client
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    client = initialize_gemini_client(GOOGLE_API_KEY, model_id)

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    # Load object lists and task-relevant objects if provided
    object_lists = None
    if object_lists_path and os.path.exists(object_lists_path):
        print(f"Loading object lists from {object_lists_path}")
        with open(object_lists_path, "r") as f:
            object_lists = json.load(f)

    for i in tqdm.tqdm(episode_indexes, desc="Generating reasonings for Episodes"):
        entry = build_single_reasoning(i, builder, client, model_id, captions_dict, object_lists)

        if entry["metadata"]["file_path"] in reasonings.keys():
            reasonings[entry["metadata"]["file_path"]][entry["metadata"]["episode_id"]] = entry
        else:
            reasonings[entry["metadata"]["file_path"]] = {entry["metadata"]["episode_id"]: entry}

        print("computed reasoning:", entry)

        with open(save_path, "w") as out_f:
            json.dump(jsonify(reasonings), out_f, indent=2)



if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from PIL import Image
 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default="cobot_rlds")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--object_lists_path", type=str, default=None, 
                       help="Path to pre-generated object lists JSON file")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                       help="Gemini model to use (e.g., gemini-2.0-flash, gemini-2.0-flash)")
    args = parser.parse_args()

    result_dir = f"./outputs/{args.dataset_name}"

    with open(os.path.join(result_dir, "object_lists_gemini/full_object_lists.json"), "r") as captions_file:
        captions_dict = json.load(captions_file)

    # Define the object lists path if not provided
    object_lists_path = args.object_lists_path
    if not object_lists_path:
        object_lists_path = os.path.join(result_dir, "object_lists_gemini/full_object_lists.json")
        if not os.path.exists(object_lists_path):
            print(f"Warning: Object lists file not found at {object_lists_path}. Continuing without task-relevant objects.")
            object_lists_path = None

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

    save_path = os.path.join(result_dir, f"reasonings_gemini/reasonings_{args.id}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    generate_reasonings(builder, episode_indexes, captions_dict, save_path=save_path, 
                        object_lists_path=object_lists_path, model_id=args.model)

