import argparse
import json
import os
import time
import warnings
import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import re
from utils import NumpyFloatValuesEncoder
import cv2

# Import Gemini
from google import genai
from google.genai import types

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore")

# System prompts
TASK_OBJECT_EXTRACTION_SYSTEM_INSTRUCTIONS = """
You are an expert at analyzing robotic manipulation instructions. Your task is to extract all object names mentioned in the instruction that are relevant for the task.

Return a JSON array of object names in the following format:
["object1", "object2", "object3", ...]

Guidelines:
1. Only include objects explicitly mentioned in the instruction
2. Be specific about object parts when they are mentioned (e.g., "drawer handle" vs "drawer")
3. Include objects that are targets of actions or tool objects used for actions
4. Never include general categories unless specifically mentioned

Never return code fencing or additional text.
"""

OBJECT_LIST_WITH_TASK_OBJECTS_SYSTEM_INSTRUCTIONS = """
You are an expert vision system analyzing robotic arm observations. Your task is to list ALL visible objects in the scene that could be relevant for robotic manipulation tasks.

Return a JSON array of object names in the following format:
["object1", "object2", "object3", ...]

Guidelines:
1. Be as specific as possible when naming objects (e.g., "plastic cup" instead of just "cup")
2. ALWAYS include all task-relevant objects provided in the prompt
3. Include manipulatable parts as separate objects (e.g., "drawer handle" in addition to "drawer")
4. Always include the robotic gripper in your list
5. If you see multiple instances of the same object type, list them with distinguishing characteristics (e.g., "red cup", "blue cup")

Never return code fencing or additional text. Limit to 25 objects maximum.
"""

OBJECT_DEDUPLICATION_SYSTEM_INSTRUCTIONS = """
You are an expert at standardizing object names for robotic manipulation tasks. Your task is to analyze a list of object names and standardize them for consistency.

Return a JSON array of standardized object names in the following format:
["object1", "object2", "object3", ...]

Guidelines:
1. ALWAYS include all task-relevant objects provided in the prompt - these are mandatory
2. Identify and merge duplicate objects with slightly different names (keep the more detailed/specific name)
3. Ensure consistent naming for the same object type (e.g., standardize "cup" vs "mug")
4. Preserve important descriptive details (color, material, size, position)
5. Separate different instances of the same object with clear distinguishing characteristics
6. Prioritize names that describe manipulatable parts when applicable

Never return code fencing or additional text.
"""

# Updated from generate_bboxes_gemini.py
BBOX_FORMAT_INSTRUCTIONS = """
[
  {
    "label": "object_name",
    "box": [y_min, x_min, y_max, x_max],
    "confidence": confidence_score
  },
  ... more objects ...
]

Where:
- [y_min, x_min, y_max, x_max] are normalized coordinates in the 0-1000 range
- (0,0) is the top-left corner and (1000,1000) is the bottom-right corner
- confidence_score is a value between 0 and 1, which is how confident the model is that the bounding box and object name are correct
- object_name is a descriptive label for the object
"""

# Updated from generate_bboxes_gemini.py
BBOX_GUIDELINES = """
IMPORTANT GUIDELINES:
1. The box coordinates [y_min, x_min, y_max, x_max] must be normalized to a 0-1000 range, where:
   - (0,0) is the top-left corner of the image
   - (1000,1000) is the bottom-right corner of the image
2. Be as specific as possible when labeling objects (e.g., "plastic cup" instead of just "cup")
3. Detect manipulatable parts rather than just larger objects (e.g., "drawer handle" rather than just "drawer")
4. Always include the gripper in your detections
5. If the gripper is grasping an object, ensure both bounding boxes are properly detected
6. Pay attention to spatial relationships between objects and the gripper
"""

# New from generate_bboxes_gemini.py
BOUNDING_BOX_SYSTEM_INSTRUCTIONS = f"""
Return bounding boxes as a JSON array with the EXACT following format:
{BBOX_FORMAT_INSTRUCTIONS}

Never return masks or code fencing. Limit to 25 objects.
If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc.).

Always end your response with the word "FINISHED" on a new line to indicate you've completed the detection.
"""

# Add system prompt for descriptions
SCENE_DESCRIPTION_SYSTEM_INSTRUCTIONS = """
You are an expert vision system specialized in describing robotic scenes for embodied AI tasks. Your descriptions are concise, factual, and focus on:
1. Objects visible in the scene and their spatial relationships
2. The current state of the robotic gripper and any objects it's interacting with
3. The arrangement of objects relative to each other
4. Key environmental features relevant to potential robotic manipulation

Your descriptions should be 2-4 sentences long and purely objective, without speculation or instructions.
Never return code fencing or additional text.
"""

def resize_pos(pos, img_size):
    # return [(x * size) // 256 for x, size in zip(pos, img_size)]
    return pos

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]

def draw_bboxes(img, bboxes, img_size):
    for name, bbox in bboxes.items():
        show_name = name

        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return img


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--splits", default=4, type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", 
                     help="Gemini model to use, options: gemini-1.5-flash-latest, gemini-2.0-flash-lite, gemini-2.0-flash, gemini-2.5-pro-exp-03-25, gemini-2.5-pro-preview-03-25")
    parser.add_argument("--bbox_model", type=str, default="gemini-2.5-pro-exp-03-25",
                     help="Gemini model to use for bounding box detection")
    parser.add_argument("--visualize", default=True, action="store_true", help="Generate visualization images of bounding boxes on first frame")
    parser.add_argument("--output_img_dir", type=str, help="Directory to save visualization images (defaults to result_path/images/)")
    return parser.parse_args()


def initialize_gemini_client(api_key, model_id):
    """Initialize and return the Gemini client"""
    client = genai.Client(api_key=api_key)
    print(f"Using Gemini model: {model_id}")
    return client


def get_safety_settings():
    """Return safety settings for Gemini API"""
    return [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]


def parse_json_response(text_response):
    """Parse JSON array from Gemini response text"""
    # Try to find JSON array using regex
    json_pattern = r'\[(?:[^[\]]*|\[(?:[^[\]]*|\[[^[\]]*\])*\])*\]'
    json_match = re.search(json_pattern, text_response.replace('\n', ' '))
    
    if json_match:
        try:
            objects_json = json.loads(json_match.group(0))
            if isinstance(objects_json, list):
                return objects_json
        except json.JSONDecodeError:
            # Try to fix common JSON errors
            try:
                potential_json = json_match.group(0)
                corrected_json = re.sub(r'}\s*{', '},{', potential_json)
                objects_json = json.loads(corrected_json)
                if isinstance(objects_json, list):
                    return objects_json
            except Exception as e:
                print(f"Second parsing attempt failed: {str(e)}")
    
    # If no objects were found
    print(f"Warning: Could not parse JSON from response: {text_response[:100]}...")
    return []


def extract_task_relevant_objects(client, model_id, instruction, safety_settings):
    """Extract task-relevant object names from language instruction"""
    if not instruction or len(instruction.strip()) == 0:
        return []
        
    try:
        # Create prompt for extracting objects from instruction
        prompt = f"""
        Extract all object names mentioned in this robotic manipulation instruction:
        "{instruction}"
        
        Return ONLY a JSON array of object names.
        """
        
        # Call Gemini API
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=TASK_OBJECT_EXTRACTION_SYSTEM_INSTRUCTIONS,
                temperature=0.1,
                safety_settings=safety_settings,
            )
        )
        
        return parse_json_response(response.text)
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return []


# Updated function to match generate_bboxes_gemini.py style
def create_prompt_for_image(lang_instruction, task_objects):
    """Create a prompt based on language instruction for object detection"""
    # Format the task objects list for the prompt
    task_objects_str = ", ".join([f'"{obj}"' for obj in task_objects])
    
    base_prompt = f"""
    Detect the 2d bounding boxes of all visible objects in the image, including the robotic gripper.
    
    Return ONLY a JSON array with objects in the following format:
    {BBOX_FORMAT_INSTRUCTIONS}
    
    {BBOX_GUIDELINES}
    """
    
    # Add task context if available
    if lang_instruction and len(lang_instruction.strip()) > 0:
        # Clean up instruction
        lang_instruction = lang_instruction.strip()
        if lang_instruction.endswith("."):
            lang_instruction = lang_instruction[:-1]
            
        # Create task-focused prompt with stronger emphasis on task objects
        return f"""You're an expert vision system analyzing a robotic arm observation to complete this task: '{lang_instruction}'.

        Detect the 2d bounding boxes of all visible objects in the scene, with these requirements:
        1. The robotic gripper MUST be detected
        2. ALL task-relevant objects listed below MUST be detected and included in your output, even if partially visible or occluded
        3. Any objects the gripper is currently interacting with
        4. Manipulatable parts of objects (handles, buttons, etc.)
        5. Any obstacles or objects in the environment that might affect task completion
        
        REQUIRED OBJECTS THAT MUST BE INCLUDED IN YOUR OUTPUT (even if not clearly visible):
        [{task_objects_str}]
        
        These task-relevant objects are critical for the task: '{lang_instruction}'
        
        If you don't see one of these required objects clearly, still include it in your output with your best estimate of its location and a lower confidence score.
        
        Return ONLY a JSON array with objects in the following format:
        {BBOX_FORMAT_INSTRUCTIONS}
        
        {BBOX_GUIDELINES}
        """
    
    # Add additional focus for the base prompt with stronger emphasis
    return base_prompt + f"\nFocus especially on tools, objects, robotic grippers, and any manipulatable parts.\n\nIf any of these objects are present, they MUST be included in your output: [{task_objects_str}]"


def generate_object_list(client, model_id, image, prompt, system_instruction, safety_settings):
    """Generate a list of objects in the image using Gemini"""
    try:
        # Call Gemini API with image
        response = client.models.generate_content(
            model=model_id,
            contents=[
                image,
                prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                safety_settings=safety_settings,
            )
        )
        
        return parse_json_response(response.text)
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return []


# Updated detect_bounding_boxes function to match the one in generate_bboxes_gemini.py
def detect_bounding_boxes(client, model_id, image, prompt, safety_settings):
    """Detect bounding boxes for objects in the image using Gemini"""
    try:
        # Resize image to reasonable dimensions for the API
        img_resized = image.copy()
        original_width, original_height = image.size
        img_resized.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Call Gemini API
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, img_resized],
            config=types.GenerateContentConfig(
                system_instruction=BOUNDING_BOX_SYSTEM_INSTRUCTIONS,
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        
        # Parse response
        try:
            text_response = response.text
            
            # Try several approaches to extract valid JSON
            formatted_bboxes = []
            
            # Approach 1: Try to find JSON using regex with proper nested bracket handling
            # This regex tries to find a JSON array with proper bracket nesting
            json_pattern = r'\[(?:[^[\]]*|\[(?:[^[\]]*|\[[^[\]]*\])*\])*\]'
            json_match = re.search(json_pattern, text_response.replace('\n', ' '))
            
            if json_match:
                try:
                    bboxes_json = json.loads(json_match.group(0))
                    # Process valid JSON results
                    for item in bboxes_json:
                        if "box" in item and "label" in item:
                            # Extract box coordinates
                            box = item["box"]
                            assert isinstance(box, list) and len(box) == 4
                            y1, x1, y2, x2 = box
                                
                            # Denormalize coordinates to pixel values
                            y1_px = int(y1 * original_height / 1000)
                            x1_px = int(x1 * original_width / 1000)
                            y2_px = int(y2 * original_height / 1000)
                            x2_px = int(x2 * original_width / 1000)
                                
                            # Get confidence if available or use default
                            conf = item.get("confidence", 0.95)
                            
                            # Store both normalized and denormalized coordinates
                            formatted_bboxes.append((conf, item["label"], 
                                                    [x1_px, y1_px, x2_px, y2_px],  # denormalized
                                                    [x1, y1, x2, y2]))  # normalized
                            
                except json.JSONDecodeError:
                    # Approach 2: Try to fix common JSON errors before parsing
                    try:
                        potential_json = json_match.group(0)
                        # Replace common errors like missing commas between objects
                        corrected_json = re.sub(r'}\s*{', '},{', potential_json)
                        # Try parsing the corrected JSON
                        bboxes_json = json.loads(corrected_json)
                        # Process the JSON as before (same code as above)
                        for item in bboxes_json:
                            if "box" in item and "label" in item:
                                box = item["box"]
                                assert isinstance(box, list) and len(box) == 4
                                y1, x1, y2, x2 = box
                                
                                # Denormalize coordinates to pixel values
                                y1_px = int(y1 * original_height / 1000)
                                x1_px = int(x1 * original_width / 1000)
                                y2_px = int(y2 * original_height / 1000)
                                x2_px = int(x2 * original_width / 1000)
                                
                                # Get confidence if available or use default
                                conf = item.get("confidence", 0.95)
                                
                                # Store both normalized and denormalized coordinates
                                formatted_bboxes.append((conf, item["label"], 
                                                      [x1_px, y1_px, x2_px, y2_px],  # denormalized
                                                      [x1, y1, x2, y2]))  # normalized
                                
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Second parsing attempt failed: {str(e)}")
            
            # If no bboxes were found after all attempts
            if not formatted_bboxes:
                print(f"Warning: Could not parse bounding box JSON from response. Response starts with: {text_response[:100]}...")
                    
            return formatted_bboxes
        except Exception as e:
            print(f"Error parsing Gemini response: {str(e)}")
            return []
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return []


# Updated visualization function to match generate_bboxes_gemini.py
def visualize_bboxes(image, bboxes, frame_idx=None, instruction=None):
    """Draw bounding boxes with labels and confidence scores on the image"""
    # Create a copy of the image to avoid modifying original
    vis_img = image.copy()
    
    # Calculate dimensions for the combined image with legend
    img_height, img_width = vis_img.shape[:2]
    legend_width = 300  # Width of the legend area
    combined_width = img_width + legend_width
    
    # Create a new canvas with white background for the combined image
    combined_img = np.ones((img_height, combined_width, 3), dtype=np.uint8) * 255
    
    # Copy the original image to the left side
    combined_img[:, :img_width] = vis_img
    
    # Draw bounding boxes with index numbers
    legend_text = []
    for i, (confidence, label, bbox, _) in enumerate(bboxes, 1):  # Updated to handle the normalized coordinates
        # Convert bbox to dictionary format for draw_bboxes
        index_label = f"[{i}]"
        bbox_dict = {index_label: bbox}
        
        # Draw the numbered bounding box
        img_size = vis_img.shape[:2]
        vis_img = draw_bboxes(vis_img, bbox_dict, img_size=img_size)
        
        # Add to legend text
        legend_text.append(f"[{i}] {label}: {confidence:.2f}")
    
    # Copy the updated image with bounding boxes to combined image
    combined_img[:, :img_width] = vis_img
    
    # Add frame number if provided
    if frame_idx is not None:
        cv2.putText(combined_img, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add task instruction if provided
    if instruction:
        y_instruction = 10
        cv2.putText(combined_img, "Task:", (img_width + 10, y_instruction), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y_instruction += 10
        
        # Split instruction into multiple lines
        words = instruction.split()
        line = ""
        for word in words:
            if len(line) + len(word) < 40:
                line += " " + word if line else word
            else:
                cv2.putText(combined_img, line, (img_width + 15, y_instruction), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                y_instruction += 10
                line = word
                
        # Add the last line
        if line:
            cv2.putText(combined_img, line, (img_width + 15, y_instruction), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            y_instruction += 10  # Extra space after instruction
        
        # Start legend text after instruction
        y_offset = y_instruction + 10
        cv2.putText(combined_img, "Objects:", (img_width + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y_offset += 10
    else:
        # Start legend text from the top if no instruction
        y_offset = 10
    
    # Add legend text with smaller font
    for text in legend_text:
        # Split long text into multiple lines if needed
        words = text.split()
        line = words[0]
        for word in words[1:]:
            if len(line) + len(word) < 40:
                line += " " + word
            else:
                cv2.putText(combined_img, line, (img_width + 10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                y_offset += 10  # Reduced line spacing
                line = "    " + word  # Indent continuation lines
        
        cv2.putText(combined_img, line, (img_width + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        y_offset += 10  # Reduced space between different objects
    
    return combined_img


def deduplicate_objects_with_image_context(client, model_id, image, all_objects, instruction, safety_settings, task_objects=None):
    """Deduplicate object names with image context"""
    # Create prompt for deduplicating objects using image context
    object_list_str = "\n".join([f"- {obj}" for obj in all_objects])
    
    # Create task objects string with clear marking
    task_objects_str = ""
    if task_objects and len(task_objects) > 0:
        task_objects_str = "\n\nTASK-RELEVANT OBJECTS (MUST BE INCLUDED):\n" + "\n".join([f"- {obj}" for obj in task_objects])
    
    dedup_prompt = f"""
    Here is a list of object names that may contain duplicates or inconsistencies:
    
    {object_list_str}{task_objects_str}
    
    Examine the image and standardize these object names by:
    1. ALWAYS including ALL task-relevant objects listed above - these are mandatory
    2. Merging duplicate objects with slightly different names (keep the more detailed name)
    3. Ensuring consistent naming for the same object type
    4. Preserving important descriptive details (color, material, size, position)
    5. Using the image to confirm which objects are actually present
    
    Return ONLY a JSON array of standardized object names that you can see in the image.
    Always include ALL task-relevant objects even if not clearly visible in the image.
    """
    
    # Deduplicate object names with image context
    summarized_objects = []
    for attempt in range(3):
        try:
            # Call Gemini API with image
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    image,
                    dedup_prompt
                ],
                config=types.GenerateContentConfig(
                    system_instruction=OBJECT_DEDUPLICATION_SYSTEM_INSTRUCTIONS,
                    temperature=0.1,
                    safety_settings=safety_settings,
                )
            )
            
            objects_json = parse_json_response(response.text)
            if len(objects_json) > 0:
                # Ensure task objects are included
                if task_objects:
                    for task_obj in task_objects:
                        if not any(task_obj.lower() in obj.lower() for obj in objects_json):
                            objects_json.append(task_obj)
                return objects_json
                
        except Exception as e:
            print(f"Deduplication attempt {attempt+1} failed: {str(e)}")
    
    # Fallback to simple deduplication
    print("Failed to deduplicate objects with image context. Using simple deduplication with task objects.")
    # Make sure to include task objects in the fallback
    if task_objects:
        return list(set(all_objects + task_objects))
    else:
        return list(set(all_objects))


# Updated to use new create_prompt_for_image function
def create_bbox_prompt(instruction, summarized_objects):
    """Create a prompt for bounding box detection"""
    return create_prompt_for_image(instruction, summarized_objects)


def collect_file_paths_and_task_objects(ds, client, model_id, safety_settings):
    """First pass: collect file paths, instructions, and task-relevant objects"""
    file_paths_data = {}

    for episode in tqdm.tqdm(ds, desc="Collecting file paths and instructions"):
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        episode_id = episode["episode_metadata"]["episode_id"].numpy().decode() if isinstance(episode["episode_metadata"]["episode_id"].numpy(), bytes) else str(episode["episode_metadata"]["episode_id"].numpy())
        
        # Extract language instruction from first step
        lang_instruction = ""
        current_step = next(iter(episode["steps"]), None)
        if current_step and "language_instruction" in current_step:
            lang_instruction = current_step["language_instruction"].numpy().decode()

        print(f"Processing file path: {file_path} with episode id: {episode_id}")
        print(f"Language instruction: {lang_instruction}")
        
        # If we already have this file path, just update episodes list
        if file_path in file_paths_data:
            file_paths_data[file_path]["episodes"].append({
                "episode_id": episode_id,
                "instruction": lang_instruction
            })
            
            # Add the new instruction to the list
            if lang_instruction and lang_instruction not in file_paths_data[file_path]["instructions"]:
                file_paths_data[file_path]["instructions"].append(lang_instruction)
                
                # Extract and add task-relevant objects from this instruction
                if lang_instruction:
                    new_task_objects = extract_task_relevant_objects(client, model_id, lang_instruction, safety_settings)
                    # Merge with existing task objects (avoid duplicates)
                    file_paths_data[file_path]["task_objects"] = list(set(file_paths_data[file_path]["task_objects"] + new_task_objects))
        else:
            # Extract task-relevant objects from instruction
            task_objects = extract_task_relevant_objects(client, model_id, lang_instruction, safety_settings)
            
            # Store file path, instruction, task objects, and first frame
            file_paths_data[file_path] = {
                "task_objects": task_objects,
                "instructions": [lang_instruction],
                "first_frame": next(iter(episode["steps"]))["observation"]["image"].numpy(),
                "episodes": [{
                    "episode_id": episode["episode_metadata"]["episode_id"].numpy().decode() if isinstance(episode["episode_metadata"]["episode_id"].numpy(), bytes) else str(episode["episode_metadata"]["episode_id"].numpy()),
                    "instruction": lang_instruction
                }],
                "object_lists": []  # Will store object lists from multiple episodes with the same file_path
            }

    print(f"Found {len(file_paths_data)} unique file paths")
    return file_paths_data


def generate_object_lists_for_files(file_paths_data, client, model_id, safety_settings):
    """Second pass: generate object lists for each unique file path"""
    for file_path, data in tqdm.tqdm(file_paths_data.items(), desc="Generating object lists"):
        task_objects = data["task_objects"]
        instructions = data["instructions"]
        first_frame = data["first_frame"]
        
        # Use the first instruction for the prompt
        instruction = instructions[0] if instructions else ""
        
        # Generate simple object list prompt without bbox requirements
        task_objects_str = ", ".join([f'"{obj}"' for obj in task_objects])
        object_prompt = f"""
        List all visible objects in this image that could be relevant for robotic manipulation.
        
        Task instruction: "{instruction}"
        
        Task-relevant objects mentioned in the instruction: [{task_objects_str}]
        
        Please ensure you include:
        1. The robotic gripper
        2. All task-relevant objects
        3. Any manipulatable parts of objects (handles, buttons, etc.)
        4. All objects the gripper might interact with
        
        Return ONLY a JSON array of object names.
        """
        
        image = Image.fromarray(first_frame)
        
        # Generate object list with multiple attempts
        object_list = []
        for attempt in range(3):
            object_list = generate_object_list(client, model_id, image, object_prompt, 
                                              OBJECT_LIST_WITH_TASK_OBJECTS_SYSTEM_INSTRUCTIONS, 
                                              safety_settings)
            if len(object_list) > 0:
                break
            print(f"Attempt {attempt+1} failed to generate object list for {file_path}. Retrying...")
        
        if len(object_list) == 0:
            print(f"Failed to generate object list for {file_path} after 3 attempts.")
            # Include at least the task objects and gripper as fallback
            object_list = task_objects.copy()
            if "robotic gripper" not in object_list:
                object_list.append("robotic gripper")
        
        # Store the object list
        data["object_lists"].append(object_list)

    return file_paths_data


def generate_scene_description(client, model_id, image, instruction, object_list, safety_settings):
    """Generate a concise scene description using Gemini that incorporates detected objects"""
    # Format the object list for the prompt
    object_list_str = ", ".join([f'"{obj}"' for obj in object_list])
    
    # Create prompt for generating a scene description
    object_context = f"\nThe following objects have been detected in the scene: [{object_list_str}]." if object_list else ""
    
    if instruction and len(instruction.strip()) > 0:
        # Clean up instruction
        instruction = instruction.strip()
        if instruction.endswith("."):
            instruction = instruction[:-1]
            
        description_prompt = f"""
        The robot task is: '{instruction}'
        
        Briefly describe this scene, including the objects present and their spatial relationships.{object_context}
        
        Focus on:
        1. The robotic gripper and any objects it's interacting with
        2. The arrangement of objects relative to each other
        3. Key environmental features relevant to the task
        
        Keep your description objective, concise (2-4 sentences), and focused on what's visible.
        """
    else:
        description_prompt = f"""
        Briefly describe this scene, including the objects present and their spatial relationships.{object_context}
        
        Focus on:
        1. The robotic gripper and any objects it's interacting with
        2. The arrangement of objects relative to each other
        3. Notable objects or features in the environment
        
        Keep your description objective, concise (2-4 sentences), and focused on what's visible.
        """
    
    try:
        # Call Gemini API with image
        response = client.models.generate_content(
            model=model_id,
            contents=[
                image,
                description_prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction=SCENE_DESCRIPTION_SYSTEM_INSTRUCTIONS,
                temperature=0.4,
                safety_settings=safety_settings,
            )
        )
        
        # Extract text response
        return response.text
            
    except Exception as e:
        print(f"Error generating scene description: {str(e)}")
        return "No description available."


def process_bounding_boxes(file_paths_data, client, model_id, bbox_model_id, safety_settings, visualize=False, img_dir=None):
    """Third pass: detect bounding boxes, deduplicate, and generate descriptions"""
    for file_path, data in tqdm.tqdm(file_paths_data.items(), desc="Detecting bounding boxes"):
        # Combine all object lists including task objects
        all_object_lists = data["object_lists"]
        if data["task_objects"]:
            all_object_lists.append(data["task_objects"])
        
        # Flatten all object lists
        all_objects = [obj for sublist in all_object_lists for obj in sublist]
        
        # If the list is very small, no need for deduplication
        if len(all_objects) <= 5:
            # Make sure to include task objects
            summarized_objects = list(set(all_objects + data["task_objects"]))
        else:
            # Deduplicate object names with image context
            first_frame = data["first_frame"]
            image = Image.fromarray(first_frame)
            instruction = data["instructions"][0] if data["instructions"] else ""
            
            summarized_objects = deduplicate_objects_with_image_context(
                client, model_id, image, all_objects, instruction, safety_settings,
                task_objects=data["task_objects"]  # Pass task objects explicitly
            )
        
        # Store the summarized objects
        data["summarized_objects"] = summarized_objects
        
        # Create prompt for bounding box detection
        instruction = data["instructions"][0] if data["instructions"] else ""
        bbox_prompt = create_bbox_prompt(instruction, summarized_objects)
        
        # Detect bounding boxes
        image = Image.fromarray(data["first_frame"])
        bboxes = []
        for attempt in range(3):
            bboxes = detect_bounding_boxes(client, bbox_model_id, image, bbox_prompt, safety_settings)
            if len(bboxes) > 0:
                break
            print(f"Attempt {attempt+1} failed to generate bounding boxes for {file_path}. Retrying...")
        
        if len(bboxes) == 0:
            print(f"Failed to generate bounding boxes for {file_path} after 3 attempts.")
        
        # Store the bounding boxes
        data["bboxes"] = bboxes
        
        # Generate scene description
        description = generate_scene_description(
            client, model_id, image, instruction, summarized_objects, safety_settings
        )
        
        # Store the description
        data["description"] = description
        print(f"Generated description: {description[:100]}...")
        
        # Visualize bounding boxes if enabled
        if visualize and len(bboxes) > 0 and img_dir:
            img_array = np.array(image)
            instruction = data["instructions"][0] if data["instructions"] else ""
            vis_frame = visualize_bboxes(img_array, bboxes, instruction=instruction)
            
            # Create a clean file path for saving
            safe_path = file_path.replace('/', '_').replace('\\', '_').replace(':', '_')
            img_save_path = os.path.join(img_dir, f"{safe_path}_first_frame.jpg")
            cv2.imwrite(img_save_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization image to {img_save_path}")

    return file_paths_data


def organize_final_results(file_paths_data):
    """Organize results into the required format"""
    object_results_json = {}

    for file_path, data in file_paths_data.items():
        # Initialize file_path entry
        object_results_json[file_path] = {}
        
        # Add entry for each episode
        for idx, episode_info in enumerate(data["episodes"]):
            episode_id = episode_info["episode_id"]
            
            # Convert bboxes from tuple format to dict format for storage
            formatted_bboxes = []
            for confidence, label, bbox_denorm, bbox_norm in data["bboxes"]:
                x1, y1, x2, y2 = bbox_denorm
                nx1, ny1, nx2, ny2 = bbox_norm
                formatted_bboxes.append({
                    "label": label,
                    "box": [x1, y1, x2, y2],  # denormalized (pixel) coordinates
                    "normalized_box": [nx1, ny1, nx2, ny2],  # normalized coordinates (0-1000)
                    "confidence": confidence
                })
            
            # Extract labels from bounding boxes to create the object list
            bbox_labels = [bbox["label"] for bbox in formatted_bboxes]
            
            object_results_json[file_path][str(episode_id)] = {
                "object_list": bbox_labels,
                "task_relevant_objects": data["task_objects"],
                "first_frame_bboxes": formatted_bboxes,
                "description": data.get("description", ""),  # Include generated description
                "episode_id": episode_id,
                "file_path": file_path
            }

    return object_results_json


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    result_path = f"./outputs/{args.dataset_name}/object_lists_gemini"
    os.makedirs(result_path, exist_ok=True)
    object_list_json_path = os.path.join(result_path, f"results_object_lists_{args.id}.json")
    description_json_path = os.path.join(result_path, f"results_descriptions_{args.id}.json")
    
    # Create image directory for visualizations
    img_dir = args.output_img_dir if args.output_img_dir else f"{result_path}/images/{args.id}"
    os.makedirs(img_dir, exist_ok=True)

    # Load dataset
    print("Loading data...")
    split_percents = 100 // args.splits
    start = args.id * split_percents
    end = (args.id + 1) * split_percents

    ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=f"train[{start}%:{end}%]")
    print("Done.")

    # Initialize Gemini client
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    client = initialize_gemini_client(GOOGLE_API_KEY, args.model)
    safety_settings = get_safety_settings()

    # Process data in stages
    # First pass: collect file paths and task-relevant objects
    print("First pass: collecting file paths and extracting task-relevant objects...")
    file_paths_data = collect_file_paths_and_task_objects(ds, client, args.model, safety_settings)

    # Save intermediate results of task-relevant objects
    with open(os.path.join(result_path, f"task_relevant_objects_{args.id}.json"), "w") as f:
        task_objects_data = {file_path: data["task_objects"] for file_path, data in file_paths_data.items()}
        json.dump(task_objects_data, f, indent=2, cls=NumpyFloatValuesEncoder)

    # Second pass: generate object lists
    print("Generating object lists using task-relevant objects...")
    file_paths_data = generate_object_lists_for_files(file_paths_data, client, args.model, safety_settings)

    # Save intermediate results of generated object lists
    with open(os.path.join(result_path, f"generated_object_lists_{args.id}.json"), "w") as f:
        object_lists_data = {file_path: data["object_lists"] for file_path, data in file_paths_data.items()}
        json.dump(object_lists_data, f, indent=2, cls=NumpyFloatValuesEncoder)

    # Third pass: detect bounding boxes, deduplicate, and generate descriptions
    print("Detecting bounding boxes, deduplicating object names, and generating descriptions...")
    file_paths_data = process_bounding_boxes(file_paths_data, client, args.model, args.bbox_model, safety_settings, 
                                           args.visualize, img_dir)

    # Save intermediate results of summarized objects with bounding boxes
    with open(os.path.join(result_path, f"summarized_objects_with_bboxes_{args.id}.json"), "w") as f:
        summarized_data = {
            file_path: {
                "summarized_objects": data["summarized_objects"],
                "bboxes": data["bboxes"],
                "description": data.get("description", "")
            } 
            for file_path, data in file_paths_data.items()
        }
        json.dump(summarized_data, f, indent=2, cls=NumpyFloatValuesEncoder)
        
    # Save descriptions separately (similar format to generate_descriptions.py)
    description_results = {}
    for file_path, data in file_paths_data.items():
        description_results[file_path] = {}
        for episode_info in data["episodes"]:
            episode_id = episode_info["episode_id"]
            description_results[file_path][str(episode_id)] = {
                "episode_id": str(episode_id),
                "file_path": file_path,
                "caption": data.get("description", "")
            }
    
    with open(description_json_path, "w") as f:
        json.dump(description_results, f, indent=2, cls=NumpyFloatValuesEncoder)

    # Final step: organize and save results
    print("Organizing final results...")
    object_results_json = organize_final_results(file_paths_data)

    # Save final results (object lists, bboxes, and descriptions in a single file)
    with open(object_list_json_path, "w") as f:
        json.dump(object_results_json, f, indent=2, cls=NumpyFloatValuesEncoder)

    print(f"Completed! Generated object lists, bounding boxes, and descriptions for {len(object_results_json)} unique file paths.")
    print(f"Descriptions saved to {description_json_path}")


if __name__ == "__main__":
    main()
