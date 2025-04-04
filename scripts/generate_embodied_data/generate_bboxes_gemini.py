import argparse
import json
import os
import time
import warnings
import tqdm
import numpy as np
import cv2
import mediapy

import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from io import BytesIO
from utils import NumpyFloatValuesEncoder, post_process_caption

# Import Gemini
from google import genai
from google.genai import types

from prismatic.util.img_utils import draw_bboxes, name_to_random_color, resize_pos

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int, default=0)
parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--model", type=str, default="gemini-2.0-flash", 
                   help="Gemini model to use, options: gemini-1.5-flash-latest, gemini-2.0-flash-lite, gemini-2.0-flash, gemini-2.5-pro-exp-03-25")
parser.add_argument("--visualize", action="store_true", help="Generate visualization videos of bounding boxes")

args = parser.parse_args()
result_path = f"./outputs/{args.dataset_name}/bboxes_gemini"
os.makedirs(result_path, exist_ok=True)
bbox_json_path = os.path.join(result_path, f"results_bboxes_{args.id}.json")

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

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = args.model
print(f"Using Gemini model: {MODEL_ID}")

# System instructions for bounding box detection
bounding_box_system_instructions = """
Return bounding boxes as a JSON array with the EXACT following format:
[
  {
    "label": "object_name",
    "box": [x_min, y_min, x_max, y_max],
    "confidence": confidence_score
  },
  ... more objects ...
]

Where:
- [x_min, y_min, x_max, y_max] are normalized coordinates in the 0-1000 range
- (0,0) is the top-left corner and (1000,1000) is the bottom-right corner
- confidence_score is a value between 0 and 1
- object_name is a descriptive label for the object

Never return masks or code fencing. Limit to 25 objects.
If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc.).
"""

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

def create_prompt_for_image(lang_instruction):
    """Create a prompt based on language instruction for object detection"""
    # Base prompt with format requirements
    base_format = """
    Detect the 2d bounding boxes of all objects in the image. 
    Return ONLY a JSON array with objects in the following format:
    [
      {
        "label": "object_name",
        "box": [x_min, y_min, x_max, y_max],
        "confidence": confidence_score
      },
      ... more objects ...
    ]
    
    IMPORTANT: The box coordinates [x_min, y_min, x_max, y_max] must be normalized to a 0-1000 range, where:
    - (0,0) is the top-left corner of the image
    - (1000,1000) is the bottom-right corner of the image
    """
    
    # Add task context if available
    if lang_instruction and len(lang_instruction.strip()) > 0:
        # Clean up instruction
        lang_instruction = lang_instruction.strip()
        if lang_instruction.endswith("."):
            lang_instruction = lang_instruction[:-1]
            
        # Create task-focused prompt
        task_prompt = f"""Detect the 2d bounding boxes of all visible objects, which MUST include those relevant to the task: '{lang_instruction}'. Include the gripper and the object it is moving to if visible.

        Return ONLY a JSON array with objects in the following format:
        [
          {{
            "label": "object_name",
            "box": [x_min, y_min, x_max, y_max],
            "confidence": confidence_score
          }},
          ... more objects ...
        ]
        
        IMPORTANT: The box coordinates [x_min, y_min, x_max, y_max] must be normalized to a 0-1000 range, where:
        - (0,0) is the top-left corner of the image
        - (1000,1000) is the bottom-right corner of the image
        """
        return task_prompt
    
    # Default prompt for no instruction
    return base_format + "Focus especially on tools, objects, and robot grippers."

def detect_bounding_boxes(image, prompt):
    """Use Gemini to detect bounding boxes in the image"""
    try:
        # Resize image to reasonable dimensions for the API
        img_resized = image.copy()
        original_width, original_height = image.size
        img_resized.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Call Gemini API
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, img_resized],
            config=types.GenerateContentConfig(
                system_instruction=bounding_box_system_instructions,
                temperature=0.5,
                safety_settings=safety_settings,
            )
        )
        
        # Parse response
        try:
            text_response = response.text
            
            # Try to find and parse JSON in the response
            import re
            json_match = re.search(r'\[.*\]', text_response.replace('\n', ' '))
            if json_match:
                bboxes_json = json.loads(json_match.group(0))
                
                # Ensure format compatibility with existing code
                # Convert to (confidence, label, bbox) format
                formatted_bboxes = []
                for item in bboxes_json:
                    if "box" in item and "label" in item:
                        # Extract box coordinates
                        box = item["box"]
                        if isinstance(box, list) and len(box) == 4:
                            # Convert normalized coordinates (0-1000) to actual pixel values
                            x1, y1, x2, y2 = box
                            
                            # Denormalize coordinates to pixel values
                            x1_px = int(x1 * original_width / 1000)
                            y1_px = int(y1 * original_height / 1000)
                            x2_px = int(x2 * original_width / 1000)
                            y2_px = int(y2 * original_height / 1000)
                            
                            # Get confidence if available or use default
                            conf = item.get("confidence", 0.95)
                            
                            formatted_bboxes.append((conf, item["label"], [x1_px, y1_px, x2_px, y2_px]))
                        elif isinstance(box, dict) and all(k in box for k in ["x", "y", "width", "height"]):
                            # Handle old format if returned
                            # Normalize these values as well
                            x1 = int(box["x"] * original_width / 1000)
                            y1 = int(box["y"] * original_height / 1000)
                            x2 = x1 + int(box["width"] * original_width / 1000)
                            y2 = y1 + int(box["height"] * original_height / 1000)
                            
                            # Get confidence if available or use default
                            conf = item.get("confidence", 0.95)
                            
                            formatted_bboxes.append((conf, item["label"], [x1, y1, x2, y2]))
            else:
                # No valid JSON found
                print(f"Warning: Could not parse bounding box JSON from response: {text_response[:100]}...")
                formatted_bboxes = []
                
            return formatted_bboxes
        except Exception as e:
            print(f"Error parsing Gemini response: {str(e)}")
            return []
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return []

def visualize_bboxes(image, bboxes, frame_idx=None):
    """Draw bounding boxes with labels and confidence scores on the image"""
    # Create a copy of the image to avoid modifying original
    vis_img = image.copy()
    
    # Convert bboxes from (confidence, label, bbox) format to dictionary format for draw_bboxes
    bboxes_dict = {}
    for confidence, label, bbox in bboxes:
        # Add confidence to label
        label_with_conf = f"{label}: {confidence:.2f}"
        bboxes_dict[label_with_conf] = bbox
    
    # Use draw_bboxes from prismatic.util.img_utils
    img_size = vis_img.shape[:2]
    vis_img = draw_bboxes(vis_img, bboxes_dict, img_size=img_size)
    
    # Add frame number if provided
    # if frame_idx is not None:
    #     cv2.putText(vis_img, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return vis_img


bbox_results_json = {}
for ep_idx, episode in tqdm.tqdm(
    enumerate(ds),
    total=len(ds),
    desc=f"Generating bounding boxes [{start}%:{end}%]",
):

    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    if isinstance(episode_id, bytes):
        episode_id = episode_id.decode()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"ID {args.id} starting ep: {episode_id}, {file_path}")

    if file_path not in bbox_results_json.keys():
        bbox_results_json[file_path] = {}

    start_time = time.time()
    bboxes_list = []
    
    # Initialize list to store visualization frames if enabled
    visualization_frames = [] if args.visualize else None
    
    # Extract language instruction from first step
    lang_instruction = ""
    current_step = next(iter(episode["steps"]), None)
    if current_step and "language_instruction" in current_step:
        lang_instruction = current_step["language_instruction"].numpy().decode()
    
    # Create prompt based on language instruction
    prompt = create_prompt_for_image(lang_instruction)
    print(f"Using prompt: {prompt}")
    
    for step_idx, step in tqdm.tqdm(
        enumerate(episode["steps"]),
        total=len(episode["steps"]),
        desc=f"Generating bounding boxes for Episode {episode_id}",
        leave=False,
    ):
        # Get image
        image = Image.fromarray(step["observation"]["image"].numpy())
        
        # Detect bounding boxes using Gemini
        bboxes = detect_bounding_boxes(image, prompt)
        # Add to results
        bboxes_list.append(bboxes)
        
        # Create visualization if enabled
        if args.visualize:
            # Convert PIL image to numpy for OpenCV processing
            img_array = np.array(image)
            # Draw bounding boxes on the image
            vis_frame = visualize_bboxes(img_array, bboxes, frame_idx=step_idx)
            visualization_frames.append(vis_frame)
        
        # Optional: Add sleep to avoid rate limiting
        # time.sleep(0.5)
    
    end_time = time.time()
    bbox_results_json[file_path][str(ep_idx)] = {
        "episode_id": str(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    # Save results after each episode
    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, cls=NumpyFloatValuesEncoder)
    print(f"ID {args.id} finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end_time - start_time, 2)}")

    # Save visualization video if enabled
    if args.visualize and visualization_frames:
        video_dir = f"{result_path}/videos/{args.id}"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/episode_{episode_id}.mp4"
        mediapy.write_video(video_path, visualization_frames, fps=10)
        print(f"Saved visualization video to {video_path}")

if __name__ == "__main__":
    # Create video output directory if visualization is enabled
    video_dir = None
    if args.visualize:
        video_dir = f"{result_path}/videos/{args.id}"
        os.makedirs(video_dir, exist_ok=True)
        print(f"Visualization videos will be saved to {video_dir}")
