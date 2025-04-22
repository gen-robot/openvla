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
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils import NumpyFloatValuesEncoder, post_process_caption
import glob
from prismatic.util.img_utils import draw_bboxes, name_to_random_color, resize_pos


# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int, default=0)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--visualize", default=False, action="store_true", help="Generate visualization videos of bounding boxes")
parser.add_argument("--output_video_dir", type=str, help="Directory to save visualization videos (defaults to results_path/videos/)")

args = parser.parse_args()
result_path = f"./outputs/{args.dataset_name}/bboxes_gdino_base"
os.makedirs(result_path, exist_ok=True)
bbox_json_path = os.path.join(result_path, f"results_bboxes_{args.id}.json")

print("Loading data...")
split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=f"train[{start}%:{end}%]")
print("Done.")

# Load pre-generated object lists if provided
object_lists_json = {}
first_frame_bboxes = {}
object_lists_dir = f"./outputs/{args.dataset_name}/object_lists_gemini"
if object_lists_dir:
    print(f"Loading pre-generated object lists from {object_lists_dir}...")
    # Find all JSON files in the directory
    object_lists_file = os.path.join(object_lists_dir, "full_object_lists.json")
    try:
        with open(object_lists_file, 'r') as f:
                data = json.load(f)
                # Merge with existing data
                for file_key, episodes in data.items():
                    if file_key not in object_lists_json:
                        object_lists_json[file_key] = {}
                    for ep_id, ep_data in episodes.items():
                        object_lists_json[file_key][ep_id] = ep_data
                        
                        # Store first frame bboxes separately for quick access
                        if "first_frame_bboxes" in ep_data:
                            if file_key not in first_frame_bboxes:
                                first_frame_bboxes[file_key] = {}
                            first_frame_bboxes[file_key][ep_id] = ep_data["first_frame_bboxes"]
    except Exception as e:
        print(f"Error loading {object_lists_file}: {e}")
    
    print(f"Loaded pre-generated data for {len(object_lists_json)} file paths")

if args.gpu is not None:
    device = f"cuda:{args.gpu}"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Load gDINO model
model_id = "IDEA-Research/grounding-dino-base"
print(f"Loading gDINO to device {device}...")
processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Done.")

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

def get_pre_generated_objects(file_path, episode_id):
    """
    Retrieve pre-generated object list and first frame bboxes for a specific episode.
    Returns tuple: (object_list, task_relevant_objects, first_frame_bboxes)
    """
    if not object_lists_dir or file_path not in object_lists_json:
        return None, None, None
    
    episode_id_str = str(episode_id)
    if episode_id_str not in object_lists_json[file_path]:
        return None, None, None
    
    ep_data = object_lists_json[file_path][episode_id_str]
    object_list = ep_data.get("object_list", [])
    task_relevant_objects = ep_data.get("task_relevant_objects", [])
    first_frame_bboxes = ep_data.get("first_frame_bboxes", [])
    
    return object_list, task_relevant_objects, first_frame_bboxes

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
    
    return vis_img

def merge_bboxes_by_object_name(bboxes):
    """
    Merge bounding boxes that have the same object name by taking their union.
    
    Args:
        bboxes: List of tuples (confidence, label, box)
        
    Returns:
        List of merged bounding boxes
    """
    object_groups = {}
    
    # Group bboxes by object name
    for confidence, label, box in bboxes:
        if label not in object_groups:
            object_groups[label] = []
        object_groups[label].append((confidence, box))
    
    merged_bboxes = []
    for label, box_group in object_groups.items():
        if len(box_group) == 1:
            # Only one bbox for this object, no need to merge
            merged_bboxes.append((box_group[0][0], label, box_group[0][1]))
        else:
            # Multiple bboxes for this object, take the union
            # Initialize with first box coordinates
            confidences = [conf for conf, _ in box_group]
            max_confidence = max(confidences)
            
            # Find union of all boxes
            x_min = min(box[0] for _, box in box_group)
            y_min = min(box[1] for _, box in box_group)
            x_max = max(box[2] for _, box in box_group)
            y_max = max(box[3] for _, box in box_group)
            
            merged_box = [x_min, y_min, x_max, y_max]
            merged_bboxes.append((max_confidence, label, merged_box))
    
    return merged_bboxes

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
    instruction = str(next(iter(episode["steps"]))["language_instruction"].numpy().decode())

    if file_path not in bbox_results_json.keys():
        bbox_results_json[file_path] = {}

    # Get pre-generated object list, task-relevant objects, and first frame bboxes
    pre_generated_objects, task_relevant_objects, pre_generated_bboxes = get_pre_generated_objects(file_path, episode_id)
    
    # Skip episodes without pre-generated objects
    if not pre_generated_objects or not pre_generated_bboxes:
        print(f"No pre-generated data found for episode {episode_id}. Skipping.")
        continue

    # Separate task-relevant from task-irrelevant bboxes
    task_relevant_bboxes = []
    task_irrelevant_bboxes = []
    
    for bbox in pre_generated_bboxes:
        # Skip any bboxes related to robotic grippers
        label = bbox["label"].lower()

        if "gripper" in label:
            continue
        
        # Check if this object is among the task-relevant objects
        # Note: Need to handle potential differences in naming (e.g., "bbq sauce" vs "bbq sauce bottle")
        is_relevant = False
        for rel_obj in task_relevant_objects:
            if rel_obj.lower() in label.lower() or label.lower() in rel_obj.lower():
                is_relevant = True
                break
                
        if is_relevant:
            task_relevant_bboxes.append(bbox)
        else:
            task_irrelevant_bboxes.append(bbox)
    
    print(f"Found {len(task_relevant_bboxes)} task-relevant and {len(task_irrelevant_bboxes)} task-irrelevant objects:")
    print("task relevant: ", [d['label'] for d in task_relevant_bboxes])
    print("task irrelevant: ", [d['label'] for d in task_irrelevant_bboxes])
    
    # Create prompt with only task-relevant objects for GDINO
    task_objects_text = ". ".join([s.lower() for s in task_relevant_objects])
    if not task_objects_text.endswith("."):
        task_objects_text += "."
    
    start_time = time.time()
    bboxes_list = []
    
    # Initialize list to store visualization frames if enabled
    visualization_frames = [] if args.visualize else None
    
    for step_idx, step in tqdm.tqdm(
        enumerate(episode["steps"]),
        total=len(episode["steps"]),
        desc=f"Generating bounding boxes for Episode {episode_id}",
        leave=False,
    ):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())
        
        # For all frames, track task-relevant objects with GDINO
        current_bboxes = []
        
        # Only run GDINO if there are task-relevant objects to track
        if task_relevant_objects:
            # Use GDINO to detect and track task-relevant objects
            inputs = processor(
                images=image,
                text=task_objects_text,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, 
                box_threshold=BOX_THRESHOLD, 
                text_threshold=TEXT_THRESHOLD, 
                target_sizes=[image.size[::-1]])[0]

            logits, phrases, boxes = (
                results["scores"].cpu().numpy(),
                results["labels"],
                results["boxes"].cpu().numpy(),
            )

            # Convert GDINO results to our standard format
            for lg, p, b in zip(logits, phrases, boxes):
                b = list(b.astype(int))
                lg = round(lg, 5)
                current_bboxes.append({
                    "label": p,
                    "box": b,
                    "confidence": lg
                })
        
        # For all frames, add the task-irrelevant objects from the first frame
        # (We're not tracking these, just copying them)
        for irrelevant_bbox in task_irrelevant_bboxes:
            current_bboxes.append(irrelevant_bbox)

        # add missing relevant bboxes
        for relevant_bbox in task_relevant_bboxes:
            if relevant_bbox["label"] not in [bbox["label"] for bbox in current_bboxes]:
                current_bboxes.append(relevant_bbox)
        
        # Add current bboxes to the list
        bboxes_list.append(current_bboxes)
        
        # Create visualization if enabled
        if args.visualize:
            # Format bboxes for visualization (which expects tuples)
            vis_bboxes = []
            for bbox in current_bboxes:
                vis_bboxes.append((
                    bbox["confidence"], 
                    bbox["label"], 
                    bbox["box"]
                ))
                
            # Convert PIL image to numpy for OpenCV processing
            img_array = np.array(image)
            # Draw bounding boxes on the image
            vis_frame = visualize_bboxes(img_array, vis_bboxes, frame_idx=step_idx)
            visualization_frames.append(vis_frame)

    end_time = time.time()
    bbox_results_json[file_path][str(episode_id)] = {
        "episode_id": str(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }

    with open(bbox_json_path, "w") as f:
        json.dump(bbox_results_json, f, indent=2, cls=NumpyFloatValuesEncoder)
    print(f"ID {args.id} finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end_time - start_time, 2)}")
    
    # Save visualization video if enabled
    if args.visualize and visualization_frames:
        # Use custom output directory if provided, otherwise use default
        video_dir = args.output_video_dir if args.output_video_dir else f"{result_path}/videos/{args.id}"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/episode_{episode_id}_{instruction}.mp4"
        mediapy.write_video(video_path, visualization_frames, fps=10)
        print(f"Saved visualization video to {video_path}")
