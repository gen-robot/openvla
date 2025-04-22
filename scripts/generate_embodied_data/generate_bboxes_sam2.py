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
from utils import NumpyFloatValuesEncoder, post_process_caption
import glob

# Import from sam_utils.py
from embodied_agent.core.vlm.utils.sam_utils import GroundedSAM2

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--splits", default=4, type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--visualize", default=False, action="store_true", help="Generate visualization videos of bounding boxes")
    parser.add_argument("--include_masks", default=False, action="store_true", help="Include segmentation masks in output")
    parser.add_argument("--output_video_dir", type=str, help="Directory to save visualization videos (defaults to results_path/videos/)")
    return parser.parse_args()


def load_dataset(args):
    print("Loading data...")
    split_percents = 100 // args.splits
    start = args.id * split_percents
    end = (args.id + 1) * split_percents

    ds = tfds.load(args.dataset_name, data_dir=args.data_dir, split=f"train[{start}%:{end}%]")
    print("Done.")
    return ds, start, end


def load_object_lists(dataset_name):
    """Load pre-generated object lists if provided"""
    object_lists_json = {}
    first_frame_bboxes = {}
    object_lists_dir = f"./outputs/{dataset_name}/object_lists_gemini"
    
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
    
    return object_lists_json, first_frame_bboxes, object_lists_dir


def load_model(args):
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load GroundedSAM2 model instead of just gDINO
    print(f"Loading GroundedSAM2 to device {device}...")
    grounded_sam = GroundedSAM2(device=device)
    print("Done.")
    
    return grounded_sam, device


def get_pre_generated_objects(file_path, episode_id, object_lists_json, object_lists_dir):
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

    final_object_list = [d['label'] for d in first_frame_bboxes]

    return final_object_list, task_relevant_objects, first_frame_bboxes


def visualize_results(image, bboxes, masks=None, frame_idx=None):
    """
    Draw bounding boxes and masks on the image
    If masks are provided, they will be drawn as well
    """
    # Create a copy of the image to avoid modifying original
    vis_img = image.copy()
    
    # If masks are provided, overlay masks with transparency
    if masks is not None and len(masks) > 0:
        # Convert masks to boolean format if needed
        if isinstance(masks, np.ndarray) and masks.ndim > 2:
            masks = masks.astype(bool)
        
        # Create detections object for GroundedSAM2's annotate_image method
        detections = {
            "dino_boxes": np.array([bbox["box"] for bbox in bboxes]),
            "dino_confidences": np.array([bbox["confidence"] for bbox in bboxes]),
            "dino_labels": [bbox["label"] for bbox in bboxes]
        }
        
        # Call staticmethod directly on the class instead of creating an instance
        vis_img = GroundedSAM2.annotate_image(
            image=vis_img, 
            masks=masks,
            logs=detections,
            include_masks=False,
            include_boxes=True,
            include_labels=True
        )
    else:
        # If no masks, just draw bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["box"]
            label = f"{bbox['label']}: {bbox['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return vis_img


def process_episode(episode, grounded_sam, device, object_lists_json, object_lists_dir, args):
    """Process a single episode to generate bounding boxes and masks"""
    episode_id = episode["episode_metadata"]["episode_id"].numpy()
    if isinstance(episode_id, bytes):
        episode_id = episode_id.decode()
    file_path = episode["episode_metadata"]["file_path"].numpy().decode()
    print(f"ID {args.id} starting ep: {episode_id}, {file_path}")
    instruction = str(next(iter(episode["steps"]))["language_instruction"].numpy().decode())

    # Get pre-generated object list, task-relevant objects, and first frame bboxes
    pre_generated_objects, task_relevant_objects, pre_generated_bboxes = get_pre_generated_objects(
        file_path, episode_id, object_lists_json, object_lists_dir
    )
    
    # Skip episodes without pre-generated objects
    if not pre_generated_objects or not pre_generated_bboxes:
        print(f"No pre-generated data found for episode {episode_id}. Skipping.")
        return None, None
    
    # Create prompt with only task-relevant objects for GroundedSAM
    task_objects_text = ". ".join([s.lower() for s in task_relevant_objects])
    if not task_objects_text.endswith("."):
        task_objects_text += "."
        
    start_time = time.time()
    bboxes_list = []
    masks_list = [] if args.include_masks else None
    
    # Initialize list to store visualization frames if enabled
    visualization_frames = [] if args.visualize else None
    
    for step_idx, step in tqdm.tqdm(
        enumerate(episode["steps"]),
        total=len(episode["steps"]),
        desc=f"Generating segmentations for Episode {episode_id}",
        leave=False,
    ):
        if step_idx == 0:
            lang_instruction = step["language_instruction"].numpy().decode()
        
        # Get image from step
        image = step["observation"]["image"].numpy()
        
        # Skip processing if no task-relevant objects
        if not task_relevant_objects:
            # Add empty results and continue
            bboxes_list.append([])
            if args.include_masks:
                masks_list.append([])
            
            if args.visualize:
                visualization_frames.append(image)
            continue
                
        # Use GroundedSAM2 for segmentation
        try:
            masks, logs = grounded_sam.grounded_segment(image, task_objects_text)
            
            # Extract boxes from logs
            current_bboxes = []
            for i, (box, confidence, label) in enumerate(zip(
                logs["dino_boxes"], 
                logs["dino_confidences"], 
                logs["dino_labels"]
            )):
                current_bboxes.append({
                    "label": label,
                    "box": box.astype(int).tolist(),
                    "confidence": float(confidence)
                })
            
            # Store bboxes in the existing format
            bboxes_list.append(current_bboxes)
            
            # Store masks if requested
            if args.include_masks:
                # Convert masks to boolean format for efficient storage
                if isinstance(masks, np.ndarray):
                    # If masks is 4D (batch, 1, h, w), squeeze to 3D (batch, h, w)
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)
                    masks_list.append(masks.astype(bool).tolist())
                else:
                    masks_list.append(masks)
            
            # Create visualization if enabled
            if args.visualize:
                vis_frame = visualize_results(image, current_bboxes, masks)
                visualization_frames.append(vis_frame)
                
        except Exception as e:
            print(f"Error processing frame {step_idx} in episode {episode_id}: {e}")
            # Add empty results for this frame
            bboxes_list.append([])
            if args.include_masks:
                masks_list.append([])
            
            if args.visualize:
                visualization_frames.append(image)

    end_time = time.time()
    print(f"ID {args.id} finished ep. Elapsed time: {round(end_time - start_time, 2)}")
    
    # Maintain original format while adding masks if needed
    result = {
        "episode_id": str(episode_id),
        "file_path": file_path,
        "bboxes": bboxes_list,
    }
    
    # Add masks as a separate field if included
    # if args.include_masks:
    #     result["masks"] = masks_list
    
    return result, visualization_frames


def save_visualization(visualization_frames, episode_id, instruction, result_path, args):
    """Save visualization video if enabled"""
    if args.visualize and visualization_frames:
        # Use custom output directory if provided, otherwise use default
        video_dir = args.output_video_dir if args.output_video_dir else f"{result_path}/videos/{args.id}"
        os.makedirs(video_dir, exist_ok=True)
        
        # Sanitize instruction for filename
        safe_instruction = "".join(c if c.isalnum() else "_" for c in instruction)[:50]
        video_path = f"{video_dir}/episode_{episode_id}_{safe_instruction}.mp4"
        
        mediapy.write_video(video_path, visualization_frames, fps=10)
        print(f"Saved visualization video to {video_path}")


def main():
    args = parse_args()
    result_path = f"./outputs/{args.dataset_name}/bboxes_sam2"
    os.makedirs(result_path, exist_ok=True)
    results_json_path = os.path.join(result_path, f"results_{args.id}.json")
    
    ds, start, end = load_dataset(args)
    object_lists_json, first_frame_bboxes, object_lists_dir = load_object_lists(args.dataset_name)
    grounded_sam, device = load_model(args)
    
    results_json = {}
    
    for ep_idx, episode in tqdm.tqdm(
        enumerate(ds),
        total=len(ds),
        desc=f"Processing episodes [{start}%:{end}%]",
    ):
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        episode_id = episode["episode_metadata"]["episode_id"].numpy()
        if isinstance(episode_id, bytes):
            episode_id = episode_id.decode()
        instruction = str(next(iter(episode["steps"]))["language_instruction"].numpy().decode())
        
        if file_path not in results_json.keys():
            results_json[file_path] = {}
        
        episode_result, visualization_frames = process_episode(
            episode, grounded_sam, device, object_lists_json, object_lists_dir, args
        )
        
        if episode_result:
            results_json[file_path][str(episode_id)] = episode_result
            
            # Save visualization if enabled
            save_visualization(visualization_frames, episode_id, instruction, result_path, args)
        
        # Save results after each episode
        with open(results_json_path, "w") as f:
            json.dump(results_json, f, indent=2, cls=NumpyFloatValuesEncoder)
        
        print(f"ID {args.id} processed episode {ep_idx + 1}/{len(ds)}")


if __name__ == "__main__":
    main()
