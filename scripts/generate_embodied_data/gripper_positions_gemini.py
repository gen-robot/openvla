import cv2
import os
import matplotlib
import mediapy
import numpy as np
import torch
import json
import tqdm
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
# import google.generativeai as genai
from google import genai
from transformers import SamModel, SamProcessor, pipeline
import tensorflow_datasets as tfds
import argparse
import tqdm
import re # For parsing coordinates
import time # For timing operations
import concurrent.futures # For potential parallel processing
from scripts.generate_embodied_data.utils import Gemini, NumpyFloatValuesEncoder # Use existing Gemini class

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--splits", type=int, default=2)
parser.add_argument("--dataset_name", type=str, default="cobot_rlds")
parser.add_argument("--data_dir", type=str, default="datasets")
parser.add_argument("--batch_size", type=int, default=1, help="Number of frames to process in each batch (for optimization)")
args = parser.parse_args()

image_dims = (256, 256) #(256, 256)
image_label = "image" #"image_0"
ee_pose_label = "state"

# Create a modified Gemini class that can handle image inputs
class GeminiVision(Gemini):
    def __init__(self, model_name="gemini-2.0-flash"):
        super().__init__(model_name)
        # Direct client initialization like in example.py
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    def generate_with_image(self, prompt, image):
        """Generate text response based on prompt and image input"""
        try:
            # Use the direct client.models.generate_content pattern from example.py
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    image,
                    prompt
                ],
                config=genai.types.GenerateContentConfig(
                    temperature=0.1  # Lower temperature for more deterministic results
                )
            )
            
            if hasattr(response, 'text'):
                return response.text
            else:
                print("Warning: Response has no text attribute")
                return None
        except Exception as e:
            print(f"Error generating with image: {e}")
            return None
    
    def parse_coordinates(self, response_text):
        """Parse coordinate tuple from response text"""
        if not response_text:
            return (-1, -1)
        
        # First try to parse JSON format
        try:
            # Look for JSON-like content in the response
            json_match = re.search(r'\[.*\]', response_text.replace('\n', ''))
            if json_match:
                json_data = json.loads(json_match.group(0))
                # Check if we have a list of points
                if isinstance(json_data, list) and len(json_data) > 0:
                    # Get the first point (or the one labeled as 0/start)
                    for item in json_data:
                        if "point" in item:
                            # Handle [y, x] format as used in example.py
                            if isinstance(item["point"], list) and len(item["point"]) == 2:
                                y, x = item["point"]
                                # Convert from normalized 0-1000 to image pixel coordinates if needed
                                x_pixel = int(x * image_dims[1] / 1000) if x <= 1000 else int(x)
                                y_pixel = int(y * image_dims[0] / 1000) if y <= 1000 else int(y)
                                return (x_pixel, y_pixel)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            pass  # Fall back to regex parsing
            
        # Traditional regex pattern as backup
        match = re.search(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', response_text)
        if match:
            try:
                x = int(match.group(1))
                y = int(match.group(2))
                return (x, y)
            except ValueError:
                return (-1, -1)
        
        # Check if response explicitly indicates not found
        if "(-1, -1)" in response_text:
            return (-1, -1)
            
        # No coordinate pattern found
        return (-1, -1)

# Initialize the modified Gemini client
gemini_client = GeminiVision()


def get_gripper_pos_gemini(img, max_retries=3, initial_delay=2):
    """Get gripper position from an image using Gemini with retry logic"""
    # More detailed prompt similar to example.py
    prompt = f"""
    Analyze this image and identify the precise center of the robotic gripper tip.
    
    Return the point as a single JSON list in this exact format:
    [{{"point": [y, x], "label": "gripper_tip"}}]
    
    The points should be in [y, x] format normalized to 0-1000 range where:
    - (0, 0) is the top-left corner of the image
    - (1000, 1000) is the bottom-right corner of the image
    
    If the gripper is not visible in the image, return [{{"point": [-1, -1], "label": "not_found"}}]
    """

    # Add retry logic
    for attempt in range(max_retries):
        try:
            # Call Gemini with the image
            response = gemini_client.generate_with_image(prompt, img)
            
            # Parse the coordinates from the response
            return gemini_client.parse_coordinates(response)
        
        except Exception as e:
            wait_time = initial_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt+1}/{max_retries} failed with error: {e}")
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    
    # If we've exhausted all retries, return not found
    print("All retry attempts failed, returning (-1, -1)")
    return (-1, -1)

def process_trajectory_batch(episode, batch_size=args.batch_size):
    """Process trajectory images in batches to reduce API latency"""
    # Extract images and states
    images = [Image.fromarray(step["observation"][image_label].numpy()) for step in episode["steps"]]
    states = [step["observation"][ee_pose_label] for step in episode["steps"]]
    
    # Storage for results
    raw_trajectory = []
    
    # Process in batches
    n_images = len(images)
    for i in tqdm.tqdm(range(0, n_images, batch_size), total=n_images // batch_size, desc="Processing batches"):
        batch_images = images[i:min(i+batch_size, n_images)]
        batch_states = states[i:min(i+batch_size, n_images)]
        
        # Create prompts for each image
        prompts = [f"""Analyze this image and identify the robotic gripper tip.
Return ONLY the pixel coordinates of the center of the gripper tip as (x, y).
The image has dimensions {image_dims[1]} x {image_dims[0]} (width x height).
Ensure coordinates are within range: 0 <= x < {image_dims[1]} and 0 <= y < {image_dims[0]}.
If the gripper is not visible, return (-1, -1).""" for _ in batch_images]
        
        # Process batch
        if batch_size > 1:
            assert False, "Batch processing not supported yet"
            # Process as batch
            batch_inputs = list(zip(prompts, batch_images))
            batch_positions = gemini_client.generate_batch(batch_inputs)
            
            # Store results with corresponding states
            for pos, state in zip(batch_positions, batch_states):
                raw_trajectory.append([pos, state])
        else:
            # Process sequentially for single batch size
            for img, state in zip(batch_images, batch_states):
                pos = get_gripper_pos_gemini(img)
                raw_trajectory.append([pos, state])
    
    # Handle interpolation for missing positions
    interpolated_trajectory = interpolate_missing_positions(raw_trajectory)
    
    return interpolated_trajectory

def interpolate_missing_positions(raw_trajectory):
    """Interpolate positions where the gripper wasn't found"""
    # Special value indicating gripper not found
    not_found = (-1, -1)
    
    # Find indices of valid positions (where gripper was found)
    valid_indices = [i for i, (pos, _) in enumerate(raw_trajectory) if pos != not_found]
    
    # If no valid positions found, return None to indicate failure
    if not valid_indices:
        return None
    
    # Create a copy of the trajectory to modify
    interpolated = [list(item) for item in raw_trajectory]
    
    # Interpolate for each position where gripper wasn't found
    for i in range(len(raw_trajectory)):
        if raw_trajectory[i][0] == not_found:
            # Find nearest valid positions before and after
            prev_valid = max([idx for idx in valid_indices if idx < i], default=-1)
            next_valid = min([idx for idx in valid_indices if idx > i], default=len(raw_trajectory))
            
            # Choose nearest valid position
            if prev_valid == -1:
                # No valid position before, use next
                nearest_valid = next_valid
            elif next_valid == len(raw_trajectory):
                # No valid position after, use previous
                nearest_valid = prev_valid
            else:
                # Choose nearest
                nearest_valid = prev_valid if (i - prev_valid) <= (next_valid - i) else next_valid
            
            # Use position from nearest valid index (if it exists)
            if 0 <= nearest_valid < len(raw_trajectory):
                interpolated[i][0] = raw_trajectory[nearest_valid][0]
    
    # Convert states to numpy arrays
    for i in range(len(interpolated)):
        if isinstance(interpolated[i][1], tf.Tensor):
            interpolated[i][1] = interpolated[i][1].numpy()
    
    return interpolated

def process_trajectory(episode):
    """Process a trajectory using optimized batching if possible"""
    return process_trajectory_batch(episode, batch_size=args.batch_size)

def get_corrected_positions(episode_id, builder, plot=False, output_dir=None):
    """Get corrected gripper positions using RANSAC"""
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    
    # Process trajectory to get gripper positions
    t = process_trajectory(episode)
    
    # Handle case where gripper was never found
    if t is None:
        print(f"Skipping episode {episode_id} due to no gripper detections.")
        return None, None
    
    # Extract metadata
    metadata = dict()
    for key in episode["episode_metadata"].keys():
        if isinstance(episode["episode_metadata"][key], tf.Tensor):
            metadata[key] = episode["episode_metadata"][key].numpy()
            if isinstance(metadata[key], bytes):
                metadata[key] = metadata[key].decode()
        else:
            metadata[key] = episode["episode_metadata"][key]
    
    # Extract 2D positions and 3D states
    pos = [tr[0] for tr in t]
    points_3d = np.array([tr[1][:3] for tr in t])
    
    # Check if we have enough valid points for RANSAC
    valid_indices = [i for i, p in enumerate(pos) if p != (-1, -1)]
    if len(valid_indices) < 3:  # RANSAC needs at least 3 samples
        print(f"Warning: Not enough valid positions ({len(valid_indices)}) for RANSAC in episode {episode_id}.")
        return None, metadata
    
    # Prepare data for RANSAC
    points_2d = np.array([pos[i] for i in valid_indices], dtype=np.float32)
    points_3d_valid = np.array([points_3d[i] for i in valid_indices])
    
    from sklearn.linear_model import RANSACRegressor
    
    # Augment with homogeneous coordinates for RANSAC
    points_3d_pr = np.concatenate([points_3d_valid, np.ones_like(points_3d_valid[:, :1])], axis=-1)
    points_2d_pr = np.concatenate([points_2d, np.ones_like(points_2d[:, :1])], axis=-1)
    
    try:
        # Fit RANSAC model
        reg = RANSACRegressor(random_state=0).fit(points_3d_pr, points_2d_pr)
        
        # Predict positions for all steps
        all_points_3d_pr = np.concatenate([points_3d, np.ones_like(points_3d[:, :1])], axis=-1)
        pr_pos = reg.predict(all_points_3d_pr)[:, :-1].astype(int)
        
        # Create visualization if requested
        if plot:
            images_np = [step["observation"][image_label].numpy() for step in episode["steps"]]
            images_with_circles = []
            
            for i, img_np in enumerate(images_np):
                # Create a copy of the image to avoid modifying the original
                vis_img = img_np.copy()
                
                # Draw the RANSAC-predicted position
                pred_pos = pr_pos[i]
                if 0 <= pred_pos[0] < img_np.shape[1] and 0 <= pred_pos[1] < img_np.shape[0]:
                    vis_img = cv2.circle(vis_img, tuple(pred_pos), radius=5, color=(0, 0, 255), thickness=-1) # color: blue
                    # add text to the image to show the legend of the color
                    cv2.putText(vis_img, "RANSAC-predicted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw the Gemini-detected position (if available)
                orig_pos = pos[i]
                if orig_pos != (-1, -1) and 0 <= orig_pos[0] < img_np.shape[1] and 0 <= orig_pos[1] < img_np.shape[0]:
                    vis_img = cv2.circle(vis_img, orig_pos, radius=3, color=(255, 0, 0), thickness=-1) # color: red
                    cv2.putText(vis_img, "Gemini-detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                images_with_circles.append(vis_img)
            
            # Write video if we have images
            if images_with_circles:
                mediapy.write_video(f"{output_dir}/gripper_trajectory_{episode_id}.mp4", images_with_circles, fps=20)
        
        return pr_pos, metadata
        
    except Exception as e:
        print(f"Error during RANSAC for episode {episode_id}: {e}")
        return None, metadata

def jsonify(data):
    """Convert numpy types to JSON-serializable types"""
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

if __name__ == "__main__":
    json_data = {}

    builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)
    total_num_episodes = builder.info.splits["train"].num_examples

    def get_id_range(id, splits, total_num_episodes):
        split_percents = 100 // splits
        start = id * split_percents
        end = (id + 1) * split_percents
        start_episode_id = int(total_num_episodes * start / 100)
        end_episode_id = int(total_num_episodes * end / 100)
        if id == splits - 1:  # Last split should include the final episode
            end_episode_id = total_num_episodes
        return start_episode_id, end_episode_id

    # Check split coverage
    all_episodes = set(range(total_num_episodes))
    covered_episodes = set()
    
    for id_check in range(args.splits):
        start_id, end_id = get_id_range(id_check, args.splits, total_num_episodes)
        episodes_in_split = set(range(start_id, end_id))
        covered_episodes.update(episodes_in_split)
    
    missing_episodes = all_episodes - covered_episodes
    if missing_episodes:
        print(f"WARNING: {len(missing_episodes)} episodes will not be processed by any split!")
    else:
        print(f"All {total_num_episodes} episodes will be covered by the splits.")
    
    # Get the range for the current split
    start_episode_id, end_episode_id = get_id_range(args.id, args.splits, total_num_episodes)
    print(f"This process (ID {args.id}) will handle episodes {start_episode_id} to {end_episode_id-1}")
    
    episode_indexes = list(range(start_episode_id, end_episode_id))

    # Create output directories
    output_dir = f"./outputs/{args.dataset_name}/gripper_positions" 
    video_dir = f"{output_dir}/videos/{args.id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    save_file_path = f"{output_dir}/gripper_positions_{args.id}.json"
    
    # Load existing data if available
    if os.path.exists(save_file_path):
        print(f"Loading existing data from {save_file_path}")
        try:
            with open(save_file_path, "r") as f:
                json_data = json.load(f)
            print(f"Loaded {sum(len(v) for v in json_data.values())} existing episode entries.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {save_file_path}. Starting fresh.")
            json_data = {}

    # Process episodes
    processed_count = 0
    start_time = time.time()
    
    for index in tqdm.tqdm(episode_indexes, desc=f"Processing episodes {args.id}/{args.splits}"):
        try:
            # Get corrected positions for this episode
            pr_pos, metadata = get_corrected_positions(index, builder, plot=True, output_dir=video_dir)
            
            # Skip if processing failed
            if pr_pos is None or metadata is None:
                continue
                
            # Extract metadata for storage
            file_path, episode_id_str = metadata["file_path"], str(metadata["episode_id"])
            
            # Skip if already processed
            if file_path in json_data and episode_id_str in json_data[file_path]:
                print(f"Skipping episode {index} ({file_path}, {episode_id_str}) - already processed.")
                continue
                
            # Store results
            if file_path not in json_data:
                json_data[file_path] = {}
                
            json_data[file_path][episode_id_str] = {
                "gripper_positions": pr_pos, 
                "metadata": metadata
            }
            
            processed_count += 1
            
            # Save periodically
            if processed_count > 0 and processed_count % 5 == 0:
                with open(save_file_path, "w") as f:
                    json.dump(jsonify(json_data), f, cls=NumpyFloatValuesEncoder)
                    
                # Report progress and speed
                elapsed = time.time() - start_time
                episodes_per_hour = processed_count / (elapsed / 3600)
                print(f"\nProcessed {processed_count} episodes in {elapsed:.1f}s " 
                      f"({episodes_per_hour:.1f} episodes/hour)")
                
        except Exception as e:
            print(f"Error processing episode {index}: {e}")
            # Continue with next episode

        # Final save
        with open(save_file_path, "w") as f:
            json.dump(jsonify(json_data), f, cls=NumpyFloatValuesEncoder, indent=2)
        
    # Report final statistics
    total_elapsed = time.time() - start_time
    print(f"\nFinished processing {processed_count} episodes in {total_elapsed:.1f}s")
    if processed_count > 0:
        print(f"Average speed: {processed_count / (total_elapsed / 3600):.1f} episodes/hour")