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
from transformers import SamModel, SamProcessor, pipeline
import tensorflow_datasets as tfds
import argparse
import tqdm

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--splits", type=int, default=2)
parser.add_argument("--dataset_name", type=str, default="cobot_rlds")
parser.add_argument("--data_dir", type=str, default="datasets")
args = parser.parse_args()

checkpoint = "google/owlvit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda" if torch.cuda.is_available() else "cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
image_dims = (256, 256) #(256, 256)
image_label = "image" #"image_0"
ee_pose_label = "state"

def get_bounding_boxes(img, prompt="the robotic gripper"):
    predictions = detector(img, candidate_labels=[prompt], threshold=0.01)

    return predictions


def show_box(box, ax, meta, color):
    x0, y0 = box["xmin"], box["ymin"]
    w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, "{:.3f}".format(meta["score"]), color="white")


def get_median(mask, p):
    row_sum = np.sum(mask, axis=1)
    cumulative_sum = np.cumsum(row_sum)

    if p >= 1.0:
        p = 1

    total_sum = np.sum(row_sum)
    threshold = p * total_sum

    return np.argmax(cumulative_sum >= threshold)


def get_gripper_mask(img, pred):
    box = [
        round(pred["box"]["xmin"], 2),
        round(pred["box"]["ymin"], 2),
        round(pred["box"]["xmax"], 2),
        round(pred["box"]["ymax"], 2),
    ]

    inputs = sam_processor(img, input_boxes=[[[box]]], return_tensors="pt")
    # make sure inputs' devices is the same as sam_model's device
    inputs = inputs.to(sam_model.device)
    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0]
    # cpu().numpy() if mask is on gpu else .numpy()
    mask = mask.cpu().numpy() if mask.device.type == "cuda" else mask.numpy()

    return mask


def sq(w, h):
    """
    Creates a coordinate grid of shape (h, w, 2) where each point contains its (x, y) coordinates.
    
    Example:
    sq(3, 2) returns:
    [[[0, 0], [1, 0], [2, 0]],
     [[0, 1], [1, 1], [2, 1]]]
    
    This is used to map between pixel positions and their coordinates in the image.
    """
    return np.concatenate(
        [(np.arange(w * h).reshape(h, w) % w)[:, :, None], (np.arange(w * h).reshape(h, w) // w)[:, :, None]], axis=-1
    )


def mask_to_pos_weighted(mask):
    pos = sq(*image_dims)

    weight = pos[:, :, 0] + pos[:, :, 1]
    weight = weight * weight

    x = np.sum(mask * pos[:, :, 0] * weight) / np.sum(mask * weight)
    y = get_median(mask * weight, 0.95)

    return x, y

def mask_to_pos_naive(mask):
    """
    Finds the position of the gripper in the image using a naive approach.
    
    This function:
    1. Creates a coordinate grid for the image
    2. Weights the mask by the sum of x and y coordinates
    3. Finds the position with the maximum weighted value
    4. Applies an offset to center the position relative to the gripper
    
    Args:
        mask: Binary mask of the gripper
        
    Returns:
        (x, y): Tuple of coordinates representing the gripper position
    """
    pos = sq(*image_dims)
    weight = pos[:, :, 0] + pos[:, :, 1]
    min_pos = np.argmax((weight * mask).flatten())

    return min_pos % image_dims[0] - (image_dims[0] / 16), min_pos // image_dims[0] - (image_dims[0] / 16) #24)


def get_gripper_pos(episode_id, frame, builder, plot=True):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    images = [step["observation"][image_label] for step in episode["steps"]]

    img = Image.fromarray(images[frame].numpy())
    predictions = get_bounding_boxes(img)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img)

        for prediction in predictions:
            if prediction["score"] < 0.05:
                continue
            box = prediction["box"]
            show_box(box, ax, prediction, "red")

    if len(predictions) > 0:
        mask = get_gripper_mask(img, predictions[0])
        pos = mask_to_pos_naive(mask)

        if plot:
            plt.imshow(mask, alpha=0.5)
            plt.scatter([pos[0]], [pos[1]])
    else:
        print("No valid bounding box")

    if plot:
        plt.show()


def get_gripper_pos_raw(img):
    img = Image.fromarray(img.numpy())
    predictions = get_bounding_boxes(img)

    if len(predictions) > 0:
        mask = get_gripper_mask(img, predictions[0])
        pos = mask_to_pos_naive(mask)
    else:
        mask = np.zeros(image_dims)
        pos = (-1, -1)
        predictions = [None]

    return (int(pos[0]), int(pos[1])), mask, predictions[0]


def process_trajectory(episode):
    images = [step["observation"][image_label] for step in episode["steps"]]
    # states = [step["observation"]["state"] for step in episode["steps"]]
    states = [step["observation"][ee_pose_label] for step in episode["steps"]]

    # raw_trajectory = [(*get_gripper_pos_raw(img), state) for img, state in zip(images, states)]
    raw_trajectory = []
    for img, state in tqdm.tqdm(zip(images, states), desc="Processing trajectory", total=len(images)):
        results = get_gripper_pos_raw(img)
        raw_trajectory.append([*results, state])

    prev_found = list(range(len(raw_trajectory)))
    next_found = list(range(len(raw_trajectory)))

    prev_found[0] = -1e6
    next_found[-1] = 1e6

    for i in range(1, len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            prev_found[i] = prev_found[i - 1]

    for i in reversed(range(0, len(raw_trajectory) - 1)):
        if raw_trajectory[i][2] is None:
            next_found[i] = next_found[i + 1]

    if next_found[0] == next_found[-1]:
        # the gripper was never found
        return None

    # Replace the not found positions with the closest neighbor
    for i in range(0, len(raw_trajectory)):
        raw_trajectory[i] = raw_trajectory[prev_found[i] if i - prev_found[i] < next_found[i] - i else next_found[i]]

    return raw_trajectory


def get_corrected_positions(episode_id, builder, plot=False, output_dir=None):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    t = process_trajectory(episode)
    metadata = dict()
    for key in episode["episode_metadata"].keys():
        if isinstance(episode["episode_metadata"][key], tf.Tensor):
            metadata[key] = episode["episode_metadata"][key].numpy()
            if isinstance(metadata[key], bytes):
                metadata[key] = metadata[key].decode()
        else:
            metadata[key] = episode["episode_metadata"][key]

    images = [step["observation"][image_label] for step in episode["steps"]]
    images = [img.numpy() for img in images]

    pos = [tr[0] for tr in t]

    points_2d = np.array(pos, dtype=np.float32)
    points_3d = np.array([tr[-1][:3] for tr in t])

    from sklearn.linear_model import RANSACRegressor

    points_3d_pr = np.concatenate([points_3d, np.ones_like(points_3d[:, :1])], axis=-1)
    points_2d_pr = np.concatenate([points_2d, np.ones_like(points_2d[:, :1])], axis=-1)
    reg = RANSACRegressor(random_state=0).fit(points_3d_pr, points_2d_pr)

    pr_pos = reg.predict(points_3d_pr)[:, :-1].astype(int)

    if plot:
        images = [
            cv2.circle(img, (int(p[0]), int(p[1])), radius=5, color=(255, 0, 0), thickness=-1)
            for img, p in zip(images, pr_pos)
        ]
        mediapy.write_video(f"{output_dir}/gripper_trajectory_{episode_id}.mp4", images, fps=10)

    return pr_pos, metadata


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

    output_dir = f"./outputs/{args.dataset_name}/gripper_positions"
    video_dir = f"./outputs/{args.dataset_name}/gripper_positions/videos/{args.id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    for index in tqdm.tqdm(episode_indexes, desc=f"Processing episodes {args.id} / {args.splits}"):
        pr_pos, metadata = get_corrected_positions(index, builder, plot=True, output_dir=video_dir)
        file_path, episode_id = metadata["file_path"], metadata["episode_id"]
        if file_path not in json_data.keys():
            json_data[file_path] = {}
        json_data[file_path][episode_id] = {"gripper_positions": pr_pos, "metadata": metadata}
        with open(f"{output_dir}/gripper_positions_{args.id}.json", "w") as f:
            json.dump(jsonify(json_data), f)