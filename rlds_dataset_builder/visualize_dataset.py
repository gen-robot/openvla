import argparse
import tqdm
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb


WANDB_ENTITY = None
WANDB_PROJECT = 'vis_rlds'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', help='name of the dataset to visualize', default="panda_simpler_spoon_dataset")
parser.add_argument('--version', help='version of the dataset to visualize', default=None)
parser.add_argument('--dir', help='dir', default="/nvme_data/bingwen/tensorflow_datasets/")
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY,
               project=WANDB_PROJECT)
else:
    render_wandb = False

# create TF dataset
dataset_name = args.dataset_name
if args.version is not None:
    dataset_name = f"{dataset_name}:{args.version}"
print(f"Visualizing data from dataset: {dataset_name}")
# module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, data_dir=args.dir, split='train')
print(f"Number of episodes: {len(ds)}")
# ds = ds.shuffle(100)
save_dir = os.path.join(args.dir, args.dataset_name, args.version, "visualize")
os.makedirs(save_dir, exist_ok=True)

# visualize episodes
for i, episode in enumerate(ds.take(5)):
    images = []
    for step in episode['steps']:
        images.append(step['observation']['image'].numpy())
    image_strip = np.concatenate(images[::2], axis=1)
    caption = step['language_instruction'].numpy().decode() + ' (temp. downsampled 4x)'

    if render_wandb:
        wandb.log({f'image_{i}': wandb.Image(image_strip, caption=caption)})
    else:
        # Save the image with a caption
        save_path = os.path.join(save_dir, f"episode_{i}_image.jpg")
        plt.figure(figsize=(image_strip.shape[1] / 100, image_strip.shape[0] / 100))  # Adjust figure size based on image dimensions
        plt.imshow(image_strip)
        plt.title(caption)
        plt.axis('off')  # Turn off axes
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the image
        plt.close()  # Close the plot to free up memory
        print(f"Image saved: {save_path}")

# visualize action and state statistics
actions, states = [], []
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode['steps']:
        actions.append(step['action'].numpy())
        # states.append(step['observation']['state'].numpy())
actions = np.array(actions)
# states = np.array(states)
action_mean = actions.mean(0)
abs_action_mean = np.abs(actions).mean(0)
# state_mean = states.mean(0)

print(f"action demo: {actions[0]}")
print(f"action_mean demo: {action_mean}")

def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5*n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem+1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(f'Feature {elem+1} with Mean: {vector_mean[elem]:.6f}') # Title with the feature index and mean value
        # Optional: Add a text annotation to the plot above the title
        plt.text(0.5, 1.05, f"Feature {elem+1} Analysis", horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=10, color='blue')

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})
    save_path = os.path.join(save_dir, f"action_stats_image.jpg")
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close(fig)
    print(f"Image saved: {save_path}")

vis_stats(actions, abs_action_mean, 'action_stats')
# vis_stats(states, state_mean, 'state_stats')

if not render_wandb:
    plt.show()


