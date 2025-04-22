import tensorflow_datasets as tfds
import argparse

import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--task_name", type=str, required=True, help="Name of the RLDS task")
parser.add_argument("--output_dir", type=str, default="videos", help="Path to the output directory")
args = parser.parse_args()

print(f"Data directory: {args.data_dir}")
print(f"Task name: {args.task_name}")

os.makedirs(args.output_dir, exist_ok=True)

dataset = tfds.load(args.task_name, split='train', data_dir=args.data_dir)

for sample in dataset.take(1):
    for step in sample['steps']:
        language_instruction = step['task']['language_instruction'].numpy().decode('utf-8')
        obs = step['observation']
        for key in obs.keys():
            if 'image' in key:
                image = obs[key]
                image = Image.fromarray(image)
                image.save(os.path.join(args.output_dir, f"{language_instruction}_{key}.png"))
