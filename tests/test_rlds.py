import tensorflow_datasets as tfds
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--task_name", type=str, required=True, help="Name of the RLDS task")
args = parser.parse_args()

print(f"Data directory: {args.data_dir}")
print(f"Task name: {args.task_name}")

dataset = tfds.load(args.task_name, split='train', data_dir=args.data_dir)
import pdb; pdb.set_trace()
for sample in dataset.take(1):
    for subsample in sample['steps'].take(1):
        print(subsample.keys())
        import pdb; pdb.set_trace()

        print(subsample['observation']['cam_high'].shape)