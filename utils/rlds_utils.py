import tensorflow_datasets as tfds

import numpy as np


def get_data_from_rlds(data_dir: str, dataset_name: str, split: str = "train", image_key: str = "images"):
    """
    Get images from RLDS dataset.
    """
    dataset = tfds.load(dataset_name, data_dir=data_dir, split=split)
    data = dict()
    for sample in dataset.take(1):
        import pdb; pdb.set_trace()
        data['episode_id'] = sample['episode_metadata']['episode_id'].numpy()
        data['file_path'] = sample['episode_metadata']['file_path'].numpy().decode()
        for i, step in enumerate(sample['steps']):
            if i == 0:
                data['language_instruction'] = step['language_instruction'].numpy().decode()
                data['action'] = []
                data['observation'] = {}
            data['action'].append(step['action'].numpy())
            for k, v in step['observation'].items():
                if k not in data['observation']:
                    data['observation'][k] = []
                data['observation'][k].append(v.numpy())
    return data

if __name__ == "__main__":
    data = get_data_from_rlds("/home/gaofeng/arm_ws/openvla/datasets", "bridge_orig", split="train", image_key="images")
    import pdb; pdb.set_trace()

