from typing import Iterator, Tuple, Any
from pathlib import Path
import glob
import numpy as np
import tensorflow_datasets as tfds
from simpler_env import SIMPLER_ROOT_DIR
import cv2
import random
from third_party.openvla.rlds_dataset_builder.utils import filter_small_actions


class PandaSimplerSpoonDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.1.0')
    RELEASE_NOTES = {
        '1.1.0': """panda simpler spoon with 125 traj, with filter, 0,9 ratio. """,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = SIMPLER_ROOT_DIR+"/videos/"
        self.tasks = [
            "scp/sft_125/panda/spoon/data",
        ]
        assert len(self.tasks)==1, "task_num is false."

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format='jpeg',
                            doc='Observation image.'
                        ),
                    }),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32, ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    # actually, we have the number of tasks times the number of episodes examples in _split generators
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # Use _generate_examples to generate train and eval splits
        split_ratio = 0.9
        apply_action_filter = True
        return {
            'train': self._generate_examples(split_ratio=split_ratio, is_train=True, apply_action_filter=apply_action_filter),
            'val': self._generate_examples(split_ratio=split_ratio, is_train=False, apply_action_filter=apply_action_filter),
        }

    def _generate_examples(self, split_ratio=0.9, is_train=True, apply_action_filter=True) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, apply_action_filter=True):
            data = np.load(episode_path, allow_pickle=True).tolist()

            actions = np.array(data["action"])
            images = data["image"]
            is_image_encode = data.get("is_image_encode", False)

            if apply_action_filter:
                # === Filter small actions and get valid indices ===
                filtered_actions, valid_mask = filter_small_actions(actions, pos_thresh=0.0015, rot_thresh=0.0015, check_gripper=True)
                # === Filter images using the same mask ===
                filtered_images = [images[i] for i in range(len(images)) if valid_mask[i]]
                print(f"remove minor action numbers: {len(actions)-len(filtered_actions)}")
            else:
                filtered_actions = actions
                filtered_images = images

            episode = []
            for i in range(len(filtered_actions)):
                if is_image_encode:
                    image = np.array(cv2.imdecode(np.frombuffer(filtered_images[i], np.uint8), cv2.IMREAD_COLOR))
                else:
                    image = np.asarray(filtered_images[i])
                episode.append({
                    'observation': {
                        'image': image,
                    },
                    'action': filtered_actions[i].astype(np.float32),
                    'language_instruction': data['instruction'][0],
                })

            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return sample

        # Read all files, and shuffle them
        all_files = []
        for task in self.tasks:
            path = Path(self.path) / task
            files = sorted(glob.glob(str(path / "*.npy")))
            all_files.extend(files)

        # Calculate the split index based on the ratio
        split_idx = int(len(all_files) * split_ratio)  # Example: 0.9 for training and 0.1 for validation
        train_files = all_files[:split_idx]
        eval_files = all_files[split_idx:]

        if is_train:
            # Yield examples for training split
            for ep_path in train_files:
                sample = _parse_example(ep_path, apply_action_filter)
                yield ep_path, sample
        else:
            # Yield examples for validation split
            for ep_path in eval_files:
                sample = _parse_example(ep_path, apply_action_filter)
                yield ep_path, sample

        # # create list of all examples
        # episode_paths = glob.glob(path)

        # # for smallish datasets, use single-thread parsing
        # # for sample in episode_paths:
        # #     yield _parse_example(sample)

        # # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
