"""
obs: {
    agent: {
        qpos: <Dataset, shape=(75, 9)>
        qvel: <Dataset, shape=(75, 9)>
    }
    extra: {
        is_grasped: <Dataset, shape=(75,)>
        tcp_pose: <Dataset, shape=(75, 7)>
        goal_pos: <Dataset, shape=(75, 3)>
    }
    sensor_param: {
        base_camera: {
            extrinsic_cv: <Dataset, shape=(75, 3, 4)>
            cam2world_gl: <Dataset, shape=(75, 4, 4)>
            intrinsic_cv: <Dataset, shape=(75, 3, 3)>
        }
    }
    sensor_data: {
        base_camera: {
            rgb: <Dataset, shape=(75, 128, 128, 3)>
        }
    }
}
actions: <Dataset, shape=(74, 8)>
terminated: <Dataset, shape=(74,)>
truncated: <Dataset, shape=(74,)>
success: <Dataset, shape=(74,)>
env_states: {
    actors: {
        table-workspace: <Dataset, shape=(75, 13)>
        cube: <Dataset, shape=(75, 13)>
        goal_site: <Dataset, shape=(75, 13)>
    }
    articulations: {
        panda: <Dataset, shape=(75, 31)>
    }
}
rewards: <Dataset, shape=(74,)>
"""

from typing import Iterator, Tuple, Any

import cv2
import glob
import numpy as np
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image

import h5py

class PandaRldsDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Text(doc='Unit ID.'),
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'language_instruction': tfds.features.Text(doc='Language instruction for the task.'),
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(480, 480, 3), dtype=np.uint8),
                        'state': tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        'wrist_image': tfds.features.Image(shape=(480, 480, 3), dtype=np.uint8),
                    }),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(paths=[
                                                    '/nvme_data/liangzhi/franka-dataset/process/pick_to_plate_multi-real/',
                                                    '/nvme_data/liangzhi/franka-dataset/process/pick_to_plate-sim_simple/'
                                                    ],
                                             max_items=[50, 1000]),
        }

    def _generate_examples(self, paths, max_items) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        for path, max_item in zip(paths, max_items):
            def _parse_example(data, task_dir, episode_id, file_path):
                lang = data["insruction"]
                states = data["proprio_state"][:]
                actions = data['actions'][:]

                states = states.astype(np.float32)
                actions = actions.astype(np.float32)
                num_steps = len(actions)

                episode = []
                for i in range(num_steps):
                    episode.append({
                        'observation': {
                            'image': data["front_rgb"][i].astype(np.uint8),
                            "state": states[i],
                            'wrist_image': data["wrist_rgb"][i].astype(np.uint8),
                        },
                        'language_instruction': lang,
                        'action': actions[i],
                    })

                # create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': task_dir,
                        'episode_id': episode_id,
                    }
                }
                return file_path, sample

            sample_idx = 0
            for episode_idx in os.listdir(path):
                if sample_idx >= max_item:
                    break
                sample_idx += 1
                episode_dir = os.path.join(path, episode_idx)
                if not os.path.isdir(episode_dir):
                    continue
                
                file_path = os.path.join(episode_dir, "data.npy")
                traj_data = np.load(file_path, allow_pickle=True).item()

                yield _parse_example(traj_data, path, episode_idx, file_path)



