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

GRIPPER_SCALE = {
    "qpos": [0.066, 0.066],
    "action": [0.072, 0.072]
}



DATASET_STATS = {'state_min': np.array([-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0, 0.0]), 
                 'state_max': np.array([0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033, 0.03999999910593033]), 
                 'action_min': np.array([-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0]), 
                 'action_max': np.array([0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0]), 
                 'action_std': np.array([0.2199309915304184, 0.18780815601348877, 0.13044124841690063, 0.30669933557510376, 0.1340624988079071, 0.24968451261520386, 0.9589747190475464, 0.9827960729598999]), 
                 'action_mean': np.array([-0.00885344110429287, 0.5523102879524231, -0.007564723491668701, -2.0108158588409424, 0.004714342765510082, 2.615924596786499, 0.08461848646402359, -0.19301606714725494])}


TASK2LANG = {
    "StackCube-v1-1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCube-v1-2":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "StackCube-v1-3":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
}


class ManiSkillRldsDataset(tfds.core.GeneratorBasedBuilder):
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
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_primary': tfds.features.Image(
                            shape=(512, 512, 3), dtype=np.uint8, encoding_format='jpeg',
                        ),
                    }),
                    'qpos': tfds.features.Tensor(shape=(9,), dtype=np.float32,),
                    'action': tfds.features.Tensor(shape=(8,), dtype=np.float32,),
                    'lang_instruction': tfds.features.Text(
                        doc='Language instruction for the task.'
                    ),
                    'terminate_episode': tfds.features.Tensor(shape=(), dtype=np.bool_,),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'task_inner_index': tfds.features.Tensor(shape=(), dtype=np.int32,),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/nvme1n1/embodied_agent/maniskill_traj_pair_data_openvla'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(data, task_dir, traj_idx, lang):
            states = data['obs']['agent']['qpos'][:]
            actions = data['actions'][:]

            # normalize the states
            states = (states - DATASET_STATS['state_min']) / (DATASET_STATS['state_max'] - DATASET_STATS['state_min']) * 2 - 1
            actions = (actions - DATASET_STATS['action_min']) / (DATASET_STATS['action_max'] - DATASET_STATS['action_min']) * 2 - 1
            states = states.astype(np.float32)
            actions = actions.astype(np.float32)

            num_steps = len(actions)
            proc_index = traj_idx // 100
            episode_index = traj_idx % 100

            episode = []
            for i in range(num_steps):
                img_path = os.path.join(task_dir, 'motionplanning', f'{proc_index}', f'{episode_index}', f"{i}.png")
                with Image.open(img_path) as image:
                    img = np.array(image)
                    episode.append({
                        'observation': {
                            'image_primary': img.copy(),
                        },
                        'qpos': states[i],
                        'action': actions[i],
                        'lang_instruction': lang,
                        'terminate_episode': i == num_steps - 2
                    })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': task_dir,
                    'task_inner_index': traj_idx,
                }
            }
            example_id = os.path.basename(task_dir) + f'_{traj_idx}'
            return example_id, sample

        for task in os.listdir(path):
            task_dir = os.path.join(path, task)
            file_path = glob.glob(os.path.join(task_dir, 'motionplanning', '*.h5'))[0]
            lang = TASK2LANG[task]
            with h5py.File(file_path, "r") as f:
                trajs = f.keys() #  traj_0, traj_1,
                # sort by the traj number
                trajs = sorted(trajs, key=lambda x: int(x.split('_')[-1]))
                for traj_idx, traj in enumerate(trajs):
                    if task == 'PegInsertionSide-v1' and traj_idx > 400:
                        break
                    if traj_idx % 10 != 9:
                        continue
                    yield _parse_example(f[traj], task_dir, traj_idx, lang)

        # # for smallish datasets, use single-thread parsing
        # for sample in episode_paths:
        #     yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

