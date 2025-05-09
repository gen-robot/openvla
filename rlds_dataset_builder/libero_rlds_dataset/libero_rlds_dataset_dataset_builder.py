"""
FeaturesDict({
    'episode_metadata': FeaturesDict({
        'episode_id': string,
        'file_path': Text(shape=(), dtype=string),
    }),
    'steps': Dataset({
        'action': Tensor(shape=(7,), dtype=float32),
        'discount': Scalar(shape=(), dtype=float32),
        'is_first': Scalar(shape=(), dtype=bool),
        'is_last': Scalar(shape=(), dtype=bool),
        'is_terminal': Scalar(shape=(), dtype=bool),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(256, 256, 3), dtype=uint8),
            'joint_state': Tensor(shape=(7,), dtype=float32),
            'state': Tensor(shape=(8,), dtype=float32),
            'wrist_image': Image(shape=(256, 256, 3), dtype=uint8),
        }),
        'reward': Scalar(shape=(), dtype=float32),
    }),
})
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
    "OpenDrawerAndPutBowl": "open the top drawer and put the bowl inside",
    "OpenMiddleDrawer": "open the middle drawer of the cabinet",
    "PushPlate": "push the plate to the front of the stove",
    "PutBowlOnDrawer": "put the bowl on top of the cabinet",
    "PutBowlOnPlate": "put the bowl on the plate",
    "PutBowlOnStove": "put the bowl on the stove",
    "PutCheese": "put the cream cheese in the bowl",
    "TurnOnStove": "turn on the stove",
    "AlphabetSoup": "pick up the alphabet soup and place it in the basket",
    "CreamCheese": "pick up the cream cheese and place it in the basket",
    "SaladDressing": "pick up the salad dressing and place it in the basket",
    "BbqSauce": "pick up the bbq sauce and place it in the basket",
    "Ketchup": "pick up the ketchup and place it in the basket",
    "TomatoSauce": "pick up the tomato sauce and place it in the basket",
    "Butter": "pick up the butter and place it in the basket",
    "Milk": "pick up the milk and place it in the basket",
    "ChocolatePudding": "pick up the chocolate pudding and place it in the basket",
    "OrangeJuice": "pick up the orange juice and place it in the basket",
    "MokaPots": "put both moka pots on the stove",
    "ACB": "put both the alphabet soup and the cream cheese box in the basket",
    "ATB": "put both the alphabet soup and the tomato sauce in the basket",
    "CBB": "put both the cream cheese box and the butter in the basket",
    "BowlDrawer": "put the black bowl in the bottom drawer of the cabinet and close it",
    "TwoMugPlate": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "MugChocolate": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
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
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Text(doc='Unit ID.'),
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
                'steps': tfds.features.Dataset({
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'is_correction': tfds.features.Tensor(shape=(), dtype=np.bool_,),
                    'language_instruction': tfds.features.Text(doc='Language instruction for the task.'),
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
                        'joint_state': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                        'state': tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        'wrist_image': tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8),
                    }),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/nvme_data/liangzhi/datasets/libero_correction/libero_10-total/'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(data, task_dir, traj_idx, lang, task_id):
            states = data['obs']['agent']['state'][:]
            joint_states = data['obs']['agent']['joint_state'][:]
            is_correction = data['is_correction'][:]
            actions = data['actions'][:]

            states = states.astype(np.float32)
            joint_states = joint_states.astype(np.float32)
            is_correction = is_correction.astype(bool)
            actions = actions.astype(np.float32)

            num_steps = len(actions)
            proc_index = traj_idx // 100
            episode_index = traj_idx % 100

            episode = []
            for i in range(num_steps):
                full_img_path = os.path.join(task_dir, 'full', f'{proc_index}', f'{episode_index}', f"{i}.png")
                wrist_img_path = os.path.join(task_dir, 'wrist', f'{proc_index}', f'{episode_index}', f"{i}.png")

                with Image.open(full_img_path) as image:
                    np_full_img = np.array(image)
                    np_full_img = np_full_img[::-1, ::-1]
                    full_img = np_full_img.copy()
                
                with Image.open(wrist_img_path) as image:
                    np_wrist_img = np.array(image)
                    np_wrist_img = np_wrist_img[::-1, ::-1]
                    wrist_img = np_wrist_img.copy()

                episode.append({
                    'observation': {
                        'image': full_img,
                        'joint_state': joint_states[i],
                        'state': states[i],
                        'wrist_image': wrist_img,
                    },
                    'is_correction': is_correction[i],
                    'action': actions[i],
                    'language_instruction': lang,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': task_dir,
                    'episode_id': f"{traj_idx}-{num_steps}-{task_id}",
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
                    yield _parse_example(f[traj], task_dir, traj_idx, lang, task)

        # # for smallish datasets, use single-thread parsing
        # for sample in episode_paths:
        #     yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

# open the middle drawer of the cabinet