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

import h5py

GRIPPER_SCALE = {
    "qpos": [0.066, 0.066],
    "action": [0.072, 0.072]
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
                    'obs': tfds.features.FeaturesDict({
                        'agent': tfds.features.FeaturesDict({
                            'qpos': tfds.features.Tensor(shape=(9), dtype=np.float32,),
                            'qvel': tfds.features.Tensor(shape=(9), dtype=np.float32,),
                        }),
                        'extra': tfds.features.FeaturesDict({
                            'is_grasped': tfds.features.Tensor(shape=(1,), dtype=np.bool_),
                            'tcp_pose': tfds.features.Tensor(shape=(7), dtype=np.float32,),
                            'goal_pos': tfds.features.Tensor(shape=(3), dtype=np.float32,),
                        }),
                        'sensor_param': tfds.features.FeaturesDict({
                            'base_camera': tfds.features.FeaturesDict({
                                'extrinsic_cv': tfds.features.Tensor(shape=(3, 4), dtype=np.float32,),
                                'cam2world_gl': tfds.features.Tensor(shape=(4, 4), dtype=np.float32,),
                                'intrinsic_cv': tfds.features.Tensor(shape=(3, 3), dtype=np.float32,),
                            }),
                        }),
                        'sensor_data': tfds.features.FeaturesDict({
                            'base_camera': tfds.features.Image(
                                shape=(512, 512, 3), dtype=np.uint8, encoding_format='jpeg',
                            ),
                        }),
                    }),
                    'action': tfds.features.Tensor(shape=(8,), dtype=np.float32,),
                    'terminated': tfds.features.Tensor(shape=(1,), dtype=np.bool_,),
                    'truncated': tfds.features.Tensor(shape=(1,), dtype=np.bool_,),
                    'success': tfds.features.Tensor(shape=(1,), dtype=np.bool_,),
                    'env_states': tfds.features.FeaturesDict({
                        'actors': tfds.features.FeaturesDict({
                            'table-workspace': tfds.features.Tensor(shape=(13,), dtype=np.float32,),
                            'cube': tfds.features.Tensor(shape=(13,), dtype=np.float32,),
                            'goal_site': tfds.features.Tensor(shape=(13,), dtype=np.float32,),
                        }),
                    }),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/nvme_data/embodied_agent/cobot_data/new_open_drawer/episode_*.hdf5'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            f = h5py.File(episode_path, 'r')
            with open(os.path.join(
                os.path.dirname(episode_path), 
                'expanded_instruction_gpt-4-turbo.json'), 'r'
            ) as f_instr:
                instruction = json.load(f_instr)['instruction']
            # Remove the first few still steps
            EPS = 1e-2
            num_episodes = f['action'].shape[0]
            qpos = f['observations']['qpos'][:]
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            def parse_img(key, step, compressed=True):
                if compressed:
                    return cv2.imdecode(np.frombuffer(
                        f['observations']['images'][key][step], np.uint8), cv2.IMREAD_COLOR)
                else:
                    return f['observations']['images'][key][step]
                
            def process_qpos(qpos, step):
                return qpos[step] / np.array([
                    1, 1, 1, 1, 1, 1, GRIPPER_SCALE["qpos"][0], 
                    1, 1, 1, 1, 1, 1, GRIPPER_SCALE["qpos"][1]
                ])
            
            def process_action(action, step):
                return action[step] / np.array([
                    1, 1, 1, 1, 1, 1, GRIPPER_SCALE["qpos"][0], 
                    1, 1, 1, 1, 1, 1, GRIPPER_SCALE["qpos"][1]
                ])
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(first_idx-1, num_episodes-1):
                # print("check img size", parse_img('cam_high', i, f.attrs.get('compress', True)).shape)
                # import pdb; pdb.set_trace()
                episode.append({
                    'observation': {
                        'cam_high': parse_img('cam_high', i, f.attrs.get('compress', True)),
                        'cam_left_wrist': parse_img('cam_left_wrist', i, f.attrs.get('compress', True)),
                        'cam_right_wrist': parse_img('cam_right_wrist', i, f.attrs.get('compress', True)),
                    },
                    'qpos': process_qpos(f['observations']['qpos'], i).astype(np.float32),
                    'qvel': f['observations']['qvel'][i].astype(np.float32),
                    'action': process_qpos(f['observations']['qpos'], i+1).astype(np.float32), #process_action(f['action'], i).astype(np.float32),
                    'base_action': f['base_action'][i].astype(np.float32),
                    'instruction': instruction,
                    'terminate_episode': i == num_episodes - 2,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

