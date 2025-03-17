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


class CobotRldsDataset(tfds.core.GeneratorBasedBuilder):
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
                        'cam_high': tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format='jpeg',
                        ),
                        'cam_left_wrist': tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format='jpeg',
                        ),
                        'cam_right_wrist': tfds.features.Image(
                            shape=(480, 640, 3), dtype=np.uint8, encoding_format='jpeg',
                        )
                    }),
                    'action': tfds.features.Tensor(shape=(14,), dtype=np.float32,),
                    'base_action': tfds.features.Tensor(shape=(2,), dtype=np.float32,),
                    'qpos': tfds.features.Tensor(shape=(14,), dtype=np.float32,),
                    'qvel': tfds.features.Tensor(shape=(14,), dtype=np.float32,),
                    'instruction': tfds.features.Text(), # TODO: language_instruction instead of instruction
                    'expanded_instruction': tfds.features.Sequence(tfds.features.Text()),
                    'simplified_instruction': tfds.features.Sequence(tfds.features.Text()),
                    'terminate_episode': tfds.features.Tensor(shape=(), dtype=np.bool_),
                    # 'language_embedding': tfds.features.Tensor(shape=(512,), dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        train_data_dir = os.environ['TRAIN_DATA_DIR']
        assert train_data_dir is not None, "TRAIN_DATA_DIR is not set"
        split_dict = {
            'train': self._generate_examples(path=os.path.join(train_data_dir, 'episode_*.hdf5')),
        }
        val_data_dir = os.environ['VAL_DATA_DIR']
        if val_data_dir is not None:
            split_dict['val'] = self._generate_examples(path=os.path.join(val_data_dir, 'episode_*.hdf5'))
        return split_dict

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
                json_file = json.load(f_instr)
                instruction = json_file['instruction']
                simplified_instruction = json_file['simplified_instruction']
                expanded_instruction = json_file['expanded_instruction']
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
                    'expanded_instruction': expanded_instruction,
                    'simplified_instruction': simplified_instruction,
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

