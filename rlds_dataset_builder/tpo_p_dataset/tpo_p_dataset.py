from typing import Iterator, Tuple, Any
from pathlib import Path

import glob
import numpy as np
import tensorflow_datasets as tfds

import numpy as np


def filter_small_actions(actions, pos_thresh=0.01, rot_thresh=0.06, check_gripper=True):
    actions = np.asarray(actions)
    N = actions.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6]

        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)

        if pos_thresh is None and rot_thresh is None:
            is_valid = True
        elif pos_thresh is None:
            is_valid = (rot_movement > rot_thresh)
        elif rot_thresh is None:
            is_valid = (pos_movement > pos_thresh)
        else:
            is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)

        # Preserve gripper toggle events (e.g., from -1 to 1 or vice versa)
        if check_gripper and i > 0 and actions[i - 1][6] != gripper:
            is_valid = True

        valid_mask[i] = is_valid

    return valid_mask

class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # data = np.load(episode_path, allow_pickle=True).tolist()
            data = np.load(episode_path, allow_pickle=True).tolist()

            # prepare data
            ins = data['instruction']
            ins = ins.tolist()[0] if isinstance(ins, np.ndarray) else ins
            actions = data["action"]
            images = np.asarray([np.asarray(img) for img in data["image"]])

            episode = []
            for i in range(len(actions)):
                episode.append({
                    'observation': {
                        'image': images[i],
                    },
                    'action': actions[i],
                    'language_instruction': ins,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return sample

        path = Path("../../../SimplerEnv/wandb/tpo/spc148f-tpo_2")
        files = sorted(glob.glob(str(path / "*.npy")))
        print(f"Found {len(files)} files in {path}")

        all_files = []

        for idx in range(256):
            select_run = ""
            select_reward = -100
            for t in range(4):
                fns = [f for f in files if f"data_{idx * 4 + t:0>4d}-" in f]

                fn = fns[0]

                g = "-g_True" in fn
                cg = "-cg_True" in fn
                s = "-s_True" in fn
                reward = g * 0.1 + cg * 0.1 + (g & s) * 1.0

                if reward > select_reward:
                    select_reward = reward
                    select_run = fn

            print(f"select run: {select_run}")

            all_files.append(select_run)

        print(f"{len(all_files)}")


        for idx, ep_path in enumerate(all_files):
            sample = _parse_example(ep_path)
            yield ep_path, sample


# tfds build --overwrite
# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/spc148f_tpo_p_2

