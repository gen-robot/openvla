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
            'train': self._generate_examples(5100, spare=20),
            'val': self._generate_examples(20, start=5100),
        }

    def _generate_examples(self, num_ep, spare=0, start=0) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            data = np.load(episode_path, allow_pickle=True).tolist()

            # prepare data
            ins = data['instruction']
            ins = ins.tolist()[0] if isinstance(ins, np.ndarray) else ins
            actions = data["action"]
            images = np.asarray([np.asarray(img) for img in data["image"]])

            mask = filter_small_actions(data["action"])
            actions = actions[mask]
            images = images[mask]
            num_filtered = mask.shape[0] - mask.sum()
            print(f"Filtered {num_filtered}/{mask.shape[0]} actions")

            episode = []
            success_count = 0
            for i in range(len(actions)):
                episode.append({
                    'observation': {
                        'image': images[i],
                    },
                    'action': actions[i],
                    'language_instruction': ins,
                })

                if data["info"][i]["success"]:
                    success_count += 1
                else:
                    success_count = 0

                if success_count >= 6:
                    break

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            return sample, num_filtered

        tasks = [
            {"name": "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Carrot-v1/20250424_200357/data"},
            {"name": "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Carrot-v1/20250427_170312/data"},
        ]

        all_files = []
        for task in tasks:
            path = Path(task["name"])

            files = sorted(glob.glob(str(path / "*.npy")))
            all_files.extend(files)
            print(f"Found {len(files)} files in {path}")

        if spare > 0:
            all_files = all_files[:-spare]
        if start > 0:
            start = min(start, len(all_files) - num_ep)
        all_files = all_files[start:start + num_ep]
        assert len(all_files) == num_ep

        print(f"{len(all_files)}")


        num_filtered_total = 0
        for idx, ep_path in enumerate(all_files):
            sample, num_filtered = _parse_example(ep_path)
            num_filtered_total += num_filtered
            yield ep_path, sample

        print(f"Total filtered {num_filtered_total} actions")

# tfds build --overwrite

# "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Carrot-v1/20250424_200357/data"
# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/s1024c5
# "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Carrot-v1/20250427_170312/data"
# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/s5120c5

# "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Carrot-v1/20250424_213036/data"
# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/s1024c1

# "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Instruct-v1/20250425_201123/data"
# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/s1024i5

# "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Instruct-v1/20250425_205230/data"
# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/s1024i1

# "../../../ManiSkill/videos/datasets_mp/PutOnPlateInScene25Overlay-v1/20250425_213354/data"
# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/s1024o5
