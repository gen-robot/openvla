from typing import Iterator, Tuple, Any
from pathlib import Path

import glob
import numpy as np
import tensorflow_datasets as tfds
from simpler_env import SIMPLER_ROOT_DIR


class PandaSimplerSpoonDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': """panda simpler spoon: 180+20 traj""",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = SIMPLER_ROOT_DIR+"/videos/"
        self.tasks = [
            "scp/PandaPutSpoonOnTableClothInScene-v1/20250323_142308/data",
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
        return {
            'train': self._generate_examples(180, spare=20),
            'val': self._generate_examples(20, start=180),
        }

    def _generate_examples(self, num_ep, spare=0, start=0) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            data = np.load(episode_path, allow_pickle=True).tolist()

            episode = []
            success_count = 0
            for i in range(len(data["action"])):

                episode.append({
                    'observation': {
                        'image': np.asarray(data["image"][i]),
                    },
                    'action': data["action"][i],
                    'language_instruction': data['instruction'][0],
                })

                if data["info"][i]["success"][0]:
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

            return sample

        all_files = []
        for task in self.tasks: # for every task
            path = Path(self.path) / task
            files = sorted(glob.glob(str(path / "*.npy")))
            if spare > 0:
                files = files[:-spare]
            if start + num_ep > len(files):
                start = len(files) - num_ep

            files = files[start:start + num_ep]

            print(f"{task}: {len(files)}")

            all_files.extend(files)

        for idx, ep_path in enumerate(all_files):
            sample = _parse_example(ep_path)
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
