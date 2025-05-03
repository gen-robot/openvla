from typing import Iterator, Tuple, Any
from pathlib import Path

import glob
import numpy as np
import tensorflow_datasets as tfds


class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = "../../../SimplerEnv/wandb/offline-run-20250404_230436-6rezxkqh/glob/data"

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
            'train': self._generate_examples(1150, spare=45),
            'val': self._generate_examples(45, start=1150),
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
                    'language_instruction': data['instruction'],
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

            return sample

        all_files = []
        path = Path(self.path)
        files = sorted(glob.glob(str(path / "*.npy")))
        if spare > 0:
            files = files[:-spare]
        if start > 0:
            start = min(start, len(files) - num_ep)
        files = files[start:start + num_ep]

        print(f"{len(files)}")

        all_files.extend(files)

        for idx, ep_path in enumerate(all_files):
            sample = _parse_example(ep_path)
            yield ep_path, sample

# mv -T ~/tensorflow_datasets/example_dataset ~/nfs/Project/RLVLA/thirdparty/datasets/widowx_pc1195
