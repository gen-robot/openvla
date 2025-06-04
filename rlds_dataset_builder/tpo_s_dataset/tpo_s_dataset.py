from typing import Iterator, Tuple, Any
from pathlib import Path

import glob
import numpy as np
import tensorflow_datasets as tfds
import cv2
import re

class TPOSuccessDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = [
            
            {"path": "../../../../videos/dpo/merged_002000_PutOnPlateInScene25Carrot-v1/train_carrots_num_1/20250509_125810",
             "filter": False},
            # {"path": "../../../../videos/datasets_mp/PutOnPlateInScene25Carrot-v1/20250421_195812/data",
            #  "filter": False},
        ]
        self.pair_per_ep = 1

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
            'train': self._generate_examples(num_ep=10, spare=3),
            'val': self._generate_examples(num_ep=3, start=10),
        }

    def _generate_examples(self, num_ep, spare=0, start=0) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, use_filter):
            data = np.load(episode_path, allow_pickle=True).tolist()

            is_image_encode = data.get("is_image_encode", False)
            ins = data['instruction']
            ins = ins.tolist()[0] if isinstance(ins, np.ndarray) else ins
            actions = np.array(data["action"])
            import pdb;pdb.set_trace()
            images = np.asarray([np.asarray(img) for img in data["image"]])

            if use_filter:
                mask = filter_small_actions(data["action"])
                actions = actions[mask]
                images = images[mask]
                num_filtered = mask.shape[0] - mask.sum()
                print(f"Filtered {num_filtered}/{mask.shape[0]} actions")
            else:
                num_filtered = 0

            episode = []
            success_count = 0
            for i in range(len(actions)):
                if is_image_encode:
                    image = np.array(cv2.imdecode(np.frombuffer(images[i], np.uint8), cv2.IMREAD_COLOR))
                else:
                    image = np.asarray(images[i])

                episode.append({
                    'observation': {
                        'image': image,
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

        all_files = []
        rank_pattern = re.compile(r"-rank_(\d+)")
        top_rank_num = 1 # get top num rank files in the dataset to generate new dataset
        for task in self.tasks:
            path = Path(task["path"])
            filter = task["filter"]
            files = sorted(glob.glob(str(path / "*.npy")))
            task_files = []
            episode_dirs = [f for f in path.iterdir() if f.is_dir() and f.name.startswith("episode_")]
            for episode_dir in episode_dirs:
                files = glob.glob(str(episode_dir / f"*.npy"))
                ranked_files = sorted(
                    files,
                    key=lambda f: int(rank_pattern.search(Path(f).name).group(1)) if rank_pattern.search(Path(f).name) else 0, # for the number is uint, 0 is the smallest
                    reverse=True, # False indicates asendinig order, True indicates descending order 
                )
                top_ranked_files = ranked_files[:top_rank_num]
                for file in top_ranked_files:
                    task_files.append(file)

            if spare > 0:
                task_files = task_files[:-spare]
            if start > 0:
                start = min(start, len(task_files) - num_ep)
            task_files = task_files[start:start + num_ep]
            assert len(task_files) == num_ep
            print(f"{task}: {len(task_files)}")
            all_files.extend([(f, filter) for f in task_files])
        for idx, ep_path in enumerate(all_files):
            sample = _parse_example(*ep_path)
            print(ep_path[0])
            yield ep_path, sample

def filter_small_actions():
    pass