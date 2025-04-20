import os
from pathlib import Path
from typing import Dict, Any, Optional

import datasets
from datasets import IterableDataset

from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE


_DESCRIPTION = """
RLDS dataset for vision-language-action models.
"""

_CITATION = """
"""

class RLDSDatasetBuilder(datasets.GeneratorBasedBuilder):
    """RLDS dataset for vision-language-action models."""

    VERSION = datasets.Version("1.0.0")
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            features=datasets.Features(
                {
                    "input_ids": datasets.Value("int64"),
                    "images": datasets.Image(),
                    "proprio": datasets.Sequence(datasets.Value("float32")),
                    "actions": datasets.Sequence(datasets.Value("float32")),
                    "text": datasets.Value("string"),
                }
            ),
        )
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "validation"},
            ),
        ]
    
    def _generate_examples(self, split):
        data_root_dir = Path(self.config.data_dir) if hasattr(self.config, "data_dir") else Path("datasets/rlds")
        dataset_name = self.config.dataset_name if hasattr(self.config, "dataset_name") else "aloha_scoop_x_into_bowl"
        
        # Configure the batch transform
        from transformers import AutoTokenizer, AutoImageProcessor
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        batch_transform = RLDSBatchTransform(
            None,  # action_tokenizer will be set later
            tokenizer, 
            image_transform=image_processor,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=False,
            use_proprio=True,
            history_size=0
        )
        
        # Create the dataset
        is_train = split == "train"
        dataset = RLDSDataset(
            data_root_dir=data_root_dir,
            data_mix=dataset_name,
            batch_transform=batch_transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=10000 if is_train else 1000,
            train=is_train,
            image_aug=is_train,
        )
        
        # Generate examples
        for idx, batch in enumerate(dataset):
            yield idx, {
                "input_ids": batch["input_ids"],
                "images": batch["images"],
                "proprio": batch.get("proprio", []),
                "actions": batch["actions"],
                "text": batch.get("text", ""),
            }