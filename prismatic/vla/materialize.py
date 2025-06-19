"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, Optional, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.models.backbones.vision.base_vision import WrapSequenceImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS, ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: Union[ImageTransform, WrapSequenceImageTransform],
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    action_tokenizer: str = "action_tokenizer",
    image_window_size: Optional[int] = 1,
    use_wrist_image: Optional[bool] = False,
    future_action_window_size: Optional[int] = 0,
    enable_cot: bool = False,
    cot_tags: Optional[str] = None,
    reasoning_dropout_prob: float = 0.0,
    base_model_type: str = "prismatic",
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS[action_tokenizer](tokenizer)

    # get the future action window needed from the tokenizer
    future_action_window_size = max(action_tokenizer.required_future_horizon, future_action_window_size)

    # get the observation history from the image_transform (only needed if its a WrapSequence transform)
    # if isinstance(image_transform, WrapSequenceImageTransform):
    #     if use_wrist_image:
    #         # expects groupings of two in image sequence len
    #         assert image_transform.sequence_len % 2 == 0, "With wrist images, image transform must expect 2N images!"
    #         image_window_size = max(image_transform.sequence_len // 2, image_window_size)
    #     else:
    #         image_window_size = max(image_transform.sequence_len, image_window_size)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token=predict_stop_token,
        image_window_size=image_window_size,
        use_wrist_image=use_wrist_image,
        future_action_window_size=future_action_window_size,
        reasoning_dropout_prob=reasoning_dropout_prob,
        base_model_type=base_model_type,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        window_size=image_window_size,
        future_action_window_size=future_action_window_size,
        use_wrist_image=use_wrist_image,
        enable_cot=enable_cot,
        cot_tags=cot_tags
    )

    return dataset, action_tokenizer, collator
