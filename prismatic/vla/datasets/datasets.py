"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.cot_utils import CotTag, abbreviate_tag
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights


def reasoning_dropout(reasoning: str, dropout_prob: float) -> Tuple[str, str]:
    """Dropout reasoning tokens with probability `dropout_prob`."""
    if len(reasoning) == 0:
        return reasoning, ""

    reasoning_parts = reasoning.split("@")
    tags = [(reasoning_parts[i], reasoning_parts[i + 1]) for i in range(0, len(reasoning_parts), 2)]

    subset = np.random.rand(len(tags)) > dropout_prob

    subset_string = (
        "[" + ", ".join([abbreviate_tag(tag) for (tag, _), is_taken in zip(tags, subset) if is_taken]) + "]"
    )  # abbreviation

    excluded_tags = []

    if "EXCLUDE_TAGS" in os.environ:
        excluded_tags = os.environ["EXCLUDE_TAGS"].split(",")

    return (
        " ".join(
            [f"{tag[0]} {tag[1]}" for tag, is_taken in zip(tags, subset) if (is_taken and tag[0] not in excluded_tags)]
        ),
        subset_string,
    )


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    history_size: int = 0
    print_prompt_limit: int = 20
    reasoning_dropout_prob: float = 0.0
    empty_ret_for_none_reasoning: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][self.history_size]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][self.history_size])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"][self.history_size:]
        if "is_correction" in rlds_batch:
            is_correction = rlds_batch["is_correction"]
        else:
            is_correction = True
        if "reasoning" in rlds_batch:
            reasoning, subset = reasoning_dropout(
                rlds_batch["reasoning"].decode(),
                dropout_prob=self.reasoning_dropout_prob,
            )
        else:
            reasoning, subset = "", ""

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
 
        if lang.endswith("."):
            lang = lang[:-1]

        is_correction = True

        if 'reasoning' not in rlds_batch:
            if self.empty_ret_for_none_reasoning:
                return {}
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"}, # Explain why with {subset}."},
                {"from": "gpt", "value": f"{action_chunk_string}"},
            ]
        elif len(reasoning) > 0:
            if is_correction:
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"}, # Explain why with {subset}."},
                    # {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"{reasoning} {CotTag.ACTION.value} {action_chunk_string}"},
                ]
            else:
                conversation = [
                    {"from": "human", "value": f"What action should the robot take to {lang}?"}, # Explain why with {subset}."},
                    # {"from": "human", "value": f"What action should the robot take to {lang}?"},
                    {"from": "gpt", "value": f"{reasoning}"},
                ]
        else:
            if self.empty_ret_for_none_reasoning:
                return {}
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": f"{CotTag.ACTION.value} {action_chunk_string}"},
            ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Find the sequence of split tokens in labels
        split_tokens = [13, 3744, 29901]  # part of tokens for "\nOut: "
        for i in range(len(labels) - len(split_tokens) + 1):
            if labels[i:i+len(split_tokens)] == split_tokens:
                split_idx = i + len(split_tokens)  # Use the end of the matched sequence
                break
        else:
            # Fallback if sequence not found
            split_idx = len(labels) - action_chunk_len - 1

        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        labels[:split_idx] = IGNORE_INDEX # ignore the context tokens
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        
        if self.print_prompt_limit > 0:
            print("*" * 50)
            print(f"[EXAMPLE #{self.print_prompt_limit}]")
            if len(subset) > 0:
                print(f">>> Included tags:\n", subset)
            else:
                print(">>> No CoT tags included!")
            print(f">>> Conversation:\n", conversation)
            print(f">>> Prompt:\n", prompt_builder.get_prompt())
            print(f">>> Tokenized Prompt:\n", input_ids)
            print(f">>> Labels:\n", labels)
            print("*" * 50)
            self.print_prompt_limit -= 1

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        pixel_values = self.image_transform(img)

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=actions,
            metadata=dict(
                episode_id=rlds_batch["episode_id"],
                timestep=rlds_batch["observation"]["timestep"][0],
            )
        )
        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"][self.history_size:self.history_size+1]
            return_dict["proprio"] = proprio
        if self.history_size > 0:
            return_dict["history"] = dict()
            return_dict["history"]["length"] = self.history_size
            return_dict["history"]["pixel_values"] = []
            for t in range(self.history_size):
                _pixel_values = self.image_transform(Image.fromarray(rlds_batch["observation"]["image_primary"][t]))
                return_dict["history"]["pixel_values"].append(_pixel_values)
            return_dict["history"]["pixel_values"] = torch.stack(return_dict["history"]["pixel_values"])
            return_dict["history"]["action"] = rlds_batch["action"][:self.history_size]
            return_dict["history"]["proprio"] = rlds_batch["observation"]["proprio"][:self.history_size]
        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: Optional[int] = None,
        future_action_window_size: Optional[int] = None,
        enable_cot: bool = False,
        cot_tags: Optional[str] = None,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.enable_cot = enable_cot
        self.cot_tags = cot_tags
        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        load_camera_views = dict()
        for name, _ in mixture_spec:
            if "aloha" in name or "cobot" in name:
                load_camera_views[name] = ("primary", "left_wrist", "right_wrist")
            elif "libero" in name:
                load_camera_views[name] = ("primary", "wrist")
            elif "bridge" in name:
                load_camera_views[name] = ("primary",)
            else:
                load_camera_views[name] = ("primary",)

        per_dataset_kwargs, weights, MAX_ACTION_DIM = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )

        if window_size is not None:
            window_size = window_size
        else:
            window_size = 1

        if future_action_window_size is not None:
            future_action_window_size = future_action_window_size
        else:
            future_action_window_size = NUM_ACTIONS_CHUNK-1

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,              # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            max_action_dim=MAX_ACTION_DIM,
        )

        print("RLDS Config: ", rlds_config)

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config, enable_cot=self.enable_cot, cot_tags=self.cot_tags)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            ret = self.batch_transform(rlds_batch)
            if ret == {}:
                continue
            yield ret

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            enable_cot=self.enable_cot,
            cot_tags=self.cot_tags,
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
