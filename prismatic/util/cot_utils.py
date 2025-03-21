"""
Example reasoning: /nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy_43_0 TASK:@Move the wooden arch onto the table.@PLAN:@Reach for the wooden arch. Grasp the wooden arch. Move the wooden arch to the table. Drop the wooden arch onto the table.@VISIBLE OBJECTS:@wooden blocks [150, 4, 188, 100]@SUBTASK REASONING:@The wooden arch is the object that needs to be moved, so the first step is to reach for it.@SUBTASK:@Reach for the wooden arch.@MOVE REASONING:@The arm is already in a good position to reach for the wooden arch.@MOVE:@stop@GRIPPER POSITION:@[97, 45, 97, 45, 89, 52, 83, 58, 82, 57]

Example reasoning: /nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy_43_0 
TASK: Move the wooden arch onto the table.
PLAN:
  - Reach for the wooden arch.
  - Grasp the wooden arch.
  - Move the wooden arch to the table.
  - Drop the wooden arch onto the table.
VISIBLE OBJECTS: wooden blocks [150, 4, 188, 100]
SUBTASK REASONING: The wooden arch is the object that needs to be moved, so the first step is to reach for it.
SUBTASK: Reach for the wooden arch.
MOVE REASONING: The arm is already in a good position to reach for the wooden arch.
MOVE: stop
GRIPPER POSITION: [97, 45, 97, 45, 89, 52, 83, 58, 82, 57]
"""


import enum
import torch
import tensorflow as tf


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_cot_database_keys():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }



def make_tf_hash_table(raw_dict):
    print("Building the reasoning dict...")
    keys = []
    values = []

    def reasoning_dict_to_str(d):
        tags = get_cot_tags_list()[:-1]  # exclude ACTION
        database_keys = get_cot_database_keys()
        reasoning_parts = [(tag, d[database_keys[tag]]) for tag in tags]

        return "@".join(f"{tag}@{part}" for tag, part in reasoning_parts)

    has_reasoning = [0, 0]

    for file_name in raw_dict.keys():
        for episode_id in raw_dict[file_name].keys():
            if "reasoning" not in raw_dict[file_name][episode_id].keys():
                has_reasoning[0] += 1
                continue
            else:
                has_reasoning[1] += 1

            for i in raw_dict[file_name][episode_id]["reasoning"].keys():
                keys.append(file_name + "_" + str(episode_id) + "_" + i)
                reasoning_dict = raw_dict[file_name][episode_id]["reasoning"][i]

                gripper_lookahead_n = 5  # list this many future positions of the gripper
                trajectory_features = raw_dict[file_name][episode_id]["features"]

                reasoning_dict["gripper"] = ""
                if "gripper_position" in trajectory_features.keys():
                    if trajectory_features["gripper_position"] is not None:
                        if 0 <= int(i) < len(trajectory_features["gripper_position"]):
                            future_positions = []
                            for j in range(gripper_lookahead_n):
                                if int(i) + j < len(trajectory_features["gripper_position"]):
                                    future_positions += trajectory_features["gripper_position"][int(i) + j]
                                else:
                                    future_positions += future_positions[-2:]

                            reasoning_dict["gripper"] = str(future_positions)

                reasoning_dict["bboxes"] = ""
                if "bboxes" in trajectory_features.keys():
                    if trajectory_features["bboxes"] is not None:
                        if 0 <= int(i) < len(trajectory_features["bboxes"]):
                            if len(trajectory_features["bboxes"][int(i)]) > 0:
                                boxes_list = trajectory_features["bboxes"][int(i)]
                                reasoning_dict["bboxes"] = ", ".join(
                                    [f"{name} {box!s}" for prob, name, box in boxes_list]
                                )

                values.append(reasoning_dict_to_str(reasoning_dict))

    print("Example reasoning:", keys[0], values[0])
    print("Reasoning presence statistics [# has not, # has]:", has_reasoning)

    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), 
        default_value="")


def get_cot_masks(tokens, tags, llm_tokenizer):
    tag_tokens = dict()

    for tag in tags:
        encoded_tags = llm_tokenizer.encode_plus(tag, return_tensors="pt")
        tag_ids = encoded_tags["input_ids"][0]
        tag_tokens[tag] = tag_ids[1:].to(tokens.device)

    tag_masks = dict()
    prev_tag = None
    prev_pos = 0

    def make_mask(a, b):
        mask = torch.zeros_like(tokens)
        mask[a:b] = 1
        return mask

    # find position of a small list of tokens in the larger list of tokens
    for i in range(len(tokens) - 1):
        for tag, tag_ids in tag_tokens.items():
            if i + len(tag_ids) > len(tokens):
                continue

            if torch.all(tokens[i : i + len(tag_ids)] == tag_ids):
                tag_masks[prev_tag] = make_mask(prev_pos, i)
                prev_tag = tag
                prev_pos = i + len(tag_ids)

    tag_masks[prev_tag] = make_mask(prev_pos, len(tokens))
    
    for tag in tags:
        if tag not in tag_masks:
            tag_masks[tag] = make_mask(0, 0)

    return tag_masks


def compute_cot_accuracy(predicted_token_ids, ground_truth_token_ids, llm_tokenizer):
    """
    Compute the accuracy for each CoT tag.
    Args:
        predicted_token_ids: tensor of shape (batch_size, #tokens)
        ground_truth_token_ids: tensor of shape (batch_size, #tokens)
        llm_tokenizer: tokenizer, by default it's the LlamaTokenizerFast
    Returns:
        metrics: dictionary of accuracy for each tag
    """
    tags = get_cot_tags_list()[:-1]  # exclude ACTION
    metrics = {}
    
    def get_batched_masks(tokens, tags):
        final_masks = {tag: [] for tag in tags}

        for group in tokens:
            group_masks = get_cot_masks(group, tags, llm_tokenizer)
            for tag in tags:
                final_masks[tag].append(group_masks[tag])

        for tag in tags:
            final_masks[tag] = torch.stack(final_masks[tag], dim=0)

        return final_masks
    
    final_pred_masks = get_batched_masks(predicted_token_ids, tags)
    final_gt_masks = get_batched_masks(ground_truth_token_ids, tags)

    # Compute accuracy for each tag
    for tag in tags:
        correct_tags = [0, 0]
        
        for reasoning_pred, mask_pred, reasoning_gt, mask_gt in zip(
            predicted_token_ids, final_pred_masks[tag], ground_truth_token_ids, final_gt_masks[tag]
        ):
            tag_pred = torch.masked_select(reasoning_pred, mask_pred.bool())
            tag_gt = torch.masked_select(reasoning_gt, mask_gt.bool())
            
            max_size = max(len(tag_pred), len(tag_gt))
            tag_pred = torch.nn.functional.pad(tag_pred, (0, max_size - len(tag_pred)))
            tag_gt = torch.nn.functional.pad(tag_gt, (0, max_size - len(tag_gt)))

            correct_tags[0] += (tag_pred == tag_gt).sum().float()
            correct_tags[1] += len(tag_gt)

        if correct_tags[1] > 0:
            tag_accuracy = correct_tags[0] / correct_tags[1]
            metrics.update(**{f"reasoning/{tag[:-1].lower()}_tag_accuracy": tag_accuracy})

    return metrics
