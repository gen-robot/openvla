"""
Example reasoning: /nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy_43_0 TASK:@Move the wooden arch onto the table.@PLAN:@Reach for the wooden arch. Grasp the wooden arch. Move the wooden arch to the table. Drop the wooden arch onto the table.@VISIBLE OBJECTS:@wooden blocks [150, 4, 188, 100]@SUBTASK REASONING:@The wooden arch is the object that needs to be moved, so the first step is to reach for it.@SUBTASK:@Reach for the wooden arch.@MOVE REASONING:@The arm is already in a good position to reach for the wooden arch.@MOVE:@stop@GRIPPER POSITION:@[97, 45, 97, 45, 89, 52, 83, 58, 82, 57]

Example reasoning: /nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy_43_0 
TASK: The task remaining is to pick up the chocolate pudding and place it in the basket. 
PLAN: The plan is to move to the basket, and release the pudding. 
VISIBLE OBJECTS: basket [20, 107, 75, 162], bottle [109, 137, 128, 181], bottle [107, 100, 120, 128], robot [13, 0, 88, 98], bottle [181, 128, 200, 171].
SUBTASK REASONING: The object has to descend and rotate for correct placement. 
SUBTASK: The current subtask is to move to the basket. 
RELEVANT OBJECTS: [pudding, basket]. 
MOVE REASONING: Movement down and clockwise ensures good positioning. 
MOVE: move down, rotate clockwise. 
GRIPPER POSITION: [52, 62, 46, 73, 45, 81, 45, 83, 44, 82].
"""
import enum
import os
import textwrap
from typing import Dict, List

import prismatic
import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from .img_utils import draw_2d_points, draw_bboxes
    from ..models import load
except ImportError:
    from prismatic.util.img_utils import draw_2d_points, draw_bboxes
    from prismatic.models import load

class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    RELEVANT_OBJECTS = "RELEVANT OBJECTS:"
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
        CotTag.RELEVANT_OBJECTS.value,
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
        CotTag.RELEVANT_OBJECTS.value: "relevant_objects",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }

def get_inverse_cot_database_keys():
    forward_map = get_cot_database_keys()
    return {v: k for k, v in forward_map.items()}


def make_tf_hash_table(raw_dict, cot_tags=None):
    print("Building the reasoning dict...")
    keys = []
    values = []

    def reasoning_dict_to_str(d):
        tags = get_cot_tags_list()[:-1]  # exclude ACTION
        database_keys = get_cot_database_keys()
        # reasoning_parts = [(tag, d[database_keys[tag]]) for tag in tags if database_keys[tag] in d.keys()] #

        if cot_tags is not None:
            included_tags = cot_tags.split(",")
            inverse_database_keys = get_inverse_cot_database_keys()
            tags = [inverse_database_keys[t] for t in included_tags if t in inverse_database_keys.keys()]
            if len(tags) < len(included_tags):
                print(f"Warning: Some tags in {cot_tags} were not found in the CoT tags list.")
                import pdb; pdb.set_trace()

        reasoning_parts = []
        for tag in tags:
            if database_keys[tag] in d.keys():
                part = d[database_keys[tag]]
                part = str(part).strip()
                if not part.endswith("."):
                    part += "."
                reasoning_parts.append((tag, part))

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

                control_freq = 20 # FIXME: this is hardcoded for libero dataset
                gripper_lookahead_n = 5  # list this many future positions of the gripper
                jump_n = int(control_freq / gripper_lookahead_n)
                trajectory_features = raw_dict[file_name][episode_id]["features"]

                reasoning_dict["gripper"] = ""
                if "gripper_position" in trajectory_features.keys():
                    if trajectory_features["gripper_position"] is not None:
                        if 0 <= int(i) < len(trajectory_features["gripper_position"]):
                            future_positions = []
                            
                            for j in range(gripper_lookahead_n):
                                if int(i) + j * jump_n < len(trajectory_features["gripper_position"]):
                                    future_positions += trajectory_features["gripper_position"][int(i) + j * jump_n]
                                else:
                                    future_positions += future_positions[-2:]

                            reasoning_dict["gripper"] = str(future_positions)

                reasoning_dict["bboxes"] = ""
                if "bboxes" in trajectory_features.keys():
                    if trajectory_features["bboxes"] is not None:
                        if 0 <= int(i) < len(trajectory_features["bboxes"]):
                            if len(trajectory_features["bboxes"][int(i)]) > 0:
                                boxes_list = trajectory_features["bboxes"][int(i)]
                                if isinstance(boxes_list[0], list):
                                    reasoning_dict["bboxes"] = ", ".join(
                                        [f"{name} {box!s}" for prob, name, box in boxes_list]
                                    )
                                elif isinstance(boxes_list[0], dict):
                                    reasoning_dict["bboxes"] = ", ".join(
                                        [f"{box['label']} {box['box']!s}" for box in boxes_list]
                                    )
                                    # reasoning_dict["bboxes"] = ", ".join(
                                    #     [f"{name} {box!s}" for prob, name, box in boxes_list]
                                    # )
                                else:
                                    import pdb; pdb.set_trace()

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


def compute_cot_accuracy(predicted_token_ids, ground_truth_token_ids, llm_tokenizer, log_dir=None, step=None, batch_idx=None, is_main_process=True, mode="train"):
    """
    Compute the accuracy for each CoT tag.
    Args:
        predicted_token_ids: tensor of shape (batch_size, #tokens)
        ground_truth_token_ids: tensor of shape (batch_size, #tokens)
        llm_tokenizer: tokenizer, by default it's the LlamaTokenizerFast
        log_dir: directory to save text logs (optional)
        step: current training step (for logging)
        batch_idx: current batch index (for logging)
        is_main_process: whether this is the main process (to avoid duplicate logs in distributed training)
        mode: either "train" or "val" to distinguish between training and validation
    Returns:
        metrics: dictionary of accuracy for each tag
    """
    # Ensure mode is either "train" or "val"
    assert mode in ["train", "val"], "Mode must be either 'train' or 'val'"
    
    tags = get_cot_tags_list()  # exclude ACTION
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

    # Calculate accuracy for each tag
    total_correct = 0
    total_tokens = 0
    tag_accuracies = {}
    
    # Set up text logging if log_dir is provided and this is the main process
    text_log_file = None
    if log_dir is not None and is_main_process:
        import os
        # Log less frequently to avoid too many files
        should_log = (step is None or batch_idx is None or 
                      (step % 1 == 0 and batch_idx < 3))
        
        if should_log:
            os.makedirs(log_dir, exist_ok=True)
            # Use a separate log file for training and validation
            log_file_path = os.path.join(log_dir, f"cot_tag_analysis_{mode}.txt")
            text_log_file = open(log_file_path, "a")
            text_log_file.write(f"\n\n===== CoT Tag Outputs ({mode.upper()}, Step {step}, Batch {batch_idx}) =====\n\n")
            
            # Log full decoded text - decode each example in the batch separately
            text_log_file.write("===== FULL DECODED TEXT =====\n")
            # Decode one example at a time to avoid the list issue
            for i in range(min(3, len(predicted_token_ids))):
                # Sanitize token IDs to prevent overflow errors
                pred_tokens = predicted_token_ids[i].tolist()
                gt_tokens = ground_truth_token_ids[i].tolist()
                
                # Log token overflow information
                vocab_size = llm_tokenizer.vocab_size
                pred_overflow = [t for t in pred_tokens if t < 0 or t >= vocab_size]
                gt_overflow = [t for t in gt_tokens if t < 0 or t >= vocab_size]
                
                if pred_overflow or gt_overflow:
                    text_log_file.write(f"Example {i} has token overflow:\n")
                    if pred_overflow:
                        text_log_file.write(f"  Predicted tokens outside vocab range [0, {vocab_size-1}]: {pred_overflow}\n")
                    if gt_overflow:
                        text_log_file.write(f"  Ground truth tokens outside vocab range [0, {vocab_size-1}]: {gt_overflow}\n")
                
                # Filter out invalid token IDs
                pred_tokens = [t for t in pred_tokens if 0 <= t < vocab_size]
                gt_tokens = [t for t in gt_tokens if 0 <= t < vocab_size]
                
                try:
                    pred_text = llm_tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    gt_text = llm_tokenizer.decode(gt_tokens, skip_special_tokens=True)
                    
                    text_log_file.write(f"Example {i}:\n")
                    text_log_file.write(f"  PREDICTED:\n{pred_text}\n\n")
                    text_log_file.write(f"  GROUND TRUTH:\n{gt_text}\n\n")
                except Exception as e:
                    text_log_file.write(f"Example {i}: Error decoding tokens: {e}\n")
            text_log_file.write("\n")

    # Compute accuracy for each tag
    for tag in tags:
        correct_tags = [0, 0]
        tag_name = tag[:-1].lower()  # Remove the colon at the end
        
        # Log decoded texts for this tag if we have a log file
        if text_log_file is not None:
            text_log_file.write(f"===== TAG: {tag} =====\n")
            
            # Sample up to 3 examples from the batch for logging
            log_samples = min(3, len(predicted_token_ids))
            for i in range(log_samples):
                reasoning_pred = predicted_token_ids[i]
                mask_pred = final_pred_masks[tag][i]
                reasoning_gt = ground_truth_token_ids[i]
                mask_gt = final_gt_masks[tag][i]
                
                tag_pred = torch.masked_select(reasoning_pred, mask_pred.bool())
                tag_gt = torch.masked_select(reasoning_gt, mask_gt.bool())
                
                # Log token overflow information for this tag
                vocab_size = llm_tokenizer.vocab_size
                tag_pred_overflow = [t.item() for t in tag_pred if t.item() < 0 or t.item() >= vocab_size]
                tag_gt_overflow = [t.item() for t in tag_gt if t.item() < 0 or t.item() >= vocab_size]
                
                if tag_pred_overflow or tag_gt_overflow:
                    text_log_file.write(f"Example {i}, Tag {tag} has token overflow:\n")
                    if tag_pred_overflow:
                        text_log_file.write(f"  Predicted tokens outside vocab range [0, {vocab_size-1}]: {tag_pred_overflow}\n")
                    if tag_gt_overflow:
                        text_log_file.write(f"  Ground truth tokens outside vocab range [0, {vocab_size-1}]: {tag_gt_overflow}\n")
                
                # Sanitize token IDs before decoding
                tag_pred_list = [t.item() for t in tag_pred if 0 <= t.item() < vocab_size]
                tag_gt_list = [t.item() for t in tag_gt if 0 <= t.item() < vocab_size]
                
                try:
                    # Decode tokens
                    pred_text = llm_tokenizer.decode(tag_pred_list, skip_special_tokens=True)
                    gt_text = llm_tokenizer.decode(tag_gt_list, skip_special_tokens=True)
                    
                    text_log_file.write(f"Example {i}:\n")
                    text_log_file.write(f"  Predicted: {pred_text}\n")
                    text_log_file.write(f"  Ground truth: {gt_text}\n\n")
                except Exception as e:
                    text_log_file.write(f"Example {i}: Error decoding tag tokens: {e}\n")
        
        # Calculate accuracy metrics
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

        # Update total counts for overall accuracy
        total_correct += correct_tags[0]
        total_tokens += correct_tags[1]

        if correct_tags[1] > 0:
            tag_accuracy = correct_tags[0] / correct_tags[1]
            tag_accuracies[tag_name] = tag_accuracy.item()
            metrics.update(**{f"reasoning/{tag_name}_tag_accuracy": tag_accuracy})
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    metrics.update({"reasoning/overall_tag_accuracy": overall_accuracy})
    
    # Log accuracies to file
    if text_log_file is not None:
        text_log_file.write("===== ACCURACY METRICS =====\n")
        for tag_name, acc in tag_accuracies.items():
            text_log_file.write(f"{tag_name}: {acc:.4f}\n")
        text_log_file.write(f"Overall: {overall_accuracy:.4f}\n\n")
        
        # Also write a CSV-formatted line for easier parsing/plotting
        # Use separate CSV files for training and validation
        if step is not None:
            csv_file_path = os.path.join(log_dir, f"cot_tag_accuracies_{mode}.csv")
            # Check if file exists, if not create with header
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, "w") as csv_file:
                    header = "step,batch,overall," + ",".join([t[:-1].lower() for t in tags])
                    csv_file.write(header + "\n")
            
            # Append accuracy data
            with open(csv_file_path, "a") as csv_file:
                values = [str(step), str(batch_idx), f"{overall_accuracy:.4f}"]
                for tag in tags:
                    tag_name = tag[:-1].lower()
                    values.append(f"{tag_accuracies.get(tag_name, 0):.4f}")
                csv_file.write(",".join(values) + "\n")
    
    # Close the log file if we opened one
    if text_log_file is not None:
        text_log_file.write("\n\n")
        text_log_file.close()

    return metrics


def split_reasoning(text, tags: List[CotTag] = None):
    if tags is None:
        tags = get_cot_tags_list()

    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v

    return new_parts


def get_metadata(reasoning: Dict[str, str]):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        try:
            gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
            gripper_pos = gripper_pos.split("[")[-1]
            gripper_pos = gripper_pos.split("]")[0]
            gripper_pos = [int(x) for x in gripper_pos.split(",")]
            gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
            metadata["gripper"] = gripper_pos
        except:
            print("Error in gripper pos!")

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            if "[" not in sample:
                continue
            obj = sample.split("[")[0]
            if obj.strip() == "":
                continue
            try:
                coords = [int(n) for n in sample.split("[")[-1].split(",")]
                metadata["bboxes"][obj] = coords
            except Exception as e:
                print(f"Error parsing bbox: {e}")

    return metadata


def visualize_reasoning(image: np.ndarray, instruction: str, reasoning_text: str=''):
    tags = [f" {tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(reasoning_text, tags)
    text = ['Instruction: ' + instruction]
    for tag in tags:
        if tag in reasoning:
            text += [tag + reasoning[tag]]
        else:
            text += [tag + ' ']
    metadata = get_metadata(reasoning)
    bboxes = {}
    for k, v in metadata["bboxes"].items():
        if k[0] == ",":
            k = k[1:]
        bboxes[k.lstrip().rstrip()] = v

    caption = ""
    for t in text:
        wrapper = textwrap.TextWrapper(width=80, replace_whitespace=False)
        word_list = wrapper.wrap(text=t)
        caption_new = ''
        for ii in word_list[:-1]:
            caption_new = caption_new + ii + '\n    '
        caption_new += word_list[-1]

        caption += caption_new.lstrip() + "\n"

    img_arr = np.array(image)
    img_size = img_arr.shape[:2]
    draw_2d_points(img_arr, metadata["gripper"], img_size=img_size)
    draw_bboxes(img_arr, bboxes, img_size=img_size)

    base = Image.fromarray(np.ones((img_size[0], img_size[0] * 2, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(base)
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=7)
    color = (0,0,0) # RGB
    draw.text((5, 5), caption, color, font=font)

    text_arr = np.array(base)
    # resize text_arr to make it can be concatenated with img_arr at the same height, keep the aspect ratio
    target_height = img_arr.shape[0]
    text_arr = cv2.resize(text_arr, (int(text_arr.shape[1] * target_height / text_arr.shape[0]), target_height))
    
    reasoning_img = np.concatenate([img_arr, text_arr], axis=1)

    return reasoning_img


class RuntimeCoTGenerator:
    def __init__(self, device: str):
        # models for generating bboxes
        print(f"Loading Prismatic VLM...")
        hf_token = os.environ["HF_TOKEN"]
        vlm_model_id = "prism-dinosiglip+7b"
        self.local_vlm = load(vlm_model_id, hf_token=hf_token)
        self.local_vlm = self.local_vlm.to(device, dtype=torch.bfloat16)

        print(f"Loading gDINO...")
        gdino_model_id = "IDEA-Research/grounding-dino-base"
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_model_id, size={"shortest_edge": 256, "longest_edge": 256})
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(device)
        self.gdino_model = self.gdino_model.to(device, dtype=torch.bfloat16)

    # PART 1: Prismatic VLM + Grounding DINO for generating object bboxes
    def create_vlm_prompt(self, instruction: str):
        """Create a prompt for the vision-language model to detect objects"""
        user_prompt = "List all the objects you can see in this image, especially including any objects mentioned in the language instruction. Format your response as a simple list of object names separated by periods (e.g., 'cup. table. robot gripper.'). Be specific and comprehensive, but avoid using commas or other punctuation."
        instruction = instruction.strip()
        if len(instruction) > 0 and instruction[-1] == ".":
            instruction = instruction[:-1]
        if len(instruction) > 0 and " " in instruction:
            user_prompt = f"The robot task is: '{instruction}.' " + user_prompt
        return user_prompt

    def post_process_object_list(self, caption):
        """
        Process the VLM output to create a clean list of objects separated by periods.
        This format works better for gDINO object detection.
        """
        # Remove any explanatory text or prefixes
        if ":" in caption:
            caption = caption.split(":", 1)[1]
        
        # Replace commas with periods
        caption = caption.replace(",", ".")
        
        # Replace other list markers and clean up
        caption = caption.replace("-", "").replace("•", "").replace("\n", " ")
        
        # Split by periods, clean each item, and rejoin
        items = [item.strip() for item in caption.split(".") if item.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = [item for item in items if not (item in seen or seen.add(item))]
        
        # Join with periods
        result = ". ".join(unique_items)
        
        # Ensure it ends with a period
        if not result.endswith("."):
            result += "."
            
        return result

    def generate_bboxes(self, image: np.ndarray, instruction: str):
        """Generate bounding boxes for objects in the image using VLM and gDINO"""
        BOX_THRESHOLD = 0.4
        TEXT_THRESHOLD = 0.3

        image_pil = Image.fromarray(image)
        user_prompt = self.create_vlm_prompt(instruction)
        prompt_builder = self.local_vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()
        
        object_list = self.local_vlm.generate(
            image_pil, prompt_text, do_sample=True, temperature=0.4, max_new_tokens=64, min_length=1)
        # Post-process the object list for gDINO
        object_list = self.post_process_object_list(object_list)

        gdino_inputs = self.gdino_processor(
            images=image_pil,
            text=object_list,
            return_tensors="pt",
        ).to(self.gdino_model.device)
        
        with torch.no_grad():
            outputs = self.gdino_model(**gdino_inputs)

        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs, gdino_inputs.input_ids, 
            box_threshold=BOX_THRESHOLD, 
            text_threshold=TEXT_THRESHOLD, 
            target_sizes=[image_pil.size[::-1]])[0]

        logits, phrases, boxes = (
            results["scores"].cpu().numpy(),
            results["labels"],
            results["boxes"].cpu().numpy(),
        )

        bboxes = []
        for lg, p, b in zip(logits, phrases, boxes):
            b = list(b.astype(int))
            lg = float(lg)
            bboxes.append((lg, p, b))

        return bboxes

    # PART 2: Gemini for generating reasoning
    def create_gemini_prompt(self, instruction: str, bboxes: list, gripper_pos=None):
        """Create a prompt for Gemini to generate reasoning based on current observation"""
        # Format bounding boxes as expected in CoT format
        bbox_str = ""
        for _, name, box in bboxes:
            bbox_str += f"{name} {box}, "
        bbox_str = bbox_str.rstrip(", ")
        
        # We're adapting the existing format but for a single step rather than trajectory
        prompt = f"""# Generate reasoning for robot action

## Task Information
The robot is given the following instruction: "{instruction}"

## Scene Information
The following objects have been detected in the current scene with their bounding boxes:
{bbox_str}

## Format Requirements
Your response must strictly follow this exact format:

TASK: {instruction}

PLAN:
  - Step 1: [First step to accomplish the task]
  - Step 2: [Second step to accomplish the task]
  - [Continue with additional steps as needed]

VISIBLE OBJECTS: {bbox_str}

SUBTASK REASONING: [Explain what the immediate next subtask should be and why]

SUBTASK: [State the specific immediate subtask to execute]

MOVE REASONING: [Explain what movement the robot should make next and why]

MOVE: [Specify the exact movement command: "forward", "backward", "left", "right", "up", "down", "stop", "grasp", "release"]

Your reasoning should be clear, concise, and directly actionable for the robot.
"""
        return prompt

    def generate_reasoning(self, image: np.ndarray, instruction: str, gripper_position=None):
        """Generate a complete reasoning for the current observation"""
        # Step 1: Generate bounding boxes for objects in the image
        bboxes = self.generate_bboxes(image, instruction)
        
        # Step 2: Create prompt for Gemini
        prompt = self.create_gemini_prompt(instruction, bboxes, gripper_position)
        
        # Step 3: Call Gemini to generate reasoning
        try:
            import google.generativeai as genai
            
            # Configure the Gemini API with your API key
            if "GOOGLE_API_KEY" in os.environ:
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            else:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            # Set up the model
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            # Use Gemini Pro model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=generation_config,
            )
            
            # Get the response
            response = model.generate_content(prompt)
            reasoning_text = response.text
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Fallback reasoning if API call fails
            reasoning_text = f"""TASK: {instruction}

PLAN:
  - Analyze the scene
  - Identify relevant objects
  - Execute appropriate action

VISIBLE OBJECTS: {', '.join([f"{name} {box}" for _, name, box in bboxes])}

SUBTASK REASONING: Unable to generate detailed reasoning due to API error.

SUBTASK: Analyze the scene first.

MOVE REASONING: Need more information to determine the next move.

MOVE: stop"""
        
        # Step 4: Add gripper position information if available
        if gripper_position is not None:
            reasoning_text += f"\n\nGRIPPER POSITION: {gripper_position}"
        else:
            # Use placeholder values if no gripper position is provided
            default_gripper = [97, 45, 97, 45, 89, 52, 83, 58, 82, 57]
            reasoning_text += f"\n\nGRIPPER POSITION: {default_gripper}"
        
        return reasoning_text

def plot_cot_accuracy_curves(log_dir):
    """
    Generate plots of CoT tag accuracy over training steps.
    
    Args:
        log_dir: Directory containing the cot_tag_accuracies.csv files
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    for mode in ["train", "val"]:
        csv_path = os.path.join(log_dir, f"cot_tag_accuracies_{mode}.csv")
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}")
            continue
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Plot overall accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['overall'], marker='o', linestyle='-', label='Overall')
        plt.title(f'Overall CoT Tag Accuracy ({mode.upper()})')
        plt.xlabel('Training Step')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'overall_accuracy_{mode}.png'))
        plt.close()
        
        # Plot individual tag accuracies
        tag_columns = [col for col in df.columns if col not in ['step', 'batch', 'overall']]
        plt.figure(figsize=(12, 8))
        for tag in tag_columns:
            plt.plot(df['step'], df[tag], marker='.', linestyle='-', label=tag)
        
        plt.title(f'CoT Tag Accuracies ({mode.upper()})')
        plt.xlabel('Training Step')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'tag_accuracies_{mode}.png'))
        plt.close()
    
    # Create comparison plots if both train and val data exist
    train_csv = os.path.join(log_dir, "cot_tag_accuracies_train.csv")
    val_csv = os.path.join(log_dir, "cot_tag_accuracies_val.csv")
    
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        # Plot overall accuracy comparison
        plt.figure(figsize=(10, 6))
        plt.plot(train_df['step'], train_df['overall'], 'b-', marker='o', label='Train')
        
        # For validation, we might have fewer points, so we need to be careful
        val_steps = val_df['step'].values
        plt.plot(val_steps, val_df['overall'], 'r-', marker='s', label='Validation')
        
        plt.title('Overall CoT Tag Accuracy (Train vs. Validation)')
        plt.xlabel('Training Step')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'overall_accuracy_comparison.png'))
        plt.close()
    
    print(f"Plots saved to {plot_dir}")