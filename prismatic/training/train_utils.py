"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids, action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    # First identify action tokens (those greater than ACTION_TOKEN_BEGIN_IDX)
    action_tokens_mask = token_ids > action_token_begin_idx
    
    # Create a tensor marking valid positions (not IGNORE_INDEX)
    valid_positions = token_ids != IGNORE_INDEX
    
    # Calculate cumulative sum of valid action tokens to identify regions
    action_valid_positions = action_tokens_mask & valid_positions
    cumsum = torch.cumsum(action_valid_positions, dim=1)
    
    # Create the mask for current actions (first ACTION_DIM action tokens)
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)
    
    # Final mask is where we have action tokens that are in the current action region
    mask = action_tokens_mask & mask

    return mask


def get_next_actions_mask(token_ids, action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    # First identify action tokens (those greater than ACTION_TOKEN_BEGIN_IDX)
    action_tokens_mask = token_ids > action_token_begin_idx
    
    # Create a tensor marking valid positions (not IGNORE_INDEX)
    valid_positions = token_ids != IGNORE_INDEX
    
    # Calculate cumulative sum of valid action tokens to identify regions
    action_valid_positions = action_tokens_mask & valid_positions
    cumsum = torch.cumsum(action_valid_positions, dim=1)
    
    # Create the mask for future actions (after the first ACTION_DIM action tokens)
    mask = cumsum > ACTION_DIM
    
    # Final mask is where we have action tokens that are in the future action region
    mask = action_tokens_mask & mask
    
    return mask


def get_valid_text_mask(token_ids, action_token_begin_idx=ACTION_TOKEN_BEGIN_IDX):
    # It will actually include the stop token if it's not ignored

    # Identify valid tokens (not IGNORE_INDEX)
    valid_tokens = token_ids != IGNORE_INDEX
    
    # Identify non-action tokens (not greater than ACTION_TOKEN_BEGIN_IDX)
    non_action_tokens = token_ids <= action_token_begin_idx
    
    # Reasoning tokens are valid tokens that are not action tokens
    reasoning_mask = valid_tokens & non_action_tokens
    
    return reasoning_mask



def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy

def compute_action_token_accuracy(predicted_action_token_ids, ground_truth_token_ids, mask):
    if predicted_action_token_ids.shape[0] != ground_truth_token_ids[mask].shape[0]:
        print("Error: Shape not match!")
        return torch.tensor(0)
    correct_preds = (predicted_action_token_ids == ground_truth_token_ids[mask])
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_token_accuracy_abs(predicted_token_ids, ground_truth_token_ids, mask):
    total_token = mask.sum().float()
    if predicted_token_ids.shape[1] > ground_truth_token_ids.shape[1]:
        predicted_token_ids = predicted_token_ids[:, :ground_truth_token_ids.shape[1]]
    elif predicted_token_ids.shape[1] < ground_truth_token_ids.shape[1]:
        ground_truth_token_ids = ground_truth_token_ids[:, :predicted_token_ids.shape[1]]
        mask = mask[:, :predicted_token_ids.shape[1]]
    
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / total_token
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss

def compute_actions_l1_loss_from_action(action_tokenizer, pred_continuous_actions, ground_truth_token_ids, mask):
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    pred_continuous_actions = torch.tensor(pred_continuous_actions[0]).to(true_continuous_actions.dtype).to(true_continuous_actions.device)
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss