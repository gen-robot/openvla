"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def pad_action(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Pads actions with the first/last action or zero action according to absolute_action_mask.
    """
    def pad(traj, pad_length, pad_mode="right"):
        action_dim = traj["action"].shape[-1]
        absolute_action_mask = tf.broadcast_to(traj["absolute_action_mask"], [pad_length, action_dim])
        if pad_mode == "right":
            valid_action = tf.broadcast_to(traj["action"][-1], [pad_length, action_dim])
        else:
            valid_action = tf.broadcast_to(traj["action"][0], [pad_length, action_dim])
        zero_action = tf.broadcast_to(tf.zeros_like(valid_action), [pad_length, action_dim])
        padding_action = tf.where(absolute_action_mask, valid_action, zero_action)
        if pad_mode == "right":
            return tf.concat([traj["action"], padding_action], axis=0)
        elif pad_mode == "left":
            return tf.concat([padding_action, traj["action"]], axis=0)
        else:
            raise ValueError(f"Invalid pad_mode: {pad_mode}")
        
    traj['action'] = pad(traj, window_size - 1, pad_mode="left")
    traj['action'] = pad(traj, future_action_window_size, pad_mode="right")
    return traj

# FIXME: unit test its correctness
def chunk_act_obs_new(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    * "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
      observations from the past and the current observation. 
    * "action" is given a new axis (at index 1) of size `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current action, and `future_action_window_size` actions from the future.
    * "pad_mask" is added to "observation" and indicates whether an observation should be considered padding (i.e. if it had come from a timestep before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    # effective_traj_len = traj_len - future_action_window_size
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )

    # action shape: [traj_len + window_size - 1 + future_action_window_size, action_dim]
    traj = pad_action(traj, window_size, future_action_window_size)

    # action_chunk_indices = tf.broadcast_to(
    #     tf.range(-window_size + 1, 1 + future_action_window_size),
    #     [traj_len, window_size + future_action_window_size],
    # ) + tf.broadcast_to(
    #     tf.range(traj_len)[:, None],
    #     [traj_len, window_size + future_action_window_size],
    # )
    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(window_size-1, window_size-1 + traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0) # if the chunk_indices is negative, set it to 0, i.e., padding with the first observation

    # goal_timestep = tf.fill([traj_len], window_size + traj_len - 2)
    # floored_action_chunk_indices = tf.minimum(action_chunk_indices, goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(traj_len))
    if "reasoning" in traj:
        traj["reasoning"] = tf.gather(traj["reasoning"], tf.range(traj_len))

    return traj


def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    * "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
      observations from the past and the current observation. 
    * "action" is given a new axis (at index 1) of size `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current action, and `future_action_window_size` actions from the future.
    * "pad_mask" is added to "observation" and indicates whether an observation should be considered padding (i.e. if it had come from a timestep before the start of the trajectory).
    """
    special_keys = ["observation", "action", "task"]

    # FIXME: we should not ignore the last future_action_window_size steps, padding last obs/actions with proper values according to the absolute_action_mask
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    effective_traj_len = traj_len - future_action_window_size
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    # traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    # traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))
    # if "reasoning" in traj:
    #     traj["reasoning"] = tf.gather(traj["reasoning"], tf.range(effective_traj_len))
    for key in traj.keys():
        if key not in special_keys:
            traj[key] = tf.gather(traj[key], tf.range(effective_traj_len))

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
