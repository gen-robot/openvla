import numpy as np


def describe_move(move_vec, raw_move_vec):
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "tilt down", 0: None, 1: "tilt up"},
        {},
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},
        {-1: "close gripper", 0: None, 1: "open gripper"},
    ]

    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # identify rolling and pitching

    if move_vec[3] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[3][move_vec[3]]

    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[5][move_vec[5]]

    if move_vec[6] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[6][move_vec[6]]

    if len(description) == 0:
        # Find the most significant movement even if below threshold
        abs_raw = np.abs(raw_move_vec)
        max_idx = np.argmax(abs_raw[:7])
        direction = 1 if raw_move_vec[max_idx] > 0 else -1
        
        # Handle the case where max_idx is 4 (not in names dictionary)
        if max_idx == 4:
            max_idx = 3  # Treat as tilt, same as we do for move_vec[3]
        
        if max_idx in names and direction in names[max_idx]:
            action_name = names[max_idx][direction]
            if max_idx < 3:
                description = "move " + action_name
            else:
                description = action_name
        else:
            # Fallback to a default movement if we can't determine one
            description = "stop"
    
    return description


def classify_movement(move, threshold=0.005):
    diff = move[-1] - move[0]
    # Store the original diff for determining most significant movement
    raw_diff = diff.copy()
    
    # This code normalizes the XYZ movement vector if its magnitude exceeds a threshold
    # It ensures that large movements are scaled down to a consistent maximum size
    # while preserving the direction of movement
    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))

    diff[3:6] /= 1

    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)

    return describe_move(move_vec, raw_diff), move_vec


move_actions = dict()


def get_move_primitives_episode(episode):
    steps = list(episode["steps"])

    states = np.array([step["observation"]["state"] for step in steps])
    actions = [step["action"][:3].numpy() for step in steps]

    move_trajs = [states[i : i + 4] for i in range(len(states) - 1)]
    primitives = [classify_movement(move) for move in move_trajs]
    primitives.append(primitives[-1])

    for (move, _), action in zip(primitives, actions):
        if move in move_actions.keys():
            move_actions[move].append(action)
        else:
            move_actions[move] = [action]

    return primitives


def get_move_primitives(episode_id, builder):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))

    return get_move_primitives_episode(episode)
