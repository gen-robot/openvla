import numpy as np
from scipy.spatial.transform import Rotation as R


def describe_move(move_vec):
    """
    move_vec: (6,), is expected to include position, euler angles, and gripper state
    """

    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "tilt down", 0: None, 1: "tilt up"},
        {},
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},
        {-1: "close gripper", 0: None, 1: "open gripper"}, # FIXME: this should be fixed to fit the dataset format
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
        description = "stop"

    return description


def classify_movement(move, threshold=0.03):
    diff = move[-1] - move[0]

    diff[:3] *= 1000
    diff[3:6] *= 10

    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))


    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)

    return describe_move(move_vec), move_vec


move_actions = dict()


def get_move_primitives_episode(episode):
    steps = list(episode["steps"])

    ee_states = np.array([step["ee_pose"] for step in steps])

    left_ee_states = ee_states[:, :7]

    left_ee_pos, left_ee_quat = left_ee_states[:, :3], left_ee_states[:, 3:]
    left_ee_euler = R.from_quat(left_ee_quat, scalar_first=False).as_euler('xyz', degrees=False)
    left_gripper_states = np.array([step["qpos"][6] for step in steps])
    left_ee_states = np.concatenate([left_ee_pos, left_ee_euler, left_gripper_states[:, None]], axis=-1)

    right_ee_states = ee_states[:, 7:]
    right_ee_pos, right_ee_quat = right_ee_states[:, :3], right_ee_states[:, 3:]
    right_ee_euler = R.from_quat(right_ee_quat, scalar_first=False).as_euler('xyz', degrees=False)
    right_gripper_states = np.array([step["qpos"][13] for step in steps])
    right_ee_states = np.concatenate([right_ee_pos, right_ee_euler, right_gripper_states[:, None]], axis=-1)

    diff_left_ee_pos = left_ee_pos[1:] - left_ee_pos[:-1]
    diff_right_ee_pos = right_ee_pos[1:] - right_ee_pos[:-1]

    diff_left_ee_euler = left_ee_euler[1:] - left_ee_euler[:-1]
    diff_right_ee_euler = right_ee_euler[1:] - right_ee_euler[:-1]

    print(
        "| Array               | Min    | Max    | Mean   | Std    |\n"
        "|---------------------|--------|--------|--------|--------|\n"
        f"| diff_left_ee_pos    | {diff_left_ee_pos.min():.4f} | {diff_left_ee_pos.max():.4f} | {diff_left_ee_pos.mean():.4f} | {diff_left_ee_pos.std():.4f} |\n"
        f"| diff_right_ee_pos   | {diff_right_ee_pos.min():.4f} | {diff_right_ee_pos.max():.4f} | {diff_right_ee_pos.mean():.4f} | {diff_right_ee_pos.std():.4f} |\n"
        f"| diff_left_ee_euler  | {diff_left_ee_euler.min():.4f} | {diff_left_ee_euler.max():.4f} | {diff_left_ee_euler.mean():.4f} | {diff_left_ee_euler.std():.4f} |\n"
        f"| diff_right_ee_euler | {diff_right_ee_euler.min():.4f} | {diff_right_ee_euler.max():.4f} | {diff_right_ee_euler.mean():.4f} | {diff_right_ee_euler.std():.4f} |"
    )

    import pdb; pdb.set_trace()

    states = (left_ee_states, right_ee_states)

    move_trajs = [(states[0][i:i+4], states[1][i:i+4]) for i in range(len(ee_states) - 1)]
    primitives = [(classify_movement(left_move), classify_movement(right_move)) for left_move, right_move in move_trajs]
    primitives.append(primitives[-1])

    # for (move, _), action in zip(primitives, actions):
    #     if move in move_actions.keys():
    #         move_actions[move].append(action)
    #     else:
    #         move_actions[move] = [action]

    return primitives


def get_move_primitives(episode_id, builder):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))

    return get_move_primitives_episode(episode)
