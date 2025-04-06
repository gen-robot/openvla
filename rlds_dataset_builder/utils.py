import numpy as np
def filter_small_actions(actions, states=None, 
                            pos_thresh=1e-3, 
                            rot_thresh=1e-3, 
                            check_gripper=True):
    """
    Filters out frames with negligible motion, while retaining those with significant gripper state changes.

    Parameters:
    ----------
    actions : ndarray of shape (N, 7)
        Sequence of robot actions. The first 3 values are delta position (in meters),
        the next 3 are delta Euler angles (in radians), and the last is the gripper state (-1 or 1).
    states : ndarray of shape (N, ...) or None
        Optional array of corresponding states (e.g., images, joint states). Will be filtered in sync with actions.
    pos_thresh : float
        Minimum Euclidean norm threshold for delta position to be considered a valid action.
    rot_thresh : float
        Minimum Euclidean norm threshold for delta rotation to be considered a valid action.
    check_gripper : bool
        If True, ensures that any frame where the gripper state changes (e.g., from -1 to 1) is preserved.

    Returns:
    -------
    filtered_actions : ndarray
        Actions with insignificant motion removed, but gripper changes preserved.
    filtered_states : ndarray (optional)
        Filtered states corresponding to the returned actions. Returned only if `states` is provided.
    """
    actions = np.asarray(actions)
    N = actions.shape[0]
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6]
        
        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)

        is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)

        # Preserve gripper toggle events (e.g., from -1 to 1 or vice versa)
        if check_gripper and i > 0:
            prev_gripper = actions[i - 1][6]
            if gripper != prev_gripper:
                is_valid = True

        valid_mask[i] = is_valid

    if states is not None:
        return actions[valid_mask], states[valid_mask]
    return actions[valid_mask], valid_mask

