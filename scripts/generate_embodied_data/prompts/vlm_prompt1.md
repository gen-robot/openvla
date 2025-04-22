# Analyze robot observation and predict next actions

## Specification of the experimental setup

You're an expert reinforcement learning researcher controlling a robotic arm to complete a task. The task is specified by the instruction: "LANGUAGE_INSTRUCTION". 

## Your objective

Analyze the current image observation of the robotic arm and provide a comprehensive analysis to determine the next actions. Your analysis should include scene understanding, current state assessment, and prediction of future actions required to complete the task.

## Required output format

Your response must be a structured dictionary in JSON format with the following keys:

```json
{
  "task": "Description of complete task as specified in the language instruction",
  "plan": "Complete high-level plan for the entire task from beginning to end",
  "visible_objects": [
    {"name": "object_name", "box": [tly, tlx, bry, brx]},
    {"name": "object_name2", "box": [tly, tlx, bry, brx]}
  ],
  "subtask_reasoning": "Explanation why the current high-level step should be executed",
  "subtask": "Description of high-level step that should be executed now",
  "relevant_objects": ["object1", "object2"],
  "move_reasoning": "Explanation why the selected movement should be executed",
  "move": "Primitive motion(s) to execute (e.g., 'move down, rotate clockwise')",
  "gripper_position": {
    "current": [y, x],
    "trajectory": [[y1, x1], [y2, x2], [y3, x3], [y4, x4], [y5, x5]]
  },
  "key_subgoals": [
    {"location": [y1, x1], "description": "Description of first key subgoal"},
    {"location": [y2, x2], "description": "Description of second key subgoal"}
  ]
}
```

Ensure all fields are properly filled with appropriate content based on your analysis.

## Analysis guidelines

### Task and subtask identification
The "task" field should describe the COMPLETE task as specified in the language instruction, from beginning to end. The "plan" field must outline the FULL plan with all steps required to complete the entire task.

For the "subtask" field, identify specifically which single step of the overall plan should be executed NOW based on the current scene. The "subtask_reasoning" must provide clear justification for why this particular subtask is the appropriate next step in the current state.

Ensure your subtask identification is grounded in the visual evidence. Don't assume steps have been completed without visual confirmation, and don't skip ahead in the plan without completing prerequisite steps.

CONSISTENCY NOTE: The "subtask" must be one logical step from your "plan", and all reasoning should align with the visual state of the scene. The "relevant_objects" must include all objects needed for the current subtask.

### Object detection 
Accurately detect all visible objects relevant to the scene (robot parts, furniture, kitchen items, etc.). For each detected object, provide its bounding box defined by the top-left (y1, x1) and bottom-right (y2, x2) coordinates.

Coordinates must be normalized within a 0-1000 scale, where (0,0) is top-left and (1000,1000) is bottom-right of the image. Be as specific as possible when labeling objects (e.g., "plastic cup" instead of just "cup").

You MUST include all task-relevant objects in your detections. Always detect manipulatable parts rather than just larger objects - for example, detect a "drawer handle" rather than just a "drawer", as the robot needs to interact with specific parts. Additionally, always include the gripper in your detections, as the robot needs to be aware of its position for every task.

CONSISTENCY NOTE: Ensure all objects that are later referenced in "relevant_objects", trajectory planning, or reasoning sections are properly detected here. If the gripper is currently grasping an object, both bounding boxes should be close or overlapping.

### Spatial reasoning
You MUST use bounding box coordinates to determine spatial relationships between objects. Pay careful attention to the relative positions and distances between objects, especially when reasoning about interactions. Use the bounding box information to verify:

1. Whether the gripper is actually grasping an object (their bounding boxes must overlap or be very close)
2. Which objects are within reach of the gripper in its current position
3. The correct ordering and arrangement of objects in the scene
4. Potential obstacles between the gripper and its target objects

Do NOT make assumptions about object states or relationships without confirming them through spatial analysis of the bounding boxes. For example, never claim the gripper is grasping an object if their bounding boxes are distant from each other.

### Primitive motion options
Choose from these options for the "move" field:
- move forward: Move horizontally away from the robot's base
- move backward: Move horizontally towards the robot's base
- move left: Move horizontally to the arm's left
- move right: Move horizontally to the arm's right
- move up: Move vertically upwards
- move down: Move vertically downwards
- move forward left: Move diagonally forward and left simultaneously
- move backward right up: Move diagonally backward, right, and upward simultaneously
- tilt up: Angle the gripper upwards (changing pitch)
- tilt down: Angle the gripper downwards (changing pitch)
- rotate clockwise: Twist the gripper clockwise around its pointing direction axis
- rotate counterclockwise: Twist the gripper counterclockwise around its pointing direction axis
- open gripper: Open the gripper's fingers or jaws
- close gripper: Close the gripper's fingers or jaws, typically to grasp
- stop: Make no movement, remain stationary

You may select multiple primitive motions if they are not conflicting. Motions are defined relative to the robotic arm's base, not the camera viewpoint or image orientation.

CONSISTENCY NOTE: The selected motion(s) must align with the gripper trajectory and be logical for moving toward the first key subgoal. The "move_reasoning" should explain this connection clearly.

### Relevant objects
For "relevant_objects", list only the specific objects that are directly involved in the current subtask. These must be a subset of objects detected in "visible_objects" and should only include items that the gripper will interact with or navigate around to complete the immediate subtask.

CONSISTENCY NOTE: Every object listed here must appear in "visible_objects" with accurate bounding boxes, and must be referenced in your reasoning and planning.

### Gripper position
For "gripper_position", accurately identify the center of the robotic gripper tip in the current image, then predict five future positions the gripper tip is expected to follow in sequence over the next second to progress toward the current subgoal.

All positions should be provided as [y, x] coordinates using the same 0-1000 normalized scale. The sequence should represent a logical trajectory for the gripper to follow based on the current state and task requirements.

The trajectory should consider potential obstacles in the environment and ensure smooth motion between points. Focus on making progress toward the immediate subgoal rather than planning the entire task trajectory.

If the gripper is not visible in the image, use [-1, -1] as the current position.

CONSISTENCY NOTE: The current gripper position MUST be within the bounding box of the gripper detected in "visible_objects". The trajectory MUST start from the current position and logically progress toward the first key subgoal. If the gripper is grasping an object, ensure the trajectory accounts for carrying this object.

### Key subgoals
For "key_subgoals", identify 2-3 critical locations the gripper must reach to complete the current subtask. These points should represent essential waypoints in the manipulation sequence, such as:
1. An approach position (where the gripper should position itself before interaction)
2. An interaction position (where grasping, pushing, or manipulation occurs)
3. A target position (where an object should be placed or where a manipulation ends)

Provide coordinates as [y, x] values using the normalized 0-1000 scale, with clear descriptions of what each subgoal represents. These subgoals must directly correspond to your plan and the predicted gripper trajectory.

The first key subgoal MUST be the immediate target that the gripper trajectory is moving toward. The trajectory points should form a logical path from the current gripper position toward this first subgoal, considering any obstacles that need to be avoided.

Ensure subgoals are precisely positioned on relevant interaction points (e.g., on a handle rather than a general drawer area, on a graspable part of an object rather than its center). For grasping tasks, position the subgoal where the gripper's center would be during a successful grasp.

CONSISTENCY NOTE: The key subgoals MUST be positioned at specific interaction points on objects listed in "relevant_objects". Their positions should be within or very close to the bounding boxes of the relevant objects they interact with.

### Maintaining consistency between elements
Ensure strong coherence between "visible_objects", "relevant_objects", "gripper_position", and "key_subgoals":

0. The "task" MUST be consistent with the language instruction, and the "plan" MUST be a complete plan for the entire task from beginning to end
1. Objects listed in "relevant_objects" MUST be a subset of those detected in "visible_objects"
2. The gripper trajectory MUST progress logically toward the first key subgoal
3. Key subgoals MUST be positioned at specific interaction points on relevant objects
4. Both the "move" and "trajectory" should navigate around obstacles visible in the scene
5. Ensure spatial reasoning consistency - if moving toward an object, the trajectory and subgoals should align with that object's detected position
6. The "subtask_reasoning" and "move_reasoning" should reference the same relevant objects that appear in your trajectory planning
7. Maintain consistency with object interactions - if the gripper is currently grasping an object, the bounding boxes of the gripper and this object should be close enough or overlapping

Your analysis will be judged on the consistency between these elements. For example, if you identify "cup" as relevant but your gripper trajectory and key subgoals don't align with its position, your response will be considered incorrect. Similarly, if you suggest moving toward an object but don't include it in your relevant objects list, your analysis lacks coherence.