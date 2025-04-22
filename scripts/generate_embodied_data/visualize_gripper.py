import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io
import json
import textwrap # Import the textwrap module

# Load the image data (replace with actual image loading if needed)
# In this environment, we assume the image is available via a file path
image_path = 'image.png' # Replace with the actual path to your image

try:
    img = Image.open(image_path)
    img_width, img_height = img.size
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()

print(f"Image dimensions: {img_width} x {img_height}")

# --- Points Data ---
# NOTE: Using the gripper position from the *initial* prompt's output
gripper_tip_data = [{"point": [375, 346], "label": "gripper_tip"}]
path_data = [
  {"point": [375, 346], "label": "trajectory_point_1", "reasoning": "Current gripper position, starting point of the trajectory."},
  {"point": [395, 380], "label": "trajectory_point_2", "reasoning": "Move slightly down and right, initiating the approach towards the wine bottle."},
  {"point": [415, 415], "label": "trajectory_point_3", "reasoning": "Continue moving downwards and rightwards to position the gripper more directly above the wine bottle."},
  {"point": [435, 450], "label": "trajectory_point_4", "reasoning": "Descend further, bringing the gripper closer to the vertical alignment needed for grasping the bottle neck/body."},
  {"point": [455, 485], "label": "trajectory_point_5", "reasoning": "Continue the descent towards the bottle, positioning the gripper at an appropriate height and horizontal location to initiate the grasp action in the subsequent steps."}
]

# Combine all points (optional, but can be useful)
all_points_data = gripper_tip_data + path_data

# --- Denormalize and Prepare for Plotting ---
gripper_x_px, gripper_y_px, gripper_label = [], [], []
path_x_px, path_y_px, path_items = [], [], [] # Store full item for reasoning access

# Denormalize gripper tip
tip_point = gripper_tip_data[0]['point']
tip_label = gripper_tip_data[0]['label']
# Check if gripper was found
if tip_point[0] != -1 and tip_point[1] != -1:
    gripper_x_px.append(tip_point[1] / 1000 * img_width)
    gripper_y_px.append(tip_point[0] / 1000 * img_height)
    gripper_label.append(tip_label)
else:
    print("Gripper tip not found in data.")


# Denormalize path points
for item in path_data:
    point = item['point']
    x_pixel = point[1] / 1000 * img_width
    y_pixel = point[0] / 1000 * img_height
    path_x_px.append(x_pixel)
    path_y_px.append(y_pixel)
    path_items.append(item) # Keep the whole dictionary

# --- Plotting ---
# Increase figure size for better readability of text
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(img)

# Plot gripper tip (if found)
if gripper_x_px:
    ax.scatter(gripper_x_px, gripper_y_px, c='red', marker='X', s=200, label=gripper_label[0], zorder=5) # zorder keeps it on top
    ax.text(gripper_x_px[0] + 10, gripper_y_px[0] + 10, gripper_label[0], color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, pad=0.2, boxstyle='round,pad=0.3'))

# Plot path points and trajectory line
ax.plot(path_x_px, path_y_px, color='cyan', marker='o', linestyle='--', linewidth=2, markersize=8, label='Planned Path')

# Add labels with reasoning to path points
wrap_width = 35 # Adjust this width as needed for your text length and figure size
for i, item in enumerate(path_items):
    x_coord = path_x_px[i]
    y_coord = path_y_px[i]
    reason = item['reasoning']
    point_num = i + 1 # 1-based index for display

    # Wrap the reasoning text for better display
    wrapped_reason = textwrap.fill(reason, width=wrap_width)

    # Create the text string including the point number
    display_text = f"{point_num}: {wrapped_reason}"

    # Add the text annotation to the plot
    ax.text(x_coord + 10, # Offset slightly from the point
            y_coord - 15, # Offset slightly from the point
            display_text,
            color='lime', # Text color
            fontsize=8,   # Adjust font size as needed
            # Add a background box for readability
            bbox=dict(facecolor='black', alpha=0.6, pad=0.3, boxstyle='round,pad=0.3'))

# --- Display ---
plt.title("Robotic Gripper Path Plan with Reasoning", fontsize=14)
plt.axis('off') # Hide axes for cleaner image look

# Adjust legend position if needed (e.g., 'best', 'upper left', 'lower right')
ax.legend(loc='upper left', fontsize=10)

# Add grid lines for visual reference (optional)
# ax.grid(True, linestyle=':', alpha=0.5)

# Adjust layout to prevent labels from being cut off
plt.tight_layout(pad=0.5) # Add some padding

# If you want to save the annotated image instead of showing it:
output_filename = "annotated_image_with_reasoning.png"
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
print(f"Annotated image saved as {output_filename}")

# Uncomment the line below to display the plot interactively instead of saving
# plt.show()