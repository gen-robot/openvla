import matplotlib.pyplot as plt
import matplotlib.patches as patches # Import patches for drawing rectangles
import matplotlib.image as mpimg
from PIL import Image
import io
import json
import textwrap
import random # To generate colors for boxes

# Load the image data
image_path = 'image.png' # Replace with the actual path to your image

try:
    img = Image.open(image_path)
    img_width, img_height = img.size
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()

print(f"Image dimensions: {img_width} x {img_height}")

# --- Bounding Box Data (from the first interaction) ---
# NOTE: Make sure this JSON matches the output you received previously.
# Using the example output provided in the prompt.
box_data_json = """
[
  {"box": [0, 94, 405, 652], "label": "robot"},
  {"box": [370, 513, 528, 565], "label": "wine bottle"},
  {"box": [403, 156, 545, 385], "label": "burner"},
  {"box": [496, 442, 616, 577], "label": "bowl"},
  {"box": [108, 652, 828, 1000], "label": "rack"},
  {"box": [712, 456, 782, 612], "label": "plate"},
  {"box": [641, 452, 784, 637], "label": "plate"},
  {"box": [616, 666, 723, 726], "label": "handle"},
  {"box": [64, 644, 496, 954], "label": "cutting board"},
  {"box": [576, 311, 674, 374], "label": "sponge"}
]
"""
bboxes_data = json.loads(box_data_json)

# --- Points Data (from the second interaction - subgoals) ---
# NOTE: Using the subgoal output from the first prompt
subgoal_data = [
  {"point": [378, 531], "label": "subgoal_1", "reasoning": "Approach point above the wine bottle, enabling the robot to securely grasp it."},
  {"point": [295, 820], "label": "subgoal_2", "reasoning": "Approach point near the rack to prepare for placing the grasped wine bottle on it. It provides a clear target and path for placement."}
]


# --- Denormalize and Prepare for Plotting ---

# Denormalize subgoal points
subgoal_x_px, subgoal_y_px, subgoal_items = [], [], []
for item in subgoal_data:
    point = item['point']
    x_pixel = point[1] / 1000 * img_width
    y_pixel = point[0] / 1000 * img_height
    subgoal_x_px.append(x_pixel)
    subgoal_y_px.append(y_pixel)
    subgoal_items.append(item) # Keep the whole dictionary

# --- Plotting ---
# Increase figure size for better readability
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(img)

# --- Draw Bounding Boxes ---
# Define a list of distinct colors for boxes
box_colors = ['cyan', 'lime', 'yellow', 'magenta', 'orange', 'purple', 'brown', 'pink', 'olive', 'gray']
color_index = 0

for item in bboxes_data:
    box = item['box']
    label = item['label']

    # Denormalize coordinates
    y1, x1, y2, x2 = box
    px_x1 = x1 / 1000 * img_width
    px_y1 = y1 / 1000 * img_height
    px_x2 = x2 / 1000 * img_width
    px_y2 = y2 / 1000 * img_height

    # Calculate width and height for the rectangle patch
    rect_width = px_x2 - px_x1
    rect_height = px_y2 - px_y1

    # Get color for the box
    current_color = box_colors[color_index % len(box_colors)]
    color_index += 1

    # Create rectangle patch
    rect = patches.Rectangle(
        (px_x1, px_y1),      # Top-left corner (x, y)
        rect_width,          # Width
        rect_height,         # Height
        linewidth=1.5,       # Line thickness
        edgecolor=current_color,
        facecolor='none'     # No fill
    )

    # Add the rectangle patch to the axes
    ax.add_patch(rect)

    # Add the object label text near the top-left corner of the box
    ax.text(
        px_x1, px_y1 - 5,  # Position text slightly above the box
        label,
        color='black',       # Text color
        fontsize=9,
        bbox=dict(facecolor=current_color, alpha=0.6, pad=0.2, boxstyle='round,pad=0.3') # Background box
    )


# --- Plot Subgoal Points ---
# Plot subgoal points as stars
ax.scatter(subgoal_x_px, subgoal_y_px, color='red', marker='*', s=250, label='Subgoals', zorder=5) # zorder keeps it on top

# Add labels with reasoning to subgoal points
wrap_width = 35 # Adjust this width as needed
for i, item in enumerate(subgoal_items):
    x_coord = subgoal_x_px[i]
    y_coord = subgoal_y_px[i]
    reason = item['reasoning']
    label_text = item['label'] # Use the label like "subgoal_1"

    # Wrap the reasoning text
    wrapped_reason = textwrap.fill(reason, width=wrap_width)

    # Create the text string including the label and reasoning
    display_text = f"{label_text}: {wrapped_reason}"

    # Add the text annotation to the plot
    ax.text(x_coord + 15, # Offset slightly from the point
            y_coord + 15, # Offset slightly from the point
            display_text,
            color='white', # Text color
            fontsize=9,   # Adjust font size
            # Add a background box for readability
            bbox=dict(facecolor='red', alpha=0.7, pad=0.3, boxstyle='round,pad=0.3'),
            zorder=6) # Ensure text is on top


# --- Display ---
plt.title("Object Detection and Task Subgoals", fontsize=14)
plt.axis('off') # Hide axes for cleaner image look

# Adjust legend position if needed
# ax.legend(loc='best', fontsize=10) # Legend might be cluttered, maybe remove if not needed

# Adjust layout to prevent labels from being cut off
plt.tight_layout(pad=0.5)

# Save the annotated image
output_filename = "annotated_image_with_boxes_and_subgoals.png"
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
print(f"Annotated image saved as {output_filename}")

# Uncomment the line below to display the plot interactively instead of saving
# plt.show()