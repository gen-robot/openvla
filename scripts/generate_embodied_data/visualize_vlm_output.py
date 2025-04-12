import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image
import io
import json
import textwrap
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize VLM output including bounding boxes and gripper trajectory.')
    parser.add_argument('--image_path', type=str, default='image.png', help='Path to the input image')
    parser.add_argument('--json_path', type=str, default='vlm_output.json', help='Path to the VLM output JSON file')
    parser.add_argument('--output_path', type=str, default='annotated_image.png', help='Path to save the annotated image')
    parser.add_argument('--show', action='store_true', help='Display the annotated image instead of saving')
    return parser.parse_args()

def load_image(image_path):
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
        print(f"Image dimensions: {img_width} x {img_height}")
        return img, img_width, img_height
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

def load_vlm_output(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        sys.exit(1)

def visualize_vlm_output(img, img_width, img_height, vlm_data, output_path, show_image=False, normalized=True):
    """
    Visualize VLM output including bounding boxes, gripper position, and key subgoals.
    
    Args:
        img: PIL Image object
        img_width: Width of the image
        img_height: Height of the image
        vlm_data: Dictionary containing VLM output data
        output_path: Path to save the annotated image
        show_image: Whether to display the image instead of saving
        normalized: Whether coordinates are in normalized 0-1000 range (True) or in pixel space (False)
    """
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)
    
    # --- Draw Bounding Boxes ---
    if 'visible_objects' in vlm_data:
        visualize_bboxes(ax, vlm_data['visible_objects'], img_width, img_height, normalized)
    
    # --- Draw Gripper Position and Trajectory ---
    if 'gripper_position' in vlm_data:
        visualize_gripper(ax, vlm_data['gripper_position'], img_width, img_height, normalized)
    
    # --- Draw Key Subgoals ---
    if 'key_subgoals' in vlm_data:
        visualize_subgoals(ax, vlm_data['key_subgoals'], img_width, img_height, normalized)
    
    # --- Display ---
    plt.title("VLM Output Visualization", fontsize=14)
    plt.axis('off')  # Hide axes for cleaner image look
    
    # Adjust layout
    plt.tight_layout(pad=0.5)
    
    # Save or show the annotated image
    if show_image:
        plt.show()
    else:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Annotated image saved as {output_path}")

    plt.close()

def visualize_bboxes(ax, bbox_data, img_width, img_height, normalized=True):
    # Define a list of distinct colors for boxes
    box_colors = ['cyan', 'lime', 'yellow', 'magenta', 'orange', 'purple', 'brown', 'pink', 'olive', 'gray']
    color_index = 0
    
    for item in bbox_data:
        box = item['box']
        label = item['name']
        
        # Get coordinates
        y1, x1, y2, x2 = box
        
        # Apply normalization if needed
        if normalized:
            px_x1 = x1 / 1000 * img_width
            px_y1 = y1 / 1000 * img_height
            px_x2 = x2 / 1000 * img_width
            px_y2 = y2 / 1000 * img_height
        else:
            px_x1 = x1
            px_y1 = y1
            px_x2 = x2
            px_y2 = y2
        
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
            px_x1, px_y1 - 5,    # Position text slightly above the box
            label,
            color='black',       # Text color
            fontsize=9,
            bbox=dict(facecolor=current_color, alpha=0.6, pad=0.2, boxstyle='round,pad=0.3')  # Background box
        )

def visualize_gripper(ax, gripper_data, img_width, img_height, normalized=True):
    # Process current gripper position
    current_pos = gripper_data['current']
    px_current_x = None
    px_current_y = None
    
    # Check if gripper was found
    if current_pos[0] != -1 and current_pos[1] != -1:
        current_y, current_x = current_pos
        
        # Apply normalization if needed
        if normalized:
            px_current_x = current_x / 1000 * img_width
            px_current_y = current_y / 1000 * img_height
        else:
            px_current_x = current_x
            px_current_y = current_y
        
        # Plot current gripper position
        ax.scatter(px_current_x, px_current_y, c='red', marker='X', s=200, label='Current Gripper Position', zorder=5)
        ax.text(px_current_x + 10, px_current_y + 10, 'Gripper', color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, pad=0.2, boxstyle='round,pad=0.3'))
    
    # Process trajectory
    trajectory = gripper_data['trajectory']
    traj_points_x = []
    traj_points_y = []
    
    # Add current position as the first point if it exists
    if px_current_x is not None and px_current_y is not None:
        traj_points_y.append(px_current_y)
        traj_points_x.append(px_current_x)
    
    # Add trajectory points
    for i, point in enumerate(trajectory):
        y, x = point
        
        # Apply normalization if needed
        if normalized:
            px_x = x / 1000 * img_width
            px_y = y / 1000 * img_height
        else:
            px_x = x
            px_y = y
            
        traj_points_x.append(px_x)
        traj_points_y.append(px_y)
        
        # Mark each trajectory point
        ax.scatter(px_x, px_y, c='cyan', marker='o', s=80, zorder=4)
        ax.text(px_x + 5, px_y + 5, f"T{i+1}", color='cyan', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.7, pad=0.2, boxstyle='round,pad=0.3'))
    
    # Draw the trajectory line
    if traj_points_x:
        ax.plot(traj_points_x, traj_points_y, color='cyan', linestyle='--', linewidth=2, label='Predicted Trajectory')

def visualize_subgoals(ax, subgoals_data, img_width, img_height, normalized=True):
    wrap_width = 35  # Adjust based on your text length and figure size
    
    for i, subgoal in enumerate(subgoals_data):
        location = subgoal['location']
        description = subgoal['description']
        
        # Denormalize coordinates
        y, x = location
        
        # Apply normalization if needed
        if normalized:
            px_x = x / 1000 * img_width
            px_y = y / 1000 * img_height
        else:
            px_x = x
            px_y = y
        
        # Plot subgoal point
        ax.scatter(px_x, px_y, color='red', marker='*', s=250, zorder=5)
        
        # Add description text
        wrapped_description = textwrap.fill(description, width=wrap_width)
        display_text = f"Subgoal {i+1}: {wrapped_description}"
        
        ax.text(px_x + 15, px_y + 15, display_text,
                color='white', fontsize=9,
                bbox=dict(facecolor='red', alpha=0.7, pad=0.3, boxstyle='round,pad=0.3'),
                zorder=6)

def main():
    args = parse_args()
    img, img_width, img_height = load_image(args.image_path)
    vlm_data = load_vlm_output(args.json_path)
    visualize_vlm_output(img, img_width, img_height, vlm_data, args.output_path, args.show)

if __name__ == "__main__":
    main() 