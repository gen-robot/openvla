import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image


def images_to_video(images, output_path, fps=30 ):
    """
    Convert a list of images to a video.
    """
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axes for cleaner video

    def update(frame_index):
        ax.clear()
        ax.imshow(images[frame_index])
        ax.set_title(f"Frame {frame_index}")  # Optional: Add frame number

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=1000/fps)

    # Use PillowWriter for broader compatibility and simpler setup
    writer = animation.PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)

    plt.close(fig) # Close the figure to release resources

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]

def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]

def draw_2d_points(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name

        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )