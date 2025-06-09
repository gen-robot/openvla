import os
import sys
import cv2
import numpy as np
import tqdm
from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")

from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
)

def try_rendering_backend(backend):
    """Try to initialize environment with a specific rendering backend."""
    print(f"Trying rendering backend: {backend}")
    
    if backend == "osmesa":
        os.environ["MUJOCO_GL"] = "osmesa"
        # Remove EGL-specific variables
        if "MUJOCO_EGL_DEVICE_ID" in os.environ:
            del os.environ["MUJOCO_EGL_DEVICE_ID"]
    elif backend == "egl":
        os.environ["MUJOCO_GL"] = "egl"
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    
    # Print current environment
    print(f"MUJOCO_GL = {os.environ.get('MUJOCO_GL', 'not set')}")
    if "MUJOCO_EGL_DEVICE_ID" in os.environ:
        print(f"MUJOCO_EGL_DEVICE_ID = {os.environ['MUJOCO_EGL_DEVICE_ID']}")
    
    return True

def main():
    """
    Test script to check if the libero environment can run on a headless server.
    This script performs random actions in the environment without loading any model.
    """
    print("Starting LIBERO environment test.")

    # Create a directory to save rendered frames
    output_dir = "libero_test_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Frames will be saved to '{output_dir}/'")

    # Try different rendering backends
    backends_to_try = ["osmesa", "egl"]
    
    env = None
    task_description = None
    
    for backend in backends_to_try:
        try:
            try_rendering_backend(backend)
            
            print("Initializing LIBERO task suite...")
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict["libero_spatial"]()
            task = task_suite.get_task(0)  # Get the first task
            env, task_description = get_libero_env(task, "openvla")
            print(f"Successfully initialized environment for task: {task_description}")
            print(f"Rendering backend '{backend}' works!")
            break
            
        except Exception as e:
            print(f"Backend '{backend}' failed with error: {e}")
            if env:
                try:
                    env.close()
                except:
                    pass
                env = None
            continue
    
    if env is None:
        print("\nAll rendering backends failed!")
        print("Please try one of the following:")
        print("1. Install xvfb and run: xvfb-run -a python experiments/robot/libero/test_libero_env.py")
        print("2. Check if your system has proper EGL/OpenGL drivers installed")
        print("3. Try running the script with MUJOCO_GL=glfw (requires a display)")
        return

    # Reset the environment
    print("Resetting environment...")
    try:
        obs = env.reset()
        print("Environment reset successfully.")
    except Exception as e:
        print(f"Error resetting environment: {e}")
        env.close()
        return

    # Run a short episode with random actions
    num_steps = 100
    print(f"Running for {num_steps} steps with random actions...")
    for t in tqdm.tqdm(range(num_steps)):
        try:
            # Get a random action
            action = env.action_space.sample()

            # Step the environment
            obs, reward, done, info = env.step(action)

            # Get an image from the observation
            img = get_libero_image(obs)

            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{t:04d}.png")
            # libero returns RGB, opencv expects BGR
            cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            if done:
                print("Episode finished (done=True). Resetting environment.")
                env.reset()

        except Exception as e:
            print(f"Error during step {t}: {e}")
            break

    print("Test finished successfully!")
    print(f"Check the '{output_dir}/' directory for rendered frames.")
    env.close()

if __name__ == "__main__":
    main() 