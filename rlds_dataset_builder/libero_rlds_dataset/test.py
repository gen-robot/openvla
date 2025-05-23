def print_structure(d, indent=0):
    """
    Recursively prints the structure of a nested dictionary.
    For each leaf, prints its type and, if available, its shape.
    
    Parameters:
        d (dict): The dictionary to traverse.
        indent (int): The current indentation level (for nested output).
    """
    spacing = '    ' * indent  # 4 spaces per level
    for key, value in d.items():
        print(f"{spacing}{key}:", end=" ")
        if isinstance(value, (dict, h5py.Group)):
            print("{")
            print_structure(value, indent=indent + 1)
            print(f"{spacing}}}")
        else:
            # Try to get the shape attribute; if not available, catch the AttributeError.
            try:
                shape = value.shape
                print(f"<{type(value).__name__}, shape={shape}>")
            except AttributeError:
                print(f"<{type(value).__name__}>")
import h5py

data = h5py.File("/nvme_data/embodied_agent/pretrained/rdt-maniskill/demo_1k/PickCube-v1/motionplanning/20241126_193234.h5", "r")
print_structure(data['traj_0'])