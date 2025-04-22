import tensorflow_datasets as tfds

import numpy as np
from PIL import Image

# episode_path_template = "/nvme_data/embodied_agent/cot_cobot_data/wipe_the_board/episode_{}.hdf5"
# episode_path = episode_path_template.format(1)
# with open(os.path.dirname(episode_path) + "/expanded_instruction_gpt-4-turbo.json", "r") as f:
#     INSTRUCTION = json.load(f)["expanded_instruction"][0]

# prompt = get_openvla_prompt(INSTRUCTION)
# print(prompt.replace(". ", ".\n"))

# episode_data = h5py.File(episode_path, "r")
# # import pdb; pdb.set_trace()
# images = episode_data["observations"]["images"]["cam_high"]
# image = cv2.imdecode(np.frombuffer(images[-100], np.uint8), cv2.IMREAD_COLOR)
# # convert image to PIL image
# image = Image.fromarray(image)


def get_data_from_rlds(data_dir: str, dataset_name: str, split: str = "train"):
    """
    Get images from RLDS dataset.
    """
    dataset = tfds.load(dataset_name, data_dir=data_dir, split=split)
    all_data = dict()
    for i, sample in enumerate(dataset.take(10)):
        data = dict()
        # import pdb; pdb.set_trace()
        data['episode_id'] = sample['episode_metadata']['episode_id'].numpy()
        data['file_path'] = sample['episode_metadata']['file_path'].numpy().decode()
        for j, step in enumerate(sample['steps']):
            if j == 0:
                data['language_instruction'] = step['language_instruction'].numpy().decode()
                data['action'] = []
                data['observation'] = {}
            data['action'].append(step['action'].numpy())
            for k, v in step['observation'].items():
                if k not in data['observation']:
                    data['observation'][k] = []
                data['observation'][k].append(
                    Image.fromarray(v.numpy())
                )
        all_data[data['file_path'] + '_' + str(data['episode_id'])] = data

    return all_data

if __name__ == "__main__":
    data = get_data_from_rlds("/home/gaofeng/arm_ws/openvla/datasets", "bridge_orig", split="train", image_key="images")
    import pdb; pdb.set_trace()
