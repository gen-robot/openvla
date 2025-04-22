import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()


    dataset_path = args.dataset_path.rstrip('/')
    name = args.name
    old_name = os.path.basename(dataset_path)

    renamed_path = os.path.join(os.path.dirname(dataset_path), name)

    dataset_json = os.path.join(dataset_path, '1.0.0', 'dataset_info.json')
    with open(dataset_json, 'r') as f:
        dataset = json.load(f)

    # Rename the dataset
    dataset['name'] = name
    dataset['version'] = '1.0.0'

    # rename the original dataset json
    os.rename(dataset_json, os.path.join(dataset_path, '1.0.0', f'dataset_info.json.backup'))

    # save the new dataset json in a readable format
    with open(os.path.join(dataset_path, '1.0.0', f'dataset_info.json'), 'w') as f:
        json.dump(dataset, f, indent=4)

    # rename all files starting with old_name to new_name
    for file in os.listdir(os.path.join(dataset_path, '1.0.0')):
        if file.startswith(old_name):
            os.rename(os.path.join(dataset_path, '1.0.0', file), 
                      os.path.join(dataset_path, '1.0.0', file.replace(old_name, name)))

    # mv the dataset to the new name
    os.rename(dataset_path, renamed_path)

if __name__ == '__main__':
    main()
