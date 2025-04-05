`dataset_name` is the name of the dataset to generate the embodied data for.
`data_dir` is the directory to save the generated data.

```bash
./generate_cot.sh <dataset_name> <data_dir> <mode>
```

`mode` is one of the following:
- `reasoning`: generate full reasonings
- `gripper`: generate gripper positions
- `bboxes`: generate bounding boxes
- `descriptions`: generate descriptions

```bash
# this process needs to be run with GPU
./generate_cot.sh $dataset_name $data_dir descriptions
python merge_descriptions.py --results_path ./outputs/$dataset_name/descriptions

# this process needs to be run with GPU
./generate_cot.sh $dataset_name $data_dir bboxes

# this process needs to be run with network proxy
./generate_cot.sh $dataset_name $data_dir reasoning

# this process needs to be run with network proxy
./generate_cot.sh $dataset_name $data_dir gripper

# merge the generated data
python merge_all_json.py --results_path ./outputs/$dataset_name
```