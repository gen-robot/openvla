1. Run `scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py` to generate a file containing captions for all trajectories.
2. Run `scripts/generate_embodied_data/full_reasonings.py` to generate full reasonings. I've updated the script so that now it calls the right functions when executed.
3. Run `scripts/generate_embodied_data/bounding_boxes/generate_bboxes.py` to compute bounding boxes using the captions.
4. Merge the resulting dictionaries of features.