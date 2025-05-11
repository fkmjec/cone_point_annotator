import os

def get_ds_subfolder(path: str):
    # Normalize and split the path
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3:
        raise ValueError("Path does not have enough levels to get the third folder up")
    return parts[-3]

def match_images_labels(image_paths: list, label_paths: list) -> list:
    # Create dictionaries mapping filename (no suffix) -> full path
    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    label_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in label_paths}

    # Find matching filenames
    common_keys = image_dict.keys() & label_dict.keys()

    # Get matched pairs
    matched_pairs = [(image_dict[k], label_dict[k]) for k in common_keys]
    return matched_pairs
