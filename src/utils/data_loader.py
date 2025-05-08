import os
import cv2
import numpy as np
import json

def load_data(data_path):
    """
    Load and preprocess the dataset from the given path.
    Assumes each split (train, val, test) contains images and a JSON annotations file.
    The annotations file name is dynamically detected in the split directory.
    """
    def load_split(split_path):
        images = []
        labels = []
        
        # Find the annotations file dynamically
        annotations_file = None
        for file in os.listdir(split_path):
            if file.endswith(".json"):  # Look for a JSON file
                annotations_file = os.path.join(split_path, file)
                break
        
        if not annotations_file:
            raise FileNotFoundError(f"No JSON annotations file found in {split_path}")
        
        # Load annotations
        with open(annotations_file, "r") as f:
            annotations = json.load(f)  # Assumes JSON format with 'filename' and 'label' keys

        for annotation in annotations:
            file_path = os.path.join(split_path, annotation['filename'])
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(annotation['label'])
        return np.array(images), np.array(labels)

    train_images, train_labels = load_split(os.path.join(data_path, "train"))
    val_images, val_labels = load_split(os.path.join(data_path, "valid"))
    test_images, test_labels = load_split(os.path.join(data_path, "test"))

    train_data = {"images": train_images, "labels": train_labels}
    val_data = {"images": val_images, "labels": val_labels}
    test_data = {"images": test_images, "labels": test_labels}

    return train_data, val_data, test_data