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
    
        # Find the annotations file
        annotations_file = None
        for file in os.listdir(split_path):
            if file.endswith(".json"):
                annotations_file = os.path.join(split_path, file)
                break
    
        if not annotations_file:
            raise FileNotFoundError(f"No JSON annotations file found in {split_path}")
    
        # Load COCO-style annotations
        with open(annotations_file, "r") as f:
            data = json.load(f)
    
        # Map image_id to filename
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        # Map category_id to category_name
        id_to_category = {cat['id']: cat['name'] for cat in data['categories']}
    
        # Collect one label per image (you can modify this for multilabel if needed)
        image_labels = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if img_id not in image_labels:
                image_labels[img_id] = id_to_category[cat_id]
    
        for img_id, filename in id_to_filename.items():
            file_path = os.path.join(split_path, filename)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None and img_id in image_labels:
                images.append(image)
                labels.append(image_labels[img_id])
        
        return np.array(images), np.array(labels)
    
        train_images, train_labels = load_split(os.path.join(data_path, "train"))
        val_images, val_labels = load_split(os.path.join(data_path, "valid"))
        test_images, test_labels = load_split(os.path.join(data_path, "test"))

    train_data = {"images": train_images, "labels": train_labels}
    val_data = {"images": val_images, "labels": val_labels}
    test_data = {"images": test_images, "labels": test_labels}

    return train_data, val_data, test_data
