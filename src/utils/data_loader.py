import os
import cv2
import numpy as np
import json

def load_data(data_path):
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

        # Load JSON annotations
        with open(annotations_file, "r") as f:
            data = json.load(f)

        # Map image_id to filename and filter for depth labels
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        id_to_category = {cat["id"]: cat["name"] for cat in data["categories"]}

        # Filter annotations for depth categories only
        depth_labels = ["depth-deep", "depth-shallow"]  # Adjust based on your dataset
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            category_name = id_to_category[cat_id]
            if category_name in depth_labels:
                file_path = os.path.join(split_path, id_to_filename[img_id])
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    labels.append(category_name)

        return np.array(images), np.array(labels)

    # Load train, validation, and test splits
    train_images, train_labels = load_split(os.path.join(data_path, "train"))
    val_images, val_labels = load_split(os.path.join(data_path, "valid"))
    test_images, test_labels = load_split(os.path.join(data_path, "test"))

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }
