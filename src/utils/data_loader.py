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

        # SAFE load JSON
        with open(annotations_file, "r") as f:
            raw = f.read()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON Decode Error in {annotations_file}: {e}")

            if isinstance(data, str):
                # Try double parsing if it's a JSON string inside a string
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    raise ValueError(f"Double-parsing failed. File is likely corrupt: {annotations_file}")

        # Map image_id to filename
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        id_to_category = {cat["id"]: cat["name"] for cat in data["categories"]}

        # Get a label per image
        image_labels = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if img_id not in image_labels:
                image_labels[img_id] = id_to_category[cat_id]

        for img_id, filename in id_to_filename.items():
            file_path = os.path.join(split_path, filename)
            if not os.path.exists(file_path):
                continue  # Skip missing images
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None and img_id in image_labels:
                images.append(image)
                labels.append(image_labels[img_id])

        return np.array(images), np.array(labels)

    # Load each data split
    train_images, train_labels = load_split(os.path.join(data_path, "train"))
    val_images, val_labels = load_split(os.path.join(data_path, "valid"))
    test_images, test_labels = load_split(os.path.join(data_path, "test"))

    return (
        {"images": train_images, "labels": train_labels},
        {"images": val_images, "labels": val_labels},
        {"images": test_images, "labels": test_labels},
    )
