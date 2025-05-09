import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier  # Import LightGBM
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from skimage.feature import graycomatrix, graycoprops
import cv2  # For image processing
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt  # Add this import at the top of the file

class GLCMModel:
    def __init__(self, config):
        """
        Initialize the GLCMModel with a configuration object.
        """
        self.config = config
        self.model = LGBMClassifier(
            **config.model_params,
        )
        self.data = None
        self.mlb = None  # MultiLabelBinarizer instance




    def extract_glcm_features(self, images, rois=None):
        """
        Extract GLCM features from a list of grayscale images, optionally using multiple ROIs.
        Returns both the features and the indices of successfully processed ROIs.
        """
        features = []
        valid_roi_indices = []  # Track indices of successfully processed ROIs
        for idx, image in enumerate(images):
            if len(image.shape) == 3:  # Convert to grayscale if RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if rois is not None and idx < len(rois):
                for roi_idx, roi in enumerate(rois[idx]):
                    x, y, w, h = map(int, roi)

                    # Clip ROI coordinates to fit within the image dimensions
                    x = max(0, min(x, image.shape[1] - 1))
                    y = max(0, min(y, image.shape[0] - 1))
                    w = max(1, min(w, image.shape[1] - x))  # Ensure width is at least 1
                    h = max(1, min(h, image.shape[0] - y))  # Ensure height is at least 1

                    cropped_image = image[y:y+h, x:x+w]

                    # Skip invalid ROIs
                    if cropped_image.size == 0:
                        print(f"[Warning] Skipping empty ROI at index {idx}, ROI: {roi}")
                        continue

                    # Dynamically calculate GLCM parameters based on ROI size
                    min_dim = min(w, h)
                    distances = [1, max(1, min_dim // 4)]
                    angles = self.config.angles  # Access angles directly
                    levels = self.config.levels  # Access levels directly

                    # Extract GLCM features
                    glcm = graycomatrix(
                        cropped_image,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True,
                    )
                    contrast = graycoprops(glcm, 'contrast').flatten()
                    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
                    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
                    energy = graycoprops(glcm, 'energy').flatten()
                    correlation = graycoprops(glcm, 'correlation').flatten()
                    asm = graycoprops(glcm, 'ASM').flatten()

                    roi_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm])
                    features.append(roi_features)
                    valid_roi_indices.append((idx, roi_idx))  # Track valid ROI indices
            else:
                print(f"[Warning] No ROIs provided for image index {idx}")

        return np.array(features), valid_roi_indices




    def preprocess_labels(self, labels, flatten=True):
        """
        Preprocess labels using MultiLabelBinarizer for multi-label data.
        """
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            labels = self.mlb.fit_transform(labels)
        else:
            labels = self.mlb.transform(labels)

        if flatten and len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)

        return labels




    def load_data(self, data_loader):
        """
        Load data using the provided data loader.
        Extract GLCM features for each ROI and associate them with labels.
        Includes tqdm progress bars for loading and feature extraction.
        """
        print("Loading data...")
        self.data = data_loader.load_data()

        # Extract features for training data
        print("Extracting GLCM features for training data...")
        train_images = self.data['train_images']
        train_rois = self.data['train_rois']  # List of bounding boxes for each image
        train_features = []
        for idx in tqdm(range(len(train_images)), desc="Training Data Progress"):
            train_features.append(self.extract_glcm_features([train_images[idx]], [train_rois[idx]]))
        self.data['train_features'] = np.array(train_features).squeeze()
        self.data['train_labels'] = self.data['train_labels']

        # Extract features for validation data
        print("Extracting GLCM features for validation data...")
        val_images = self.data['val_images']
        val_rois = self.data['val_rois']
        val_features = []
        for idx in tqdm(range(len(val_images)), desc="Validation Data Progress"):
            val_features.append(self.extract_glcm_features([val_images[idx]], [val_rois[idx]]))
        self.data['val_features'] = np.array(val_features).squeeze()
        self.data['val_labels'] = self.data['val_labels']

        # Extract features for test data
        print("Extracting GLCM features for test data...")
        test_images = self.data['test_images']
        test_rois = self.data['test_rois']
        test_features = []
        for idx in tqdm(range(len(test_images)), desc="Test Data Progress"):
            test_features.append(self.extract_glcm_features([test_images[idx]], [test_rois[idx]]))
        self.data['test_features'] = np.array(test_features).squeeze()
        self.data['test_labels'] = self.data['test_labels']

        print("Data loading and feature extraction completed.")




    def train(self, train_data, val_data):
        """
        Train the model using the provided training and validation data.
        Implements early stopping based on validation accuracy.
        """
        print("Extracting GLCM features for training data...")
        train_features = []
        for idx in tqdm(range(len(train_data["images"])), desc="Training GLCM Extraction"):
            features, _ = self.extract_glcm_features([train_data["images"][idx]], [train_data.get("rois", [])[idx]])
            train_features.extend(features)
        train_features = np.array(train_features)

        print("Extracting GLCM features for validation data...")
        val_features = []
        for idx in tqdm(range(len(val_data["images"])), desc="Validation GLCM Extraction"):
            features, _ = self.extract_glcm_features([val_data["images"][idx]], [val_data.get("rois", [])[idx]])
            val_features.extend(features)
        val_features = np.array(val_features)

        # Preprocess labels
        train_labels = self.preprocess_labels(train_data["labels"])
        val_labels = self.preprocess_labels(val_data["labels"])

        print("Starting training...")
        self.model.fit(
            train_features,
            train_labels,
            eval_set=[(val_features, val_labels)],  # Validation data for early stopping
            eval_metric="logloss",                 # Evaluation metric
            callbacks=[early_stopping(self.config.early_stopping_rounds)],  # Early stopping callback
            verbose=True                           # Display training progress
        )
        print("Training completed.")




    def predict(self, images, rois):
        """
        Make predictions using the trained model.
        The model will infer depth labels for each ROI based on GLCM features.

        Args:
            images (list or np.array): List or array of grayscale images to predict on.
            rois (list): List of ROIs for each image.

        Returns:
            dict: A dictionary containing predictions, ROIs, and images.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before making predictions.")

        print("Extracting GLCM features for the specified data...")
        input_features = []
        valid_roi_indices = []  # Track valid ROI indices
        for idx in tqdm(range(len(images)), desc="Prediction Progress"):
            features, valid_indices = self.extract_glcm_features([images[idx]], [rois[idx]])
            input_features.extend(features)
            valid_roi_indices.extend(valid_indices)

        input_features = np.array(input_features)

        print(f"[Debug] Extracted input features shape: {input_features.shape}")
        print(f"[Debug] Total ROI predictions expected: {len(input_features)}")

        # Predict depth labels for all ROIs
        predictions = self.model.predict(input_features)

        # Map predictions back to their corresponding images and ROIs
        mapped_predictions = [[] for _ in range(len(images))]
        for (img_idx, roi_idx), prediction in zip(valid_roi_indices, predictions):
            mapped_predictions[img_idx].append(prediction)

        # Return predictions mapped to images and ROIs
        return {
            "predictions": mapped_predictions,  # List of lists of predictions for each image
            "rois": rois,                       # Original ROIs from input data
            "images": images                    # Original images from input data
        }    




    def evaluate(self, images, rois, labels):
        """
        Evaluate the model's performance on the specified images and ROIs.
        Displays example predictions with visual comparisons.
        Returns accuracy, classification report, and predictions.

        Args:
            images (list or np.array): List or array of grayscale images to evaluate.
            rois (list): List of ROIs for each image.
            labels (list): Ground truth labels for the ROIs.
        """
        predictions_data = self.predict(images, rois)
        predictions = predictions_data["predictions"]

        # Align ground truth labels with valid ROIs
        valid_roi_indices = []
        for idx, roi_list in enumerate(rois):
            valid_roi_indices.extend([(idx, roi_idx) for roi_idx in range(len(roi_list))])

        test_labels = []
        for img_idx, roi_idx in valid_roi_indices:
            test_labels.append(labels[img_idx][roi_idx])

        test_labels = self.preprocess_labels(test_labels)

        # Flatten predictions for evaluation
        flat_predictions = [pred for preds in predictions for pred in preds]

        # Calculate accuracy
        accuracy = accuracy_score(test_labels, flat_predictions)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Classification report
        report = classification_report(test_labels, flat_predictions, target_names=self.mlb.classes_)
        print("Classification Report:\n", report)

        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, flat_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.mlb.classes_)
        disp.plot(cmap="Blues", xticks_rotation="vertical")
        plt.title("Confusion Matrix")
        plt.show()

        # Visualize predictions vs ground truth
        print("Visualizing predictions vs ground truth...")
        for img_idx in range(min(5, len(images))):  # Show up to 5 images
            image = images[img_idx]
            image_rois = rois[img_idx]
            ground_truth_labels = labels[img_idx]
            predicted_labels = predictions[img_idx]

            # Plot the image with ROIs and labels
            plt.figure(figsize=(10, 10))
            plt.imshow(image, cmap="gray")
            plt.title(f"Image {img_idx + 1}: Predictions vs Ground Truth")
            for roi, gt_label, pred_label in zip(image_rois, ground_truth_labels, predicted_labels):
                x, y, w, h = roi
                color = "green" if gt_label == pred_label else "red"
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=2))
                plt.text(
                    x, y - 5,
                    f"GT: {gt_label}\nPred: {pred_label}",
                    color=color,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
                )
            plt.axis("off")
            plt.show()

        # Return accuracy, report, and predictions
        return accuracy, report, predictions


    def save_model(self, filepath):
        """
        Save the trained model and MultiLabelBinarizer to a file.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before saving.")
        joblib.dump({"model": self.model, "mlb": self.mlb}, filepath)
        print(f"Model and MultiLabelBinarizer saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model and MultiLabelBinarizer from a file.
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.mlb = data["mlb"]
        print(f"Model and MultiLabelBinarizer loaded from {filepath}")
