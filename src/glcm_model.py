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
                    angles = self.config.get("angles", [0, np.pi/4, np.pi/2, 3*np.pi/4])
                    levels = self.config.get("levels", 256)

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
            train_features.append(self.extract_glcm_features([train_data["images"][idx]], [train_data.get("rois", [])[idx]]))
        train_features = np.array(train_features).squeeze()

        print("Extracting GLCM features for validation data...")
        val_features = []
        for idx in tqdm(range(len(val_data["images"])), desc="Validation GLCM Extraction"):
            val_features.append(self.extract_glcm_features([val_data["images"][idx]], [val_data.get("rois", [])[idx]]))
        val_features = np.array(val_features).squeeze()

        # Preprocess labels
        train_labels = self.preprocess_labels(train_data["labels"])
        val_labels = self.preprocess_labels(val_data["labels"])

        epochs = self.config.epochs
        best_accuracy = 0
        patience = max(5, int(len(train_data["images"]) / 100))
        patience_counter = 0

        with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(epochs):
                self.model.fit(
                    train_features,
                    train_labels,
                    eval_set=[(val_features, val_labels)],
                    eval_metric="logloss",
                    early_stopping_rounds=10,
                    verbose=True,
                )

                val_predictions = self.model.predict(val_features)
                accuracy = accuracy_score(val_labels, val_predictions)
                print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {accuracy:.2f}")

                pbar.set_postfix({"Validation Accuracy": f"{accuracy:.2f}"})
                pbar.update(1)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                    joblib.dump(self.model, "best_model.pkl")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def predict(self, input_data, rois=None):
        """
        Make predictions using the trained model.
        The model will infer depth labels for each ROI based on GLCM features.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before making predictions.")

        if rois is None:
            raise ValueError("ROIs must be provided for prediction.")

        # Extract GLCM features for the provided ROIs
        input_features = self.extract_glcm_features(input_data, rois)

        # Predict depth labels for each ROI
        predictions = self.model.predict(input_features)
        return predictions

    def evaluate(self, test_data):
        """
        Evaluate the model's performance on test data and display example predictions with visualizations.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before evaluation.")

        print("Extracting GLCM features for test data...")
        test_features = []
        valid_roi_indices = []  # Track valid ROI indices
        for idx in tqdm(range(len(test_data["images"])), desc="Testing Progress"):
            features, valid_indices = self.extract_glcm_features([test_data["images"][idx]], [test_data["rois"][idx]])
            test_features.extend(features)
            valid_roi_indices.extend(valid_indices)

        test_features = np.array(test_features)

        print(f"[Debug] Extracted test features shape: {test_features.shape}")
        print(f"[Debug] Total ROI predictions expected: {len(test_features)}")

        # Align ground truth labels with valid ROIs
        test_labels = []
        for img_idx, roi_idx in valid_roi_indices:
            test_labels.append(test_data["labels"][img_idx][roi_idx])

        test_labels = self.preprocess_labels(test_labels)

        # Debugging: Check test labels
        print(f"[Debug] Flattened test labels: {test_labels[:10]}")
        print(f"[Debug] Number of test labels: {len(test_labels)}")

        # Predict depth labels for all ROIs
        predictions = self.model.predict(test_features)

        # Debugging: Check predictions
        print(f"[Debug] Predictions shape: {predictions.shape}")
        print(f"[Debug] First 5 predictions: {predictions[:5]}")
        print(f"[Debug] Number of predictions: {len(predictions)}")
        print(f"[Debug] Label classes: {getattr(self.mlb, 'classes_', 'Not Set')}")

        # Ensure the number of predictions matches the number of test labels
        if len(predictions) != len(test_labels):
            raise ValueError(f"Mismatch between predictions ({len(predictions)}) and test labels ({len(test_labels)})")

        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Classification report
        report = classification_report(test_labels, predictions, target_names=self.mlb.classes_)
        print("Classification Report:\n", report)

        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.mlb.classes_)
        disp.plot(cmap="Blues", xticks_rotation="vertical")
        plt.title("Confusion Matrix")
        plt.show()

        return accuracy, report


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
