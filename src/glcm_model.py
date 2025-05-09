import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from skimage.feature import graycomatrix, graycoprops
import cv2  # For image processing
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt  # Add this import at the top of the file
from joblib import Parallel, delayed

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
        Extract GLCM features from a list of grayscale images using parallel processing.
        Returns both the features and the indices of successfully processed ROIs.
        """
        def process_roi(image, roi, img_idx, roi_idx):
            x, y, w, h = map(int, roi)

            # Clip ROI coordinates to fit within the image dimensions
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))  # Ensure width is at least 1
            h = max(1, min(h, image.shape[0] - y))  # Ensure height is at least 1

            cropped_image = image[y:y+h, x:x+w]

            # Skip invalid ROIs
            if cropped_image.size == 0:
                return None, None

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
            return roi_features, (img_idx, roi_idx)

        # Flatten the input for parallel processing
        tasks = []
        for img_idx, image in enumerate(images):
            if rois is not None and img_idx < len(rois):
                for roi_idx, roi in enumerate(rois[img_idx]):
                    tasks.append((image, roi, img_idx, roi_idx))

        # Process ROIs in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_roi)(image, roi, img_idx, roi_idx) for image, roi, img_idx, roi_idx in tasks
        )

        # Collect features and valid ROI indices
        features = []
        valid_roi_indices = []
        for feature, valid_index in results:
            if feature is not None:
                features.append(feature)
                valid_roi_indices.append(valid_index)

        return np.array(features), valid_roi_indices




    def preprocess_labels(self, labels):
        """
        Preprocess labels for binary classification (depth-deep vs depth-shallow).
        
        Args:
            labels: List of depth labels ('depth-deep' or 'depth-shallow')
        
        Returns:
            np.ndarray: Binary labels (0 for shallow, 1 for deep)
        """
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            # Convert string labels to binary (0 for shallow, 1 for deep)
            binary_labels = np.array([1 if label == 'depth-deep' else 0 for label in labels])
            self.mlb.fit([['depth-shallow'], ['depth-deep']])
        else:
            binary_labels = np.array([1 if label == 'depth-deep' else 0 for label in labels])
        
        return binary_labels




    def load_data(self, data_loader):
        """
        Load data using the provided data loader.
        Extract GLCM features for each ROI and associate them with labels.
        """
        print("Loading data...")
        self.data = data_loader.load_data()

        # Extract features for training data
        print("Extracting GLCM features for training data...")
        train_images = self.data['train_images']
        train_rois = self.data['train_rois']
        train_features = []
        train_valid_indices = []
        
        for idx in tqdm(range(len(train_images)), desc="Training Data Progress"):
            features, valid_indices = self.extract_glcm_features([train_images[idx]], [train_rois[idx]])
            if len(features) > 0:
                train_features.extend(features)
                train_valid_indices.extend(valid_indices)
        
        # Ensure one-to-one mapping between features and labels
        aligned_train_labels = []
        for img_idx, roi_idx in train_valid_indices:
            if img_idx < len(self.data['train_labels']) and roi_idx < len(self.data['train_labels'][img_idx]):
                aligned_train_labels.append(self.data['train_labels'][img_idx][roi_idx])
        
        self.data['train_features'] = np.array(train_features)
        self.data['train_labels'] = np.array(aligned_train_labels)

        # Similar process for validation data
        print("Extracting GLCM features for validation data...")
        val_images = self.data['val_images']
        val_rois = self.data['val_rois']
        val_features = []
        val_valid_indices = []
        
        for idx in tqdm(range(len(val_images)), desc="Validation Data Progress"):
            features, valid_indices = self.extract_glcm_features([val_images[idx]], [val_rois[idx]])
            if len(features) > 0:
                val_features.extend(features)
                val_valid_indices.extend(valid_indices)
        
        # Align validation labels
        aligned_val_labels = []
        for img_idx, roi_idx in val_valid_indices:
            if img_idx < len(self.data['val_labels']) and roi_idx < len(self.data['val_labels'][img_idx]):
                aligned_val_labels.append(self.data['val_labels'][img_idx][roi_idx])
        
        self.data['val_features'] = np.array(val_features)
        self.data['val_labels'] = np.array(aligned_val_labels)

        # Debug information
        print(f"Training features shape: {self.data['train_features'].shape}")
        print(f"Training labels shape: {self.data['train_labels'].shape}")
        print(f"Validation features shape: {self.data['val_features'].shape}")
        print(f"Validation labels shape: {self.data['val_labels'].shape}")

        print("Data loading and feature extraction completed.")



    def train(self, train_data, val_data):
        """
        Train the model using the provided training and validation data.
        Ensures one-to-one mapping between ROIs/labels and GLCM features.
        """
        print("Extracting GLCM features for training data...")
        train_features = []
        train_labels = []
        
        # Debug: Print initial counts
        print(f"Number of training images: {len(train_data['images'])}")
        print(f"Number of training ROIs: {sum(len(rois) for rois in train_data['rois'])}")
        
        # Process each image and its ROIs
        for idx in tqdm(range(len(train_data["images"])), desc="Training GLCM Extraction"):
            image = train_data["images"][idx]
            rois = train_data["rois"][idx]
            image_labels = train_data["labels"][idx]
            
            # Extract features for all ROIs in this image
            features, valid_indices = self.extract_glcm_features([image], [rois])
            
            # Create feature-label pairs ensuring one-to-one mapping
            for feature, (_, roi_idx) in zip(features, valid_indices):
                if roi_idx < len(image_labels):
                    train_features.append(feature)
                    train_labels.append(image_labels[roi_idx])
                else:
                    print(f"Warning: Skipping ROI {roi_idx} in image {idx} - no corresponding label")

        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        
        # Debug: Print shapes before preprocessing
        print(f"Shape of train_features before preprocessing: {train_features.shape}")
        print(f"Shape of train_labels before preprocessing: {train_labels.shape}")
        
        train_labels = self.preprocess_labels(train_labels)
        
        # Debug: Print final shapes
        print(f"Final shape of train_features: {train_features.shape}")
        print(f"Final shape of train_labels: {train_labels.shape}")
        
        # Similar process for validation data
        print("Extracting GLCM features for validation data...")
        val_features = []
        val_labels = []
        
        for idx in tqdm(range(len(val_data["images"])), desc="Validation GLCM Extraction"):
            image = val_data["images"][idx]
            rois = val_data["rois"][idx]
            image_labels = val_data["labels"][idx]
            
            features, valid_indices = self.extract_glcm_features([image], [rois])
            
            for feature, (_, roi_idx) in zip(features, valid_indices):
                if roi_idx < len(image_labels):
                    val_features.append(feature)
                    val_labels.append(image_labels[roi_idx])

        val_features = np.array(val_features)
        val_labels = self.preprocess_labels(val_labels)

        print(f"Shape of val_features: {val_features.shape}")
        print(f"Shape of val_labels: {val_labels.shape}")

        print("Starting training...")
        self.model.fit(
            train_features,
            train_labels,
            eval_set=[(val_features, val_labels)],
            eval_metric="logloss",
            callbacks=[
                early_stopping(self.config.early_stopping_rounds),
                log_evaluation(period=1)
            ]
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

        # Align ground truth labels with valid ROIs and preprocess them
        valid_roi_indices = []
        original_gt_labels = []
        for idx, roi_list in enumerate(rois):
            for roi_idx in range(len(roi_list)):
                valid_roi_indices.append((idx, roi_idx))
                original_gt_labels.append(labels[idx][roi_idx])

        test_labels_numerical = self.preprocess_labels(original_gt_labels)

        # Flatten predictions for evaluation
        flat_predictions = [pred for preds in predictions for pred in preds]

        # Calculate accuracy
        accuracy = accuracy_score(test_labels_numerical, flat_predictions)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Classification report
        report = classification_report(test_labels_numerical, flat_predictions, target_names=self.mlb.classes_)
        print("Classification Report:\n", report)

        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels_numerical, flat_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.mlb.classes_)
        disp.plot(cmap="Blues", xticks_rotation="vertical")
        plt.title("Confusion Matrix")
        plt.show()

        # Calculate mAP
        average_precision = average_precision_score(test_labels_numerical, flat_predictions)
        print(f"Mean Average Precision (mAP): {average_precision:.2f}")

        # Visualize predictions vs ground truth
        print("Visualizing predictions vs ground truth...")
        roi_counter = 0
        for img_idx in range(min(5, len(images))):  # Show up to 5 images
            image = images[img_idx]
            image_rois = rois[img_idx]
            ground_truth_labels_strings = labels[img_idx] # Keep original string labels
            predicted_labels_numerical = predictions[img_idx]

            # Plot the image with ROIs and labels
            plt.figure(figsize=(10, 10))
            plt.imshow(image, cmap="gray")
            plt.title(f"Image {img_idx + 1}: Predictions vs Ground Truth")
            for roi_idx, (roi, gt_label_string, pred_label_numerical) in enumerate(zip(image_rois, ground_truth_labels_strings, predicted_labels_numerical)):
                x, y, w, h = map(int, roi)
                gt_label_numerical = 1 if gt_label_string == 'depth-deep' else 0 # Convert string to numerical for indexing
                gt_label_name = self.mlb.classes_[gt_label_numerical]  # Get the actual label name
                pred_label_name = self.mlb.classes_[pred_label_numerical] # Get the predicted label name
                color = "green" if gt_label_numerical == pred_label_numerical else "red"
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=2))
                plt.text(
                    x, y - 5,
                    f"GT: {gt_label_name}\nPred: {pred_label_name}",
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
