import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import graycomatrix, graycoprops
import cv2  # For image processing
import joblib
from tqdm import tqdm

class GLCMModel:
    def __init__(self, config):
        """
        Initialize the GLCMModel with a configuration object.
        """
        self.config = config
        # Access model_params directly from the Config object
        self.model = GradientBoostingClassifier(**config.model_params)
        self.data = None

    def configure_model(self):
        """
        Configure the model based on the settings in the config.
        """
        self.model = GradientBoostingClassifier(**self.config.get("model_params", {}))

    def extract_glcm_features(self, images, rois=None):
        """
        Extract GLCM features from a list of grayscale images, optionally using multiple ROIs.
        """
        features = []
        for idx, image in enumerate(images):
            # Convert image to grayscale if not already
            if len(image.shape) == 3:  # Check if the image is RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Process multiple ROIs if provided
            if rois is not None and idx < len(rois):
                image_features = []
                for roi in rois[idx]:  # Iterate over multiple ROIs for the current image
                    x, y, w, h = roi  # ROI format: (x, y, width, height)
                    cropped_image = image[y:y+h, x:x+w]

                    # Compute GLCM and extract features for the cropped region
                    glcm = graycomatrix(
                        cropped_image,
                        distances=self.config.distances,  # Access directly
                        angles=self.config.angles,        # Access directly
                        levels=self.config.levels,        # Access directly
                        symmetric=True,
                        normed=True,
                    )
                    contrast = graycoprops(glcm, 'contrast').flatten()
                    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
                    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
                    energy = graycoprops(glcm, 'energy').flatten()
                    correlation = graycoprops(glcm, 'correlation').flatten()
                    asm = graycoprops(glcm, 'ASM').flatten()

                    # Combine features for this ROI
                    roi_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm])
                    image_features.append(roi_features)

                # Aggregate features from all ROIs for the current image (e.g., mean or concatenate)
                features.append(np.mean(image_features, axis=0))  # Example: mean of all ROI features
            else:
                # If no ROIs are provided, process the entire image
                glcm = graycomatrix(
                    image,
                    distances=self.config.distances,  # Access directly
                    angles=self.config.angles,        # Access directly
                    levels=self.config.levels,        # Access directly
                    symmetric=True,
                    normed=True,
                )
                contrast = graycoprops(glcm, 'contrast').flatten()
                dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
                homogeneity = graycoprops(glcm, 'homogeneity').flatten()
                energy = graycoprops(glcm, 'energy').flatten()
                correlation = graycoprops(glcm, 'correlation').flatten()
                asm = graycoprops(glcm, 'ASM').flatten()
                features.append(np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm]))

        return np.array(features)

    def load_data(self, data_loader):
        """
        Load data using the provided data loader.
        Extract GLCM features for each ROI and associate them with labels.
        """
        self.data = data_loader.load_data()
        
        # Extract features for training data
        train_images = self.data['train_images']
        train_rois = self.data['train_rois']  # List of bounding boxes for each image
        train_labels = self.data['train_labels']  # Depth labels for each ROI
        self.data['train_features'] = self.extract_glcm_features(train_images, train_rois)
        self.data['train_labels'] = train_labels

        # Extract features for validation data
        val_images = self.data['val_images']
        val_rois = self.data['val_rois']
        val_labels = self.data['val_labels']
        self.data['val_features'] = self.extract_glcm_features(val_images, val_rois)
        self.data['val_labels'] = val_labels

        # Extract features for test data
        test_images = self.data['test_images']
        test_rois = self.data['test_rois']
        test_labels = self.data['test_labels']
        self.data['test_features'] = self.extract_glcm_features(test_images, test_rois)
        self.data['test_labels'] = test_labels

    def train(self, train_data, val_data):
        """
        Train the model using the provided training and validation data.
        Implements early stopping based on validation accuracy.
        """
        # GLCM feature extraction with progress bar for training data
        print("Extracting GLCM features for training data...")
        train_features = []
        for idx in tqdm(range(len(train_data["images"])), desc="Training GLCM Extraction"):
            train_features.append(self.extract_glcm_features([train_data["images"][idx]], [train_data.get("rois", [])[idx]]))
        train_features = np.array(train_features).squeeze()

        # GLCM feature extraction with progress bar for validation data
        print("Extracting GLCM features for validation data...")
        val_features = []
        for idx in tqdm(range(len(val_data["images"])), desc="Validation GLCM Extraction"):
            val_features.append(self.extract_glcm_features([val_data["images"][idx]], [val_data.get("rois", [])[idx]]))
        val_features = np.array(val_features).squeeze()

        epochs = self.config.epochs  # Use the epochs attribute from Config
        best_accuracy = 0
        patience = 3  # Number of epochs to wait for improvement
        patience_counter = 0

        # Initialize tqdm progress bar for training epochs
        with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(epochs):
                # Fit the model
                self.model.fit(train_features, train_data["labels"])

                # Evaluate on validation data
                val_predictions = self.model.predict(val_features)
                accuracy = accuracy_score(val_data["labels"], val_predictions)
                print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {accuracy:.2f}")

                # Update the progress bar
                pbar.set_postfix({"Validation Accuracy": f"{accuracy:.2f}"})
                pbar.update(1)

                # Check for improvement
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                    # Optionally save the best model
                    joblib.dump(self.model, "best_model.pkl")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def predict(self, input_data, rois=None):
      """
      Make predictions using the trained model.
      """
      if self.model is None:
          raise ValueError("Model is not trained yet. Train the model before making predictions.")
      input_features = self.extract_glcm_features(input_data, rois)
      return self.model.predict(input_features)

    def evaluate(self, test_data):
        """
        Evaluate the model's performance on test data.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before evaluation.")
        test_features = self.extract_glcm_features(test_data["images"], test_data.get("rois"))
        predictions = self.model.predict(test_features)
        accuracy = accuracy_score(test_data["labels"], predictions)
        report = classification_report(test_data["labels"], predictions)
        return accuracy, report

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before saving.")
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from a file.
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
