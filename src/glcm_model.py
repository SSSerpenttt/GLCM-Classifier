import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
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
        self.model = GradientBoostingClassifier(**config.model_params)
        self.data = None
        self.mlb = None  # MultiLabelBinarizer instance

    def extract_glcm_features(self, images, rois=None):
        """
        Extract GLCM features from a list of grayscale images, optionally using multiple ROIs.
        """
        features = []
        for idx, image in enumerate(images):
            if len(image.shape) == 3:  # Convert to grayscale if RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if rois is not None and idx < len(rois):
                image_features = []
                for roi in rois[idx]:
                    x, y, w, h = map(int, roi)
                    cropped_image = image[y:y+h, x:x+w]

                    glcm = graycomatrix(
                        cropped_image,
                        distances=self.config.distances,
                        angles=self.config.angles,
                        levels=self.config.levels,
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
                    image_features.append(roi_features)

                features.append(np.mean(image_features, axis=0))
            else:
                glcm = graycomatrix(
                    image,
                    distances=self.config.distances,
                    angles=self.config.angles,
                    levels=self.config.levels,
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

    def preprocess_labels(self, labels):
        """
        Preprocess labels using MultiLabelBinarizer for multi-label data.
        """
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            labels = self.mlb.fit_transform(labels)
        else:
            labels = self.mlb.transform(labels)

        # Flatten labels for multi-class classification if necessary
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)

        return labels

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
                self.model.fit(train_features, train_labels)

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
      """
      if self.model is None:
          raise ValueError("Model is not trained yet. Train the model before making predictions.")
      input_features = self.extract_glcm_features(input_data, rois)
      return self.model.predict(input_features)

    def evaluate(self, test_data):
        """
        Evaluate the model's performance on test data and display example predictions with visualizations.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before evaluation.")

        print("Extracting GLCM features for test data...")
        test_features = []
        for idx in tqdm(range(len(test_data["images"])), desc="Testing Progress"):
            test_features.append(self.extract_glcm_features([test_data["images"][idx]], [test_data.get("rois", [])[idx]]))
        test_features = np.array(test_features).squeeze()

        test_labels = self.preprocess_labels(test_data["labels"])
        predictions = self.model.predict(test_features)

        accuracy = accuracy_score(test_labels, predictions)
        print(f"Test Accuracy: {accuracy:.2f}")

        report = classification_report(test_labels, predictions, target_names=self.mlb.classes_)
        print("Classification Report:\n", report)

        # Display example predictions with visualizations
        print("\nExample Predictions with Visualizations:")
        num_examples = min(5, len(predictions))  # Show up to 5 examples
        for i in range(num_examples):
            plt.figure(figsize=(4, 4))
            plt.imshow(test_data["images"][i], cmap="gray")
            plt.title(f"Predicted: {predictions[i]}\nGround Truth: {test_data['labels'][i]}")
            plt.axis("off")
            plt.show()

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
