import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from skimage.feature import graycomatrix, graycoprops
import cv2  # For image processing
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt  # Add this import at the top of the file
from joblib import Parallel, delayed
import random
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.callback import EarlyStopping
import pandas as pd
from IPython.display import display
import gc
import scipy.stats

class GLCMModel:
    def __init__(self, config):
        """
        Initialize the GLCMModel with a configuration object.
        """
        self.config = config
        self.classifier_type = config.classifier_type
        # Classifier selection
        if self.classifier_type == "lightgbm":
            self.model = LGBMClassifier(**config.model_params)
        elif self.classifier_type == "randomforest":
            self.model = RandomForestClassifier(**config.model_params)
        elif self.classifier_type == "xgboost":
            self.model = XGBClassifier(**config.model_params)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
        
        self.data = None
        self.mlb = None  # MultiLabelBinarizer instance

        print(f"Using {self.classifier_type} as the classifier.")


    @staticmethod
    def preprocess_image(image, clipLimit=2.0, tileGridSize=(8, 8)):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
        """
        # Ensure image is 8-bit unsigned integer type, as required by CLAHE
        if image.dtype != np.uint8:
            # Scale to 0-255 if it's float or other types, then convert
            if np.max(image) > 1.0: # Assuming it's not already normalized to [0, 1]
                image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            else: # Assuming it's normalized to [0, 1]
                image = image * 255
            image = image.astype(np.uint8)

        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        # Apply CLAHE to the image
        processed_image = clahe.apply(image)

        return processed_image

      

    def extract_glcm_features(self, images, rois=None):
        """
        Extract summarized GLCM features from a list of grayscale images.
        For each ROI:
            - Compute full ROI GLCM features.
            - Divide ROI into 3x3 patches and compute GLCM features.
            - Compute summary statistics (mean, median, std, min, max, skewness) over patch features.
            - Concatenate ROI features with the summarized patch features.
        """

        def process_roi(image, roi, img_idx, roi_idx):
            x, y, w, h = map(int, roi)
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))

            cropped_image = image[y:y + h, x:x + w]
            if cropped_image.size == 0:
                return None, None

            def compute_glcm_features(region):
                min_dim = min(region.shape[:2])
                distances = [1, max(1, min_dim // 4)]
                angles = self.config.angles
                levels = self.config.levels

                glcm = graycomatrix(region, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
                features = [graycoprops(glcm, prop).flatten() for prop in
                            ['contrast', 'homogeneity']]

                entropy_vals, max_prob_vals, variance_vals, diff_entropy_vals = [], [], [], []
                i_vals, j_vals = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')

                for i in range(glcm.shape[2]):
                    for j in range(glcm.shape[3]):
                        slice_glcm = glcm[:, :, i, j]
                        nonzero = slice_glcm[slice_glcm > 0]
                        entropy_vals.append(-np.sum(nonzero * np.log2(nonzero)))
                        max_prob_vals.append(np.max(slice_glcm))
                        mean = np.sum(i_vals * slice_glcm)
                        variance_vals.append(np.sum(slice_glcm * ((i_vals - mean) ** 2)))
                        abs_diff = np.abs(i_vals - j_vals)
                        diff_hist = np.zeros(levels)
                        np.add.at(diff_hist, abs_diff.ravel(), slice_glcm.ravel())
                        diff_nonzero = diff_hist[diff_hist > 0]
                        diff_entropy_vals.append(-np.sum(diff_nonzero * np.log2(diff_nonzero)))

                del glcm, i_vals, j_vals, abs_diff, diff_hist, diff_nonzero
                gc.collect()

                return np.hstack(features + [entropy_vals, max_prob_vals, variance_vals, diff_entropy_vals])

            cropped_image = self.preprocess_image(cropped_image)
            full_features = compute_glcm_features(cropped_image)

            # Divide ROI into 3x3 patches
            patch_w = max(1, w // 3)
            patch_h = max(1, h // 3)

            def extract_patch_feature(col, row):
                px = x + col * patch_w
                py = y + row * patch_h
                pw = min(patch_w, image.shape[1] - px)
                ph = min(patch_h, image.shape[0] - py)
                patch = image[py:py + ph, px:px + pw]
                if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
                    return np.zeros_like(full_features)
                patch = self.preprocess_image(patch)
                return compute_glcm_features(patch)

            patch_features = Parallel(n_jobs=9, backend='threading')(
                delayed(extract_patch_feature)(col, row)
                for row in range(3) for col in range(3)
            )

            patch_features_arr = np.vstack(patch_features)

            # Compute statistical summaries across patches
            mean_feat = np.mean(patch_features_arr, axis=0)
            median_feat = np.median(patch_features_arr, axis=0)
            std_feat = np.std(patch_features_arr, axis=0)
            min_feat = np.min(patch_features_arr, axis=0)
            max_feat = np.max(patch_features_arr, axis=0)
            skew_feat = scipy.stats.skew(patch_features_arr, axis=0)
            skew_feat = np.nan_to_num(skew_feat, nan=0.0)

            # Combine summary stats and full ROI features
            summary_stats = np.hstack([mean_feat, median_feat, std_feat, min_feat, max_feat, skew_feat])
            roi_features = np.hstack([full_features, summary_stats])

            del cropped_image, patch_features, full_features, patch_features_arr
            gc.collect()

            return roi_features, (img_idx, roi_idx)

        # Create tasks
        tasks = [
            (image, roi, img_idx, roi_idx)
            for img_idx, image in enumerate(images)
            if rois and img_idx < len(rois)
            for roi_idx, roi in enumerate(rois[img_idx])
        ]

        # Process in parallel
        results = Parallel(n_jobs=4)(delayed(process_roi)(img, roi, i, j)
                                    for img, roi, i, j in tasks)

        # Collect results
        features = []
        valid_roi_indices = []
        for feature, valid_index in results:
            if feature is not None:
                features.append(feature)
                valid_roi_indices.append(valid_index)

        del results, tasks
        gc.collect()

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
            # Convert string labels to binary (1 for shallow, 0 for deep)
            binary_labels = np.array([1 if label == 'depth-shallow' else 0 for label in labels])
            self.mlb.fit([['depth-deep'], ['depth-shallow']])
        else:
            binary_labels = np.array([1 if label == 'depth-shallow' else 0 for label in labels])
        
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




    def create_feature_summary_tables(self, features, labels, save_to_csv=False, prefix="GLCM_Stats"):
        """
        Create summary tables for each statistic with:
        - Rows: class labels (depth classes)
        - Columns: GLCM base features (aggregated over patches)
        """
        import pandas as pd
        from scipy.stats import skew
        from IPython.display import display

        if features.ndim != 2:
            raise ValueError("Features must be a 2D array.")

        base_feature_names = [
            "contrast", "homogeneity", "entropy", "max_prob",
            "variance", "diff_entropy"
        ]

        num_features = features.shape[1]
        num_base = len(base_feature_names)
        if num_features % num_base != 0:
            print("⚠️ Feature count not divisible by base features. Using generic names.")
            feature_names = [f"f{i}" for i in range(num_features)]
        else:
            reps = num_features // num_base
            feature_names = [f"{base}_{i}" for i in range(reps) for base in base_feature_names]

        df = pd.DataFrame(features, columns=feature_names)
        df["Label"] = labels

        grouped = df.groupby("Label")

        # Initialize dict to store aggregated stats per feature
        agg_stats = {stat: pd.DataFrame(index=grouped.groups.keys(), columns=base_feature_names) for stat in ["Mean", "Median", "Std", "Min", "Max", "Skew"]}

        for base_feature in base_feature_names:
            # All columns for this feature (all patches)
            related_cols = [col for col in df.columns if col.startswith(base_feature)]
            subset = df[related_cols + ["Label"]]

            grouped_subset = subset.groupby("Label")

            # Compute all stats per patch per class
            mean_df = grouped_subset.mean()
            median_df = grouped_subset.median()
            std_df = grouped_subset.std()
            min_df = grouped_subset.min()
            max_df = grouped_subset.max()
            skew_df = grouped_subset.apply(lambda g: skew(g.drop(columns="Label"), axis=0)).apply(pd.Series)

            # Aggregate patch stats by taking mean across patches for each class
            agg_stats["Mean"][base_feature] = mean_df.mean(axis=1)
            agg_stats["Median"][base_feature] = median_df.median(axis=1)
            agg_stats["Std"][base_feature] = std_df.mean(axis=1)
            agg_stats["Min"][base_feature] = min_df.min(axis=1)
            agg_stats["Max"][base_feature] = max_df.max(axis=1)
            agg_stats["Skew"][base_feature] = skew_df.mean(axis=1)

        for stat_name, stat_df in agg_stats.items():
            stat_df.index.name = "Class"
            print(f"\n📊 GLCM Feature {stat_name} by Depth Class:")
            with pd.option_context("display.max_columns", None, "display.precision", 4):
                display(stat_df)

            if save_to_csv:
                filename = f"{prefix}_{stat_name.lower()}.csv"
                stat_df.to_csv(filename)
                print(f"💾 Saved {stat_name} stats to {filename}")

        return agg_stats





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

        print("\n📋 GLCM Feature Summary by Class (Training Set):")
        self.create_feature_summary_tables(train_features, train_labels)

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

        print("\n📋 GLCM Feature Summary by Class (Validation Set):")
        self.create_feature_summary_tables(train_features, train_labels)

        print("Starting training...")
        
        classifier = self.classifier_type.lower()

        if classifier == "lightgbm":
            best_score = float("-inf")  # for aucpr, higher is better
            best_metric = "aucpr"       # change to "logloss" or others if needed

            def save_best_callback(env):
                nonlocal best_score
                # env.evaluation_result_list: List of tuples like ('valid_0', 'metric_name', score, is_higher_better)
                for name, metric, score, _ in env.evaluation_result_list:
                    if metric == best_metric:
                        if score > best_score:
                            best_score = score
                            print(f"📈 New best {best_metric}: {best_score:.5f}. Saving model...")
                            self.save_model("lightgbm.best_trained-glcm_model.txt", "lightgbm_mlb.json")
                        break  # Stop once target metric is found

            self.model.fit(
                train_features,
                train_labels,
                eval_set=[(val_features, val_labels)],
                eval_metric=["logloss", "error", "auc", "aucpr"],
                callbacks=[
                    early_stopping(self.config.early_stopping_rounds),
                    log_evaluation(period=1),
                    save_best_callback
                ]
            )
            self.save_model("lightgbm.best_trained-glcm_model.txt", "/content/lightgbm.mlb.json")

        elif classifier == "xgboost":
            self.model.fit(
                train_features,
                train_labels,
                eval_set=[(val_features, val_labels)],
                verbose=True
            )

            # Retrieve evals_result after training
            # Retrieve evals_result after training
            evals_result = self.model.evals_result()

            # Determine the dataset names (usually 'validation_0' and 'validation_1')
            eval_set_names = list(evals_result.keys())
            primary_val_set = eval_set_names[0]  # usually 'validation_0'

            # Retrieve best iteration
            best_iter = self.model.best_iteration

            # Display all evaluation metrics at the best iteration
            print(f"\n📊 Evaluation Metrics at Best Iteration ({best_iter}):")
            for metric in self.model.eval_metric:
                if metric in evals_result[primary_val_set]:
                    best_value = evals_result[primary_val_set][metric][best_iter]
                    print(f"✅ {metric.upper()}: {best_value:.4f}")
                else:
                    print(f"⚠️  Metric '{metric}' not found in evals_result.")

            # Save the best model and feature importance if needed
            self.save_model("xgboost.best_trained-glcm_model.txt", "xgboost_mlb.json")

        elif classifier == "randomforest":
            self.model.fit(train_features, train_labels)

            # Predict probabilities and classes for validation set
            val_probs = self.model.predict_proba(val_features)
            val_preds = np.argmax(val_probs, axis=1) if val_probs.shape[1] > 1 else (val_probs > 0.5).astype(int)
            
            # If using MultiLabelBinarizer
            if hasattr(self, "mlb") and self.mlb:
                val_labels_bin = self.preprocess_labels(val_labels)  # binary labels
            else:
                val_labels_bin = val_labels

            # Check class balance in validation set
            unique_val_classes = np.unique(val_labels_bin)
            if len(unique_val_classes) < 2:
                print("⚠️ Only one class present in validation labels. Skipping logloss, AUC, AUCPR.")
                logloss = auc = aucpr = gini = float('nan')
            else:
                logloss = log_loss(val_labels_bin, val_probs, labels=np.unique(train_labels))
                auc = roc_auc_score(val_labels_bin, val_probs[:, 1])
                aucpr = average_precision_score(val_labels_bin, val_probs[:, 1])
                gini = 2 * auc - 1

            accuracy = accuracy_score(val_labels_bin, val_preds)

            print("\n📊 Validation Metrics (RandomForest):")
            print(f"✅ Accuracy : {accuracy:.4f}")
            print(f"✅ LogLoss  : {logloss:.4f}")
            print(f"✅ AUC      : {auc:.4f}")
            print(f"✅ GINI     : {gini:.4f}")
            print(f"✅ AUCPR    : {aucpr:.4f}")

            # Save the model and label binarizer
            self.save_model("randomforest.best_trained-glcm_model.txt", "randomforest_mlb.json")

        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

        print("✅ Training completed.")

        # Log number of trees if available
        if hasattr(self.model, "booster_"):
            print("🌲 Number of trees in trained model:", self.model.booster_.num_trees())
        elif hasattr(self.model, "estimators_"):
            print("🌲 Number of trees in trained model:", len(self.model.estimators_))



    def predict(self, images, rois):
        """
        Make predictions using the trained model.
        The model will infer depth labels for each ROI based on GLCM features.
        Also displays summarized GLCM feature means.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Train the model before making predictions.")

        print("Extracting GLCM features for the specified data...")
        input_features = []
        valid_roi_indices = []

        for idx in tqdm(range(len(images)), desc="Prediction Progress"):
            features, local_indices = self.extract_glcm_features([images[idx]], [rois[idx]])
            input_features.extend(features)
            valid_roi_indices.extend([(idx, roi_idx) for (_, roi_idx) in local_indices])

        input_features = np.array(input_features)
        print(f"[Debug] Extracted input features shape: {input_features.shape}")
        print(f"[Debug] Total ROI predictions expected: {len(input_features)}")

        predictions = self.model.predict(input_features)

        mapped_predictions = [[] for _ in range(len(images))]
        for (img_idx, roi_idx), prediction in zip(valid_roi_indices, predictions):
            mapped_predictions[img_idx].append(prediction)

        sampled_image_indices = random.sample(range(len(images)), min(5, len(images)))

        for img_idx in sampled_image_indices:
            print(f"\n📷 Predictions for Image {img_idx + 1}:")
            image = images[img_idx]
            image_rois = rois[img_idx]
            predicted_labels = mapped_predictions[img_idx]

            if not predicted_labels:
                print("⚠️ No valid ROIs were processed for this image.")
                continue

            print(f"ROIs: {image_rois}")
            print(f"Predicted labels (numeric): {predicted_labels}")
            if self.mlb:
                label_names = [self.mlb.classes_[p] for p in predicted_labels]
                print(f"Predicted labels (named): {label_names}")
            else:
                label_names = [str(p) for p in predicted_labels]
                print("🔍 Note: MultiLabelBinarizer not initialized yet, showing numeric labels only.")

            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            axs[0].imshow(image, cmap="gray")
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(image, cmap="gray")
            axs[1].set_title(f"Predictions for Image {img_idx + 1}")
            axs[1].axis("off")

            glcm_table_rows = []

            sampled_data = list(zip(image_rois, predicted_labels))
            sampled_features = [input_features[i] for i, (img_i, _) in enumerate(valid_roi_indices) if img_i == img_idx]

            for i, (roi, pred_label) in enumerate(sampled_data[:5]):
                x, y, w, h = map(int, roi)
                label_name = self.mlb.classes_[pred_label] if self.mlb else str(pred_label)
                axs[1].add_patch(plt.Rectangle((x, y), w, h, edgecolor="blue", facecolor="none", linewidth=1.5))
                axs[1].text(
                    x, y - 5,
                    f"Pred: {label_name}",
                    color="blue",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
                )

                glcm_row = {
                    "ROI": f"[{x}, {y}, {w}, {h}]",
                    "Predicted Label": label_name
                }

                if i < len(sampled_features):
                    full_vector = sampled_features[i]
                    full_length = full_vector.shape[0]

                    num_stats = 6  # mean, median, std, min, max, skew
                    roi_feature_len = full_length // (num_stats + 1)

                    if full_length != roi_feature_len * (num_stats + 1):
                        print(f"⚠️ Feature split mismatch. Total: {full_length}, expected: {(num_stats + 1) * roi_feature_len}")
                        glcm_row["feature_mean"] = round(full_vector.mean(), 4)
                    else:
                        roi_features = full_vector[:roi_feature_len]
                        stats_block = full_vector[roi_feature_len:]

                        stats_matrix = stats_block.reshape(num_stats, roi_feature_len)
                        full_stack = np.vstack([roi_features, stats_matrix])
                        mean_metrics = full_stack.mean(axis=0)

                        # Group by GLCM feature names
                        base_feature_names = ["contrast", "homogeneity", "entropy", "max_prob", "variance", "diff_entropy"]
                        num_base = len(base_feature_names)

                        if roi_feature_len % num_base != 0:
                            print("⚠️ Cannot group features cleanly into GLCM metrics. Showing raw indices instead.")
                            for j in range(mean_metrics.shape[0]):
                                glcm_row[f"feature_{j}_mean"] = round(mean_metrics[j], 4)
                        else:
                            values_per_metric = roi_feature_len // num_base
                            for i, name in enumerate(base_feature_names):
                                start = i * values_per_metric
                                end = start + values_per_metric
                                glcm_row[f"{name}_mean"] = round(mean_metrics[start:end].mean(), 4)

                glcm_table_rows.append(glcm_row)

            plt.tight_layout()
            plt.show()

            glcm_df = pd.DataFrame(glcm_table_rows)
            print("📋 GLCM Feature Summary Table:")
            display(glcm_df)

        return {
            "predictions": mapped_predictions,
            "rois": rois,
            "images": images,
            "valid_roi_indices": valid_roi_indices,
            "features": input_features
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
        valid_roi_indices = predictions_data["valid_roi_indices"]
        original_gt_labels = [labels[img_idx][roi_idx] for img_idx, roi_idx in valid_roi_indices]

        test_labels_numerical = self.preprocess_labels(original_gt_labels)

        # Flatten predictions for evaluation
        flat_predictions = [pred for preds in predictions for pred in preds]  # For metrics
        reshaped_predictions = predictions  # Already per-image from predict()

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
        plt.title("Confusion Matrix for GLCM model ({self.classifier_type} classifier)")
        plt.show()

        # Calculate mAP
        average_precision = average_precision_score(test_labels_numerical, flat_predictions)
        print(f"Mean Average Precision (mAP): {average_precision:.2f}")
        
        for img_idx in random.sample(range(len(images)), 5):
            print(f"Visualizing predictions vs ground truth for image {img_idx + 1}...")
            image = images[img_idx]
            image_rois = rois[img_idx]
            ground_truth_labels_strings = labels[img_idx]
            predicted_labels_numerical = reshaped_predictions[img_idx]

            print(f"ROIs for image {img_idx}: {image_rois}")
            print(f"GT labels: {ground_truth_labels_strings}")
            print(f"Predictions: {predicted_labels_numerical}")

            # Create a figure with 2 subplots: original + annotated
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))

            axs[0].imshow(image)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            # Plot the image with ROIs and labels
            axs[1].imshow(image, cmap="gray")
            axs[1].set_title(f"Image {img_idx + 1}: Predictions vs Ground Truth")

            for roi_idx, (roi, gt_label_string, pred_label_numerical) in enumerate(zip(image_rois, ground_truth_labels_strings, predicted_labels_numerical)):
                x, y, w, h = map(int, roi)
                gt_label_numerical = 1 if gt_label_string == 'depth-shallow' else 0
                gt_label_name = self.mlb.classes_[gt_label_numerical]
                pred_label_name = self.mlb.classes_[pred_label_numerical]
                color = "green" if gt_label_numerical == pred_label_numerical else "red"
                axs[1].add_patch(plt.Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=1))
                axs[1].text(
                    x, y - 5,
                    f"GT: {gt_label_name}\nPred: {pred_label_name}",
                    color=color,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
              )
            axs[1].axis("off")

            plt.tight_layout()
            plt.show()

        features = np.array(predictions_data["features"])
        labels = np.array(test_labels_numerical)

        print("\n📋 GLCM Feature Summary by Class (from Evaluation Set):")
        self.create_feature_summary_tables(
            features,
            labels,
            save_to_csv=False,
            prefix=f"GLCM_Stats_{self.classifier_type}"
        )

        # Return accuracy, report, and predictions
        return accuracy, report, predictions




    def save_model(self, model_path, mlb_path):
        """
        Save the full LightGBM classifier (not just booster) and MultiLabelBinarizer.
        """
        joblib.dump(self.model, model_path)
        print(f"✅ Full model saved to {model_path}")

        mlb_data = {
            "classes": self.mlb.classes_.tolist()
        }
        with open(mlb_path, 'w') as f:
            json.dump(mlb_data, f)
        print(f"✅ MultiLabelBinarizer saved to {mlb_path}")


    
    
    def load_model(self, model_path, mlb_path):
        """
        Load the full LightGBM classifier and MultiLabelBinarizer.
        """
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")

        with open(mlb_path, 'r') as f:
            mlb_data = json.load(f)
        self.mlb = MultiLabelBinarizer(classes=mlb_data["classes"])
        self.mlb.fit([])  # dummy fit to restore internal state
        print(f"✅ MultiLabelBinarizer loaded from {mlb_path}")
