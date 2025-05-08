# GLCM Classification Model

This project implements a GLCM (Gray Level Co-occurrence Matrix) classification model suitable for image classification tasks. The model is designed to be easily configurable and trainable, with a focus on usability in both local environments and Google Colab.

## Project Structure

```
glcm-classification-model
├── src
│   ├── glcm_model.py        # Main GLCM classification model implementation
│   ├── train.py             # Script to orchestrate the training process
│   ├── config.py            # Configuration settings for the model
│   └── utils
│       ├── data_loader.py   # Functions for loading and preprocessing data
│       └── metrics.py       # Functions for evaluating model performance
├── notebooks
│   └── glcm_model_colab.ipynb # Jupyter notebook for Google Colab
├── requirements.txt         # List of dependencies
├── .gitignore               # Files and directories to ignore by Git
└── README.md                # Project documentation
```

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd glcm-classification-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation by running:
   ```bash
   python --version
   pip list
   ```

## Usage

### 1. Configuration
Modify the `src/config.py` file to set hyperparameters, dataset paths, and other configurations as needed. Key parameters include:
- `data_path`: Path to the dataset directory.
- `model_params`: Hyperparameters for the Gradient Boosting Classifier.
- `distances`, `angles`, and `levels`: Parameters for GLCM feature extraction.

### 2. Training
Run the training script to train the model:
```bash
python src/train.py
```

### 3. Google Colab
For a step-by-step guide to training the model in Google Colab:
1. Open the `notebooks/glcm_model_colab.ipynb` file in Google Colab.
2. Follow the instructions in the notebook to configure, train, and evaluate the model.

### 4. Model Saving and Evaluation
After training, the model is saved as a `.pkl` file. You can evaluate the model's performance using the test dataset:
```bash
python src/train.py
```
The script will output metrics such as accuracy and a classification report.

## Dataset Structure

The dataset should be organized as follows:
```
data_path/
├── train/
│   ├── img1.png
│   ├── img2.png
│   ├── annotations.json
├── val/
│   ├── img3.png
│   ├── img4.png
│   ├── annotations.json
└── test/
    ├── img5.png
    ├── img6.png
    ├── annotations.json
```
Each split directory (`train`, `val`, `test`) should contain images and a corresponding `annotations.json` file with the format:
```json
[
    {"filename": "img1.png", "label": 0},
    {"filename": "img2.png", "label": 1}
]
```

## Evaluation

After training, you can evaluate the model's performance using the metrics provided in `src/utils/metrics.py`. Key functions include:
- `calculate_accuracy`: Computes the accuracy of the model.
- `calculate_f1_score`: Computes the F1 score (weighted).

To evaluate the model:
1. Run the evaluation script in the notebook or training script.
2. View metrics such as accuracy and the classification report.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

Special thanks to the contributors and open-source libraries that made this project possible:
- `scikit-learn` for machine learning utilities.
- `scikit-image` for GLCM feature extraction.
- `tqdm` for progress tracking.
- `opencv-python` for image processing.
