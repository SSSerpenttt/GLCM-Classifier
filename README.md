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

## 1.Configuration

The script uses a `Config` class to manage various settings for data loading, GLCM feature extraction, and model training.  You can adjust these parameters to suit your specific dataset and experimental needs.

### Key Parameters

* **`data_path`**:  
    * Description:  Specifies the path to the dataset directory containing the image data.
    * Default: `"path/to/your/dataset"`
    * Usage:  Modify this to point to the actual location of your image files.

* **`model_params`**:  
    * Description:  A dictionary holding the hyperparameters for the LightGBM classifier.
    * Keys:
        * `"n_estimators"`:
            * Description: The number of trees in the forest.
            * Default: `300`
        * `"learning_rate"`:
            * Description: The step size at which the model learns.
            * Default: `0.05`
        * `"max_depth"`:
            * Description: The maximum depth of the trees.
            * Default: `4`
        * `"random_state"`:
            * Description:  Seed for random number generation. Ensures reproducibility.
            * Default: `42`
        * `"class_weight"`:
            * Description: Weights associated with classes. `"balanced"` automatically adjusts weights inversely proportional to class frequencies in the input data.
            * Default: `"balanced"`

* **`distances`**:  
    * Description: A list of pixel distances used in the Gray-Level Co-occurrence Matrix (GLCM) calculation.
    * Default: `[1, 2, 3, 4, 5]`
    * Usage:  These values determine how far apart pixel pairs are when calculating texture features.

* **`angles`**:  
    * Description: A list of angles (in radians) for GLCM calculation, specifying the direction between the pixel pairs.
    * Default: `[0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]`
    * Usage:  Defines the orientations to consider when analyzing texture.

* **`levels`**:  
    * Description: The number of gray levels used to quantize the image for GLCM computation.
    * Default: `256`
    * Usage:  Typically set to 256 for 8-bit images.

* **`epochs`**:  
    * Description: The number of times the entire training dataset is passed forward and backward through the neural network during training.
    * Default: `10`

* **`early_stopping_rounds`**:  
    * Description: The number of training rounds without improvement on the validation set after which training will be stopped.
    * Default: `10`

* **`logging_level`**:  
    * Description:  The level of detail in logging messages (e.g., "INFO", "DEBUG", "WARNING").
    * Default: `"INFO"`

###   Modifying the Configuration

To customize the behavior of the scripts, modify the values within the `Config` class in `src/config.py`.  For example:

```python
from src.config import Config

config = Config()
config.data_path = "/path/to/my/images"
config.model_params["n_estimators"] = 500
config.angles = [0, np.pi/2]
```

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

## Evaluation

Follow run the evaluate cell provided in the notebook. This will only work if:
1. Your newly instantiated model just finished training.
2. You created a new model instance and loaded your pre-trained weights.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
