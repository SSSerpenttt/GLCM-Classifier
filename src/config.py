import numpy as np

class Config:
    def __init__(self, epochs=10, data_path="path/to/your/dataset"):
        """
        Initialize the configuration with default or user-defined values.
        """
        self.data_path = data_path
        self.model_params = {
            "n_estimators": 300,       # Increase the number of trees
            "learning_rate": 0.05,     # Reduce the learning rate for finer updates
            "max_depth": 4,            # Reduce tree depth to prevent overfitting
            "random_state": 42,
            "class_weight": "balanced" # Handle class imbalance
        }
        self.distances = [1, 2, 3, 4, 5]  # Add more distances
        self.angles = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]  # Add more angles
        self.levels = 256  # Keep levels at 256 for 8-bit images
        self.epochs = epochs
        self.early_stopping_rounds = 10  # Early stopping for training
        self.logging_level = "INFO"  # Logging level for debugging

    def display(self):
        """
        Display the configuration settings.
        """
        print("Configuration Settings:")
        print(f"Dataset Path: {self.data_path}")
        print(f"Model Parameters: {self.model_params}")
        print(f"Distances: {self.distances}")
        print(f"Angles: {self.angles}")
        print(f"Levels: {self.levels}")
        print(f"Number of Epochs: {self.epochs}")
        print(f"Early Stopping Rounds: {self.early_stopping_rounds}")
        print(f"Logging Level: {self.logging_level}")