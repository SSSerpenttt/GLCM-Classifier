import numpy as np

class Config:
    def __init__(self, epochs=10, data_path="path/to/your/dataset"):
        """
        Initialize the configuration with default or user-defined values.
        """
        self.data_path = data_path
        self.model_params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42
        }
        self.distances = [1, 2, 3]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.levels = 256
        self.epochs = epochs

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