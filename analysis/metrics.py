import numpy as np


def calculate_rmse(pred, label):
    """Calculate Root Mean Square Error between predictions and labels."""
    return np.sqrt(np.mean((pred - label)**2))