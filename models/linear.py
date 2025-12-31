import numpy as np
from sklearn.linear_model import LinearRegression as SKLLinearRegression


class LinearRegression:

    def __init__(self, lookback_window: int):
        # Lookback window is in minutes, so we divide by 5 to get the number of 5-minute intervals
        self.lookback_window = lookback_window // 5

    def predict(self, timestamps, cgm, insulin, carbs):
        # Get the last lookback_window intervals of glucose data
        past_glucose = cgm[[-self.lookback_window-1, -1]]
        slope = (past_glucose[1] - past_glucose[0]) / self.lookback_window
        return np.array([past_glucose[1] + slope * i for i in range(12)])