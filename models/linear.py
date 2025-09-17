import numpy as np
from sklearn.linear_model import LinearRegression as SKLLinearRegression

class LinearRegression:
    def __init__(self, lookback_window: int):
        # Lookback window is in minutes, so we divide by 5 to get the number of 5-minute intervals
        self.lookback_window = lookback_window // 5
        self.model = SKLLinearRegression()

    def predict(self, subject_id, timestamps, input_glucose):
        # Get the last lookback_window intervals of glucose data
        past_glucose = input_glucose[[-self.lookback_window, -1]]
        
        # Create time features (5-minute intervals)
        time_features = np.arange(len(past_glucose)).reshape(-1, 1)
        
        # Fit the linear regression model
        self.model.fit(time_features, past_glucose)
        
        # Predict the next 12 intervals (next 60 minutes)
        future_time_features = np.arange(len(past_glucose), len(past_glucose) + 12).reshape(-1, 1)
        predictions = self.model.predict(future_time_features)
        
        return predictions