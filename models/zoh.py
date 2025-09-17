import numpy as np
class ZeroOrderHold:
    def predict(self, subject_id, timestamps, input_glucose):
        return np.array([input_glucose[-1] for _ in range(12)])