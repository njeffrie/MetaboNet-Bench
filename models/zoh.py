import numpy as np


class ZeroOrderHold:
    def predict(self, timestamps, cgm, insulin, carbs):
        return np.array([cgm[-1] for _ in range(12)])