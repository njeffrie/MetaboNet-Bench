import numpy as np


class ZeroOrderHold:
    def predict(self, timestamps, cgm, insulin, carbs):
        return cgm[:, -1].reshape(-1, 1).repeat(12, axis=1)