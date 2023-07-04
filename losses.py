import numpy as np

# Loss functions
# ===============================================================================================
class LossFunctions():
    def __init__(self):
        pass

    def se(self, activation, expected_output):
        return activation - expected_output

    def mse(self, activation, expected_output):
        (n, y) = activation.shape   # y = 1
        return 2*(activation - expected_output) / n

    def binary_cross_entropy_prime(self, activation, expected_output):
        pred_output = activation
        pred_output = np.where(pred_output == 0.0, 0.1, pred_output)
        pred_output = np.where(pred_output == 1.0, 1.1, pred_output)  # to prevent divide by zero
        return ((1 - expected_output) / (1 - pred_output) - expected_output / pred_output)
     