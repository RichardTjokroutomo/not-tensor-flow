import tensorflow as tf
import numpy as np

# fetch training data
# ===============================================================================================
def fetch_training_data():
    datasets = tf.keras.datasets.mnist.load_data()
    (input_train, output_train) = datasets[0]

    return (input_train, output_train)

# fetch test data
# ===============================================================================================
def fetch_test_data():
    datasets = tf.keras.datasets.mnist.load_data()
    (input_test, output_test) = datasets[1]

    return (input_test, output_test)


