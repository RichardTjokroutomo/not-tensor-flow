# importing modules
import numpy as np
import mnist_loader, helper_func, optimizers, layers, activations, testing



# loading the data
# ===============================================================================================
(input_train, output_train) = mnist_loader.fetch_training_data()
(input_test, output_test) = mnist_loader.fetch_test_data()

# processing the data
input_train = input_train.astype("float32") / 255.0
input_train = [np.reshape(x_train, (1, 28, 28)) for x_train in input_train]
output_train = [helper_func.hotkey(y_train) for y_train in output_train]

input_test = input_test.astype("float32") / 255.0
input_test = [np.reshape(x_test, (1, 28, 28)) for x_test in input_test]
output_test = [helper_func.hotkey(y_test) for y_test in output_test]



# make the model
# ===============================================================================================
my_model = [

    layers.Flatten(),
    layers.DenseLayer(784, 128),
    activations.Sigmoid(),

    layers.DenseLayer(128, 10),
    activations.Sigmoid()
]

# train
# ===============================================================================================

print("")
print("begin training")
print("======================================================================================")
print("")
# model
optimizers.SGD(my_model, input_train, output_train, 30, 10, 3, True, input_test, output_test)  # lr = 1 is optimal for se
print("======================================================================================")
print("training finished")

print("")
print("begin testing")
result = testing.test_model(my_model, input_test, output_test)
print("result: " + str(result) + "/" + str(len(input_test)))
print("finished!")