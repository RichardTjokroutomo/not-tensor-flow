import numpy as np
import random


import losses
import testing


# training
# ===============================================================================================
def SGD(model, input_train, output_train, epoch, batch_size, learning_rate, test_mode=False, input_test=None, output_test=None):
    # prepping stuff
    loss_func = losses.LossFunctions()
    dataset = list(zip(input_train, output_train))
    model_size = len(model)

    for i in range(epoch):
        print("epoch " + str(i+1) + " begins")
        random.shuffle(dataset)
        batches = [dataset[b:b+batch_size] for b in range(0, len(dataset), batch_size)]
        
        # testing the initial accuracy
        if test_mode:
            result = testing.test_model(model, input_test, output_test)
            test_len = len(input_test)
            result = float(result / test_len)
            print("initial accuracy: " + str(result))
            print("")

        # begin processing the batch
        counter = 0
        for batch in batches:
            counter += 1
            model_nabla_w = []
            model_nabla_b = []
            batch_len = len(batch)

            # initializing the nablas
            for layer in model:
                if layer.contains_tunable_params():
                    (w, b) = layer.get_params()
                    model_nabla_w.append(np.zeros_like(w))
                    model_nabla_b.append(np.zeros_like(b))
                else:  # doesnt matter, just a dummy
                    model_nabla_w.append(np.zeros((1)))
                    model_nabla_b.append(np.zeros((1)))

            # train on each individual training data
            orig_delta = None
            for data in batch:
                (x, y) = data

                # feedforward
                activation = x
                for layer in model:
                    activation = layer.feedforward(activation)
                
                # backprop
                delta = loss_func.binary_cross_entropy_prime(activation, y)   # WE ARE CURRENTLY USING SE
                for j in range(model_size-1, 0, -1):
                    cur_layer = model[j]
                    if cur_layer.contains_tunable_params():
                        (new_delta, weights, biases) = cur_layer.backprop(delta)
                        delta = new_delta
                        model_nabla_w[j] += weights
                        model_nabla_b[j] += biases
                    else:
                        new_delta = cur_layer.backprop(delta)
                        delta = new_delta

            multiplier = learning_rate / batch_len
            for k in range(model_size):
                if model[k].contains_tunable_params():
                    model[k].update_params( model_nabla_w[k]*multiplier, model_nabla_b[k]*multiplier)



