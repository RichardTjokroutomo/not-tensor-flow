import numpy as np
import helper_func

# testing
# ===============================================================================================
def test_model(model, input_test, output_test):
    dataset = list(zip(input_test, output_test))

    correct_ans = 0

    for x, y in dataset:
        # feedforward
        a = x
        for layer in model:
            a = layer.feedforward(a)
        
        # evaluating
        predicted_ans = helper_func.find_max(a)
        actual_ans = helper_func.find_hotkey(y)

        if predicted_ans == actual_ans:
            correct_ans += 1
            

    return correct_ans