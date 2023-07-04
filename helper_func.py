import numpy as np


# helper functions
# ===============================================================================================
def find_max(input_np_arr):
        curr_max = (input_np_arr[0])[0]
        curr_max_index = 0

        for i in range(10):
            if curr_max < input_np_arr[i][0]:
                curr_max = input_np_arr[i][0]
                curr_max_index = i

        return curr_max_index

def find_hotkey(y_output):
    correct_ans = 0
    for i in range(10):
        if y_output[i] == 1:
            correct_ans = i

    return correct_ans

def hotkey(y):
    y_hotkey = np.zeros((10, 1))
    y_hotkey[y] = 1.0

    return y_hotkey
