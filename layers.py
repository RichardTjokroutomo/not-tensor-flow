import numpy as np
import scipy as sp

class Layer():
    def __init__():
        pass
    
    def contains_tunable_params(self):
        return False
    
    def feedforward(self):
        pass

    def backprop(self):
        pass


# Dense layer
# ===============================================================================================
class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim): 
        input_nodes = input_dim
        output_nodes = output_dim
        self.weights = np.random.randn(output_nodes, input_nodes)
        self.biases = np.random.randn(output_nodes, 1)
        self.prev_layer = np.zeros((input_dim, 1))

    def contains_tunable_params(self):
        return True
    
    def feedforward(self, prev_layer):  # returns z. do activation later
        self.prev_layer = prev_layer
        return np.matmul(self.weights, prev_layer) + self.biases  

    def backprop(self, delta):
        new_delta = np.matmul(self.weights.transpose(), delta)
        nabla_b = delta.copy()
        nabla_w = np.matmul(delta, self.prev_layer.transpose())
        
        return (new_delta, nabla_w, nabla_b)
    
    def get_params(self):
        return (self.weights.copy(), self.biases.copy())
    
    def update_params(self, nabla_w, nabla_b):  # nabla_w & nabla_b has been factored with learning rate & batch size
        self.weights -= nabla_w
        self.biases -= nabla_b


# Flatten & de Flatten
# ===============================================================================================
class Flatten(Layer):
    def __init__(self):
        pass

    def feedforward(self, prev_layer):
        (channel, x, y) = prev_layer.shape
        self.channel = channel
        self.dim_x = x
        self.dim_y = y
        return prev_layer.reshape((self.dim_x*self.dim_y*self.channel, 1))

    def backprop(self, delta):
        return delta.reshape((self.channel, self.dim_x, self.dim_y))
    

# Convolution
# ===============================================================================================
class ConvLayer():
    def __init__(self, num_of_kernel, kernel_len, input_dim, input_channels, padding=0): # input dim is a tuple
        self.depth = num_of_kernel
        self.kernel_len = kernel_len
        (self.x, self.y) = input_dim
        self.channels = input_channels
        self.padding = padding
        pre_output = 1 - self.kernel_len + 2*padding

        # weight means kernel; 
        # this is done to maintain consistency so all layers can inherit from the same parent class
        self.weights = np.random.randn(self.depth, self.channels, self.kernel_len, self.kernel_len)
        self.biases = np.random.randn(self.depth, self.x + pre_output, self.y + pre_output)
        self.output = np.random.randn(self.depth, self.x + pre_output, self.y + pre_output)

    def contains_tunable_params(self):
        return True
    
    def feedforward(self, prev_layer):
        self.prev_layer = prev_layer
        self.prev_layer = np.pad(self.prev_layer, [(0, 0), (self.padding, self.padding), (self.padding, self.padding)], mode="constant")

        self.output += self.biases
        for i in range(self.depth):
            for j in range(self.channels):
                self.output[i] += sp.signal.correlate2d(self.prev_layer[j], self.weights[i][j], "valid")
        
        return self.output
        

    def backprop(self, delta):
        new_delta = np.zeros_like(self.prev_layer)
        nabla_w = np.zeros_like(self.weights)
        nabla_b = delta.copy()

        for i in range(self.depth):
            for j in range(self.channels):
                nabla_w[i][j] = sp.signal.correlate2d(self.prev_layer[j], delta[i], "valid")
                new_delta[j] += sp.signal.convolve2d(delta[i], self.weights[i][j], "full")
        
        return (new_delta, nabla_w, nabla_b)

    def get_params(self):
        return (self.weights.copy(), self.biases.copy())
    
    def update_params(self, nabla_w, nabla_b):  # nabla_w & nabla_b has been factored with learning rate & batch size
        self.weights -= nabla_w
        self.biases -= nabla_b


# Pooling
# ===============================================================================================
class AvgPool():
    def __init__(self, pool_len, input_dim, channels):
        self.pool_len = pool_len
        (x, y) = input_dim

        self.channels = channels
        self.input_dim_x = x
        self.input_dim_y = y
        self.output_dim_x = x // self.pool_len
        self.output_dim_y = y // self.pool_len

        self.weights = np.full((self.channels, self.channels, self.pool_len, self.pool_len), 0.25)
        self.output = np.zeros((channels, self.output_dim_x, self.output_dim_y))
    
    def feedforward(self, prev_layer):
        self.prev_layer = prev_layer
        self.new_delta = np.zeros_like(self.prev_layer)

        for i in range(self.channels):
            for j in range(self.output_dim_x):
                for k in range(self.output_dim_y):
                    self.output[i][j][k] = self.avg_pool_sum((i,j*self.pool_len,k*self.pool_len))

        return self.output
        
    def backprop(self, delta):
        (c, x, y) = delta.shape
        for i in range(c):
            for j in range(x):
                for k in range(y):
                    self.assign_nabla((i,j,k), delta)

        return self.new_delta
        
    def avg_pool_sum(self, start):
        (c, x, y) = start
        sum = 0

        for i in range(x, x+self.pool_len, 1):
            for j in range(y, y+self.pool_len, 1):
                sum += self.prev_layer[c][i][j]

        sum /= self.pool_len**2
        return sum

    def assign_nabla(self, start, delta):
        (c, x, y) = start
        for i in range(c):
            for j in range(x):
                for k in range(y):
                    cur_val = delta[i][j][k] / self.pool_len**2
                    for delta_x in range(self.pool_len):
                        for delta_y in range(self.pool_len):
                            self.new_delta[i][j*self.pool_len + delta_x][k*self.pool_len + delta_y] = cur_val

