import numpy as np

class Activation():
    def __init__(self):
        pass

    def contains_tunable_params(self):
        return False
    
    def feedforward(self, prev_layer):
        self.z_nodes = prev_layer
        return self.activation(self.z_nodes)

    def backprop(self, delta):
        new_delta = delta * self.activation_prime(self.z_nodes)
        return new_delta


class Sigmoid(Activation):
    def __init__(self):
        super().__init__() # does nothing for now, just for completeness

    def activation(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def activation_prime(self, z):
        return (self.activation(z)*(1-self.activation(z)))
    

class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def activation(self, z):
        return np.tanh(z)

    def activation_prime(self, z):
        return 1-np.tanh(z)**2
    

class LeakyReLu(Activation):
    def __init__(self):
        super().__init__()
    
    def activation(self, z):
        return np.maximum(z, 0.01*z)
    
    def activation_prime(self, z):
        return np.where(z > 0, 1, -0.01)
