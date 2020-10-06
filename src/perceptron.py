import numpy as np

class Perceptron:
    
    def __init__(self,number_of_input,activation_function,derivate_activation_function,weight=None,bias=None,random_weight=True,weight_range=0.2,bias_range=1):
        
        self.n = number_of_input
        self.activation_function = activation_function
        self.derivate_activation_function = derivate_activation_function
        
        if (weight==None):
            if random_weight:
                self.w = weight_range*(2*np.random.rand(self.n) - 1)
            else:
                self.w = np.zeros(self.n)
        
        else:
            if not(weight.shape== (self.n,)):
                raise ValueError(f"weight param should be of shape ({self.n},) but it is {weight.shape}")
            else:
                self.w = weight
        
        if (bias==None):
            if random_weight:
                self.b= bias_range*(2*np.random.rand()-1)
            else:
                self.b = 0
        
        else:
            self.b = bias
    
    
    def activate(self,x):        
        return self.activation_function(self.w.T@x + self.b)
    
    def derivate(self,x):
        return self.derivate_activation_function(self.w.T@x + self.b)

