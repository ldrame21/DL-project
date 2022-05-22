from torch import FloatTensor
from math import tanh
from others.module import Module
'''
class MSE(Module):
    def __init__(self, input_size):

        pass 
    def forward(self, input, target):
        self.input = input #general output y of the network
        self.target = target
        #(input-target).pow(2).sum()/input.size()[0]
        return ((input-target)**2).mean()
    def backward(self, gradwrtoutput):
        return 2*(self.input-self.target).sum()
    def param(self):
        """
        :return: empty list since the activation layers have not parameters
        """
        return []
'''
class ReLU(Module):
    def __init__(self, input_size):
        """
        :param input_size: input size of the activation layer or output size
        """
        self.hidden_size = input_size
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(input_size)
        self.grad_wrt_input = FloatTensor(input_size)

    def forward(self, input):
        """
        Forward pass.
        :param input:
        :return: 
        """
        self.input = input
        return self.input.maximum(input, 0) 
    def backward(self, gradwrtoutput):
        """
        Backward pass.
        :param gradwrtoutput:
        :return: 
        """
        deriv = self.input
        deriv[deriv<=0] = 0
        deriv[deriv>0] = 1
        return gradwrtoutput*deriv
    def param(self):
        """
        :return: empty list since the activation layers have not parameters
        """
        return []
        
class Sigmoid(Module):
     def __init__(self, input_size):
            """
        :param input_size: input size of the activation layer or output size
        """
        self.hidden_size = input_size
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(input_size)
        self.grad_wrt_input = FloatTensor(input_size)
        
    def forward(self, input):
        self.output = 1 / (1 + torch.exp(-input))
        return self.output
    def backward(self, gradwrtoutput):
        return gradwrtoutput * ( 1 - self.output) * self.output
    def param(self):
        return []

'''
class NearestUpsampling(Module):
    def __init__(self):
        pass 
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
'''
'''
class SGD(Module):
    def __init__(self):
        pass 
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
'''
'''
class Sequential(input):
    def __init__(self):
        pass 
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
'''
        