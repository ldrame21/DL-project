from torch import FloatTensor
from math import tanh
from Proj_287630_282604_288453.Miniproject_2.module import Module

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
    def __init__(self):
        """
        :param input_size: input size of the activation layer (equivalent to output size in activation layers)
        """
        #self.hidden_size = input_size
        #self.input = FloatTensor(input_size)
        #self.output = FloatTensor(input_size)
        #self.gradwrtinput = FloatTensor(input_size)

    def __call__(self,input):
      self.forward(input)
      return self.output

    def forward(self, input):
      """
      Forward pass.
      :param input: tensor of hidden size
      :return: tensor of hidden_size shape containing the element-wise ReLU of the input tensor
      """
      self.input = input
      self.output = self.input
      self.output[self.output<=0] = 0
      self.output[self.output>0] = 1
      return self.output

    def backward(self, gradwrtoutput):
        """
        Backward pass.
        :param gradwrtoutput: tensor of hidden_size shape representing the gradient with respect to the ouput of the layer
        :return: tensor of hidden_size shape representing the gradient with respect to the input of the layer
        """
        deriv = self.input
        deriv[deriv<=0] = 0
        deriv[deriv>0] = 1
        self.gradwrtinput = gradwrtoutput*deriv
        return self.gradwrtinput

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
        return []
        
class Sigmoid(Module):
    def __init__(self):
      """
      :param input_size: input size of the activation layer or output size
      """
      #self.hidden_size = input_size
      #self.input = FloatTensor(input_size)
      #self.output = FloatTensor(input_size)
      #self.gradwrtinput = FloatTensor(input_size) 
       
    def __call__(self,input):
      self.forward(input)
      return self.output

    def forward(self, input):
        """
        Forward pass.
        :param input: tensor of hidden_size shape
        :return: tensor of hidden_size shapee containing the element-wise Sigmoid of the input tensor
        """
        self.input = input
        self.output = 1 / (1 + (-self.input).exp())
        return self.output

    def backward(self, gradwrtoutput):
        """
        Backward pass.
        :param gradwrtoutput: tensor of hidden_size shape representing the gradient with respect to the ouput of the layer
        :return: tensor of hidden_size shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtinput = gradwrtoutput * ( 1 - self.output) * self.output
        return self.gradwrtinput 

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
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
        