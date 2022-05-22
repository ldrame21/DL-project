from torch import FloatTensor
from math import tanh
from Proj_287630_282604_288453.Miniproject_2.module import Module

######## Loss ########

class MSE:
    def __init__(self):
        pass 

    @staticmethod
    def compute_loss(predicted, target):
        """
        :param predicted: tensor of shape (...) consisting of the output of the network, the predictions 
        :param target: tensor of shape (...) consisting of the targets
        :return: loss
        """
        #(input-target).pow(2).sum()/input.size()[0]
        return ((predicted-target)**2).mean(dim=1)

    @staticmethod
    def compute_backward_pass_gradient(predicted, target):
        """
        :param predicted: tensor of shape (...) consisting of the output of the network, the predictions 
        :param target: tensor of shape (...) consisting of the targets
        :return: gradient of the loss wrt to the predictions
        """
        return 2*(predicted-target).sum()

######## Activation layers ########

class ReLU(Module):
    def __init__(self, input_size=0):
        """
        :param input_size: input size of the activation layer (equivalent to output size in activation layers)
        """
        if(not(input_size)): 
            pass
        self.hidden_size = input_size
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(input_size)
        self.gradwrtinput = FloatTensor(input_size)

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
    def __init__(self, input_size=0):
        """
        :param input_size: input size of the activation layer or output size
        """
        if(not(input_size)): 
            pass
        self.hidden_size = input_size
        self.input = FloatTensor(input_size)
        self.output = FloatTensor(input_size)
        self.gradwrtinput = FloatTensor(input_size) 
       
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


######## Optimizer: Stochastiuc Gradient Descent ########
class SGD(Module):
    def __init__(learn_rate=0.1, batch_size=1, n_iter=50, tolerance=1e-06, random_state=None):
        pass 
    def forward(self, input):
        #gradient, x, y, start, 
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []



######## Convolutionnal layers ########
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
YUCEF
'''

######## Container ########
'''
class Sequential(input):
    def __init__(self, loss, input_size):
        """
        :param loss: class instance object with methods compute_loss and compute_grad (cf. losses.py)
        :param input_size: size of input samples of the network
        """
        self.loss = loss
        self.input_size = input_size
        self.layers = []
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
'''
        