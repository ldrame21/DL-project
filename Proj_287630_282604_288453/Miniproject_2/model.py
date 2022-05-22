from torch import FloatTensor, random
from math import tanh
from Proj_287630_282604_288453.Miniproject_2.others.module import Module

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
class SGD():
    def __init__(learning_rate=0.1, batch_size=1, random_state=None):
        # paramètre nb_epoch ?
        # tolerance?
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        random.seed()
        
    def param(self):
        """
        :return:
        """
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
class Conv2d(object):
    def __init__(self, channels_in, channels_out, kernel_size, input_shape=None): 
        self.weight = torch.rand(channels_in,channels_out,kernel_size,kernel_size) 
        self.bias = torch.rand(kernel_size)
        self.kernel_size = kernel_size
        self.out_channels = channels_out
        self.in_channels = channels_in

    def forward (self, input): 
        self.input = input[0]
        #input shape 
        self.input_shape = self.input.size()
        #output of convolution as a matrix product
        #print(self.kernel_size[0])
        unfolded = torch.nn.functional.unfold(self.input, kernel_size=self.kernel_size)
        print((self.weight.view(self.out_channels,-1) @ unfolded).size())
        print(self.bias.view(1,-1,1).size())
        wxb = self.weight.view(self.out_channels,-1) @ unfolded + self.bias.view(1,-1,1)
        self.output = wxb.view(1,self.out_channels, self.input_shape[2] - self.kernel_size+1, self.input_shape[3] -self.kernel_size+1)
        return self.output

    def backward (self, *gradwrtoutput, learning_rate):
        #unfolded = torch.nn.functional.unfold(input,kernel_size=self.kernel_size)
        #weight_gradient = torch.dot()
        # bias gradient is the input_gradient. 
        pass

    def __call__(self,*input):
        self.forward(input)
        print(input)
        return self.output

    def param(self):
        return[]

######## Container ########

'''
class Sequential(input):
    def __init__(self, loss, input_size):
        """
        :param loss: class instance with methods compute_loss and compute_grad (in our case always MSE())
        :param input_size: size of input samples of the network
        """
        self.input_size = input_size
        self.loss = loss
        self.layers = [] # empty until the layers of the network are given
   
    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
'''
        