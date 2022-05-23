from torch import FloatTensor, random
import torch
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
        :param input_size: integer, input size of the activation layer (equivalent to output size in activation layers)
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
        self.gradwrtouput = gradwrtoutput
        deriv = self.input
        deriv[deriv<=0] = 0
        deriv[deriv>0] = 1
        self.gradwrtinput = self.gradwrtoutput*deriv
        return self.gradwrtinput

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
        return []
        
class Sigmoid(Module):
    def __init__(self, input_size=0):
        """
        :param input_size: integer, input size of the activation layer or output size
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
        :return: tensor of hidden_size shape containing the element-wise Sigmoid of the input tensor
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
        self.gradwrtouput = gradwrtoutput
        self.gradwrtinput = gradwrtoutput * ( 1 - self.output) * self.output
        return self.gradwrtinput 

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
        return []


######## Optimizer: Stochastiuc Gradient Descent ########
class SGD():
    def __init__(self, learning_rate=0.1, batch_size=1, random_state=None):
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
    def __init__(self, channels_in, channels_out, kernel_size, input_shape=0): 
        """
        :param channels_in: integer, size of the layer's input channel dimension (dimension 0)
        :param channels_out: integer, size of the layer's output channel dimension (dimension 0)
        :param kernel_size: integer, size of the kernel along both dimensions 
        :param input_shape: list of int, dimensions size of the input
        """
        self.hidden_size = (channels_in,channels_out,kernel_size,kernel_size)
        #random initialisation of weights
        self.weight = torch.rand(channels_in,channels_out,kernel_size,kernel_size) 
        self.bias = torch.rand(kernel_size)
        self.kernel_size = kernel_size
        self.out_channels = channels_out
        self.in_channels = channels_in

        self.weight_grad = FloatTensor(channels_in,channels_out,kernel_size,kernel_size).zero_()
        self.bias_grad = FloatTensor(kernel_size).zero_()
        if(not(input_shape)): self.gradwrtinput = FloatTensor(input_shape)

    def forward (self, input): 
        """
        Forward pass.
        :param input: tensor of shape (channels_in, H, W)
        :return: tensor of shape (channels_out, H-kernel_size+1, W-kernel_size+1) containing the convolution of the input tensor 
        with the kernels (nb of kernels defined by channels_out)
        """
        self.input = input[0]

        #input shape 
        self.input_shape = self.input.size()

        #output of convolution as a matrix product
        unfolded = torch.nn.functional.unfold(self.input, kernel_size=self.kernel_size)
        print((self.weight.view(self.out_channels,-1) @ unfolded).size())
        print(self.bias.view(1,-1,1).size())
        wxb = self.weight.view(self.out_channels,-1) @ unfolded + self.bias.view(1,-1,1)
        self.output = wxb.view(1,self.out_channels, self.input_shape[2] - self.kernel_size+1, self.input_shape[3] -self.kernel_size+1)
        return self.output

    def backward (self, *gradwrtoutput, learning_rate):
        """
        Backward pass.
        :param gradwrtoutput: tensor of (...) shape representing the gradient with respect to the ouput of the layer
        :return: tensor of (...) shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtouput = gradwrtoutput

        ### dL/dF (F being the kernels, filters, L being the loss)

        #convolution 
        unfolded_input = torch.nn.functional.unfold(self.input, kernel_size=self.kernel_size)
        self.weight_grad = self.gradwrtoutput.view(self.in_channels,-1) @ unfolded
        #self.weight_grad 
        unfolded = torch.nn.functional.unfold(self.input, kernel_size=self.kernel_size)
        dF = self.gradwrtoutput.view(self.in_channels,-1) @ unfolded
        #pas sûre du view
        self.weight_grad = dF.view(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        
        # bias gradient is the input_gradient. 
        self.bias_grad = self.gradwrtouput.sum(self.out_channels)

        ### dL/dX (X being the input, L being the loss)
        unfolded_kernel = torch.nn.functional.unfold(self.weight.flip([2]).flip([1]),kernel_size=self.kernel_size)
        dX = self.gradwrtoutput.view(self.out_channels,-1) @ unfolded_kernel
        #pas sûre du view
        self.gradwrtinput = dF.view(1,self.in_channels, self.input_shape[2] + self.kernel_size-1, self.input_shape[3] +self.kernel_size-1)
        return self.gradwrtinput

    def __call__(self,*input):
        self.forward(input)
        return self.output

    def param(self):
        """
        :return: A list of pairs, each composed of a parameter tensor, and a gradient tensor of same size.
        """
        #pas sûre
        return [(self.weight[:, i, :, :], self.weight_grad[:, i, :, :]) for i in range(self.hidden_size)] \
               + [(self.bias, self.bias_grad)]


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

class Model():
    def  __init __(self):
    ## instantiate model + optimizer + loss function + any other stuff you need
    pass
    def load_pretrained_model(self):
    ## This loads the parameters saved in bestmodel.pth into the model pass
    
    def train(self, train_input, train_target):
    #:train ̇input: tensor of size (N, C, H, W) containing a noisy version of the images same images, which only differs from the input by their noise.
    #:train ̇target: tensor of size (N, C, H, W) containing another noisy version of the
    pass

    def predict(self, test_input):
    #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
    #: returns a tensor of the size (N1, C, H, W) pass
'''
        