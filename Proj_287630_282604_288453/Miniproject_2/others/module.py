from torch import FloatTensor, rand, zeros
from torch.nn.functional import fold, unfold
import math

######## Module class type ########

class Module(object):
    def forward(self, *arguments):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []

######## Activation layers ########

class ReLU(Module):
    def __init__(self, input_size=0, device='cpu'):
        """
        :param input_size: integer, input size of the activation layer (equivalent to output size in activation layers)
        """
        self.device=device
        if(input_size): 
            self.hidden_size = input_size
            self.input = FloatTensor(input_size).zero_().to(self.device)
            self.output = FloatTensor(input_size).zero_().to(self.device)
            self.gradwrtinput = FloatTensor(input_size).zero_().to(self.device)

    def __call__(self,input):
        self.forward(input)
        return self.output

    def forward(self, input):
        """
        Forward pass.
        :param input: tensor of hidden size
        :return: tensor of hidden_size shape containing the element-wise ReLU of the input tensor
        """
        self.input = input.to(self.device)
        self.output = self.input
        self.output[self.output<=0] = 0
        self.output[self.output>0] = 1
        return self.output

    def backward(self, gradwrtoutput):
        """
        Backward pass.
        :param gradwrtoutput: tensor of hidden_size shape representing the gradient with respect to the output of the layer
        :return: tensor of hidden_size shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput.to(self.device)
        deriv = self.input
        deriv[deriv<=0] = 0
        deriv[deriv>0] = 1
        self.gradwrtinput = self.gradwrtoutput*deriv
        return self.gradwrtinput
    
    def zero_grad(self):
        """
        Nothing to do because no parameter. Method added for consistency with other layers
        """
        pass

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
        return []
        
class Sigmoid(Module):
    def __init__(self, input_size=0, device='cpu'):
        """
        :param input_size: integer, input size of the activation layer or output size
        """
        super().__init__()
        self.device=device
        if(input_size): 
            self.hidden_size = input_size
            self.input = FloatTensor(input_size).zero_().to(self.device)
            self.output = FloatTensor(input_size).zero_().to(self.device)
            self.gradwrtinput = FloatTensor(input_size).zero_().to(self.device)
       
    def __call__(self,input):
        self.forward(input.to(self.device))
        return self.output.to(self.device)

    def forward(self, input):
        """
        Forward pass.
        :param input: tensor of hidden_size shape
        :return: tensor of hidden_size shape containing the element-wise Sigmoid of the input tensor
        """
        self.input = input.to(self.device)
        self.output = 1 / (1 + (-self.input).exp())
        return self.output

    def backward(self, gradwrtoutput):
        """
        Backward pass.
        :param gradwrtoutput: tensor of hidden_size shape representing the gradient with respect to the output of the layer
        :return: tensor of hidden_size shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput.to(self.device)
        self.gradwrtinput = (self.gradwrtoutput * ( 1 - self.output) * self.output)
        return self.gradwrtinput 

    def zero_grad(self):
        """
        Nothing to do because no parameter. Method added for consistency with other layers
        """
        pass

    def update_gradient_step(self):
        """
        Nothing to do because no parameter. Method added for consistency with other layers
        """

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
        return []


######## Convolutionnal layers ########

class NearestUpsampling(Module):
    def __init__(self, channels_in, channels_out, scale_factor=1, input_shape=0, device='cpu'):
        """
        """
        self.device = device
        self.out_channels = channels_in
        self.in_channels = channels_in
        #stride is equivalent to scale_factor
        self.scale_factor = scale_factor

        if(input_shape): 
            self.input_shape=input_shape
            self.gradwrtinput = FloatTensor(input_shape).zero_().to(self.device)
            self.input = FloatTensor(input_shape).zero_().to(self.device)
            self.gradwrtoutput = FloatTensor((input_shape[0],input_shape[1],input_shape[2]*self.scale_factor,input_shape[2]*self.scale_factor)).zero_().to(self.device)

    def forward(self, input):
        """
        """
        self.input=input.to(self.device)
        self.input_shape=input.size()
        #print(" NearestUpsampling input",self.input.size())
        # autre methode: ? neighbors=torch.nn.functional.unfold(self.input, kernel_size=1, stride=self.stride, dilation=self.dilation)
        nearest_neighbors_v = self.input.repeat_interleave(self.scale_factor, dim=2)
        self.output = nearest_neighbors_v.repeat_interleave(self.scale_factor, dim=3)
        return self.output

    def backward(self, gradwrtoutput):
        """
        """
        self.gradwrtoutput = gradwrtoutput
        #print(" gradwrtoutput UPsmapling ",self.gradwrtoutput.size())
        unfolded_gradwrtoutput=unfold(self.gradwrtoutput, self.scale_factor, stride=self.scale_factor)#.to(self.device)
        #sum gradient of the loss wrt to output along kernels
        #print(" unflolded ",unfolded_gradwrtoutput.size())
        summed_gradient=unfolded_gradwrtoutput.reshape(self.gradwrtoutput.size(0),self.in_channels,self.scale_factor**2,-1).sum(2)
        #reshape the vector as the input vector of the upsampling
        #print(" unflolded ",summed_gradient.size())
        self.gradwrtinput=summed_gradient.view(self.gradwrtoutput.size(0),self.input_shape[1],self.input_shape[2],self.input_shape[2])
        return self.gradwrtinput

    def update_gradient_step(self):
        """
        Nothing to do because no parameter. Method added for consistency with other layers
        """

    def param(self):
        """
        """
        return []

class Conv2d(object):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, dilation=1, padding=0 ,input_shape=0, device='cpu'): 
        """
        :param channels_in: integer, size of the layer's input channel dimension (dimension 0)
        :param channels_out: integer, size of the layer's output channel dimension (dimension 0)
        :param kernel_size: integer, size of the kernel along both dimensions 
        :param input_shape: list of int, dimensions size of the input
        """
        self.device =device
        self.hidden_size = (channels_out,channels_in,kernel_size,kernel_size)

        self.kernel_size = kernel_size
        self.out_channels = channels_out
        self.in_channels = channels_in
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        #random initialisation of weights
        self.weight = rand(channels_out,channels_in,kernel_size,kernel_size).to(self.device)
        self.bias = rand(channels_out).to(self.device)

        # PyTorch like initialization of weights
        k=self.in_channels/(self.in_channels*self.kernel_size**2)
        self.weight = self.weight.uniform_(-math.sqrt(k),-math.sqrt(k))
        self.bias = self.bias.uniform_(-math.sqrt(k),-math.sqrt(k))

        self.weight_grad = FloatTensor(channels_out,channels_in,kernel_size,kernel_size).zero_().to(self.device)
        self.bias_grad = FloatTensor(channels_out).zero_().to(self.device)
        if(not(input_shape)): self.gradwrtinput = FloatTensor(input_shape).zero_().to(self.device)


    def forward (self, input): 
        """
        Forward pass.
        :param input: tensor of shape (1, channels_in, H, W)
        :return: tensor of shape (channels_out, H-kernel_size+1, W-kernel_size+1) containing the convolution of the input tensor 
        with the kernels (nb of kernels defined by channels_out)
        """
        self.input = input.to(self.device)
        #input shape 
        self.input_shape = self.input.size()

        #output of convolution as a matrix product
        self.unfolded = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding).to(self.device)

        wxb = self.weight.view(self.out_channels,-1) @ self.unfolded + self.bias.view(1,-1,1)#.to(self.device)
        self.output = wxb.view(self.input_shape[0],self.out_channels, int((self.input_shape[2] - self.kernel_size)/self.stride +1 +2*self.padding), int((self.input_shape[2] - self.kernel_size)/self.stride +1+2*self.padding))#.to(self.device)
        return self.output

    def backward (self, gradwrtoutput):
        """
        Backward pass.
        :param gradwrtoutput: tensor of (...) shape representing the gradient with respect to the output of the layer
        :return: tensor of (...) shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput.to(self.device)

        gradwrtoutput_flat = self.gradwrtoutput.permute(1, 2, 3, 0).sum(3).reshape(self.out_channels, -1)

        # weight gradient 
        dW = gradwrtoutput_flat @ self.unfolded.transpose(1,2).sum(0,keepdim=True)
        self.weight_grad =dW.reshape(self.weight.size())
        #self.weight_grad = self.weight_grad.add(dW.reshape(self.weight.size()))

        # bias gradient is the input_gradient. 
        self.bias_grad =  self.gradwrtoutput.sum(3).sum(2).sum(0).reshape(self.out_channels, -1)
        #self.bias_grad = self.bias_grad.add(self.gradwrtoutput.sum(3).sum(2).sum(0).reshape(self.out_channels, -1))
  
        # derivative of the Loss with respect to input
        W_flat = self.weight.reshape(self.out_channels, -1)
        dX_col = W_flat.transpose(0,1) @ gradwrtoutput_flat
        #dX_col = dX_col.reshape(dX_col.size(0),int(dX_col.size(1)/gradwrtoutput.size(0)),-1).permute(2,0,1)
        dX_col = dX_col.reshape(dX_col.size(0),dX_col.size(1),-1).permute(2,0,1)
        self.gradwrtinput = fold(dX_col, output_size=(self.input_shape[2],self.input_shape[3]), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        return self.gradwrtinput

    def update_gradient_step(self,lr):
        """
        Gradient step
        """
        self.weight = self.weight - lr * self.weight_grad
        self.bias[0] = self.bias_grad[0] - lr * self.bias_grad[0]
        self.bias[1] = self.bias_grad[1] - lr * self.bias_grad[1]
        self.bias[2] = self.bias_grad[2] - lr * self.bias_grad[2]

    def __call__(self,*input):
        self.forward(input[0])
        return self.output

    def zero_grad(self):
        """
        Reset all parameter gradients to 0
        """
        self.weight_grad.zero_()
        self.bias_grad.zero_()

    def param(self):
        """
        :return: A list of pairs, each composed of a parameter tensor, and a gradient tensor of same size.
        """
        #pas sûre
        return [(self.weight[i, :, :, :], self.weight_grad[i, :, :, :]) for i in range(self.hidden_size[0])] \
               + [(self.bias, self.bias_grad)]

class Upsampling(Module):
    def __init__(self, channels_in, channels_out, kernel_size, dilation=1, scale_factor=1, padding=0, input_shape=0, device='cpu'):
        """
        """
        self.device = device
        self.out_channels = channels_out
        self.in_channels = channels_in
        #stride is equivalent to scale_factor
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.padding = padding
        self.kernel_size = kernel_size

        #random initialisation of weights
        self.conv2d = Conv2d(self.in_channels, self.out_channels, self.kernel_size, dilation=self.dilation, padding=self.padding)
        self.nearestupsampling=NearestUpsampling(self.in_channels, self.out_channels, self.scale_factor)

        self.weight = self.conv2d.weight
        self.bias = self.conv2d.bias

        #to change
        self.hidden_size = (channels_out,channels_in,kernel_size,kernel_size)
        
        if(input_shape): 
            self.gradwrtinput = FloatTensor(input_shape).zero_().to(self.device)
            self.input = FloatTensor(input_shape).zero_().to(self.device)
            self.gradwrtoutput = FloatTensor((input_shape[0],input_shape[1],input_shape[2]*self.scale_factor,input_shape[2]*self.scale_factor)).zero_().to(self.device)
            self.nearestupsampling=NearestUpsampling(self.in_channels, self.out_channels, self.scale_factor, input_shape=self.input.size())
            self.kernel_size = kernel_size

    def forward(self, input=0):
        """
        """
        self.input = input
        #NNUpsampling
        self.intermediate_output=self.nearestupsampling.forward(self.input)
        #Conv2d
        self.output = self.conv2d.forward(self.intermediate_output)
        return self.output

    def __call__(self,input):
        self.output = self.forward(input)
        return self.output

    def backward(self, gradwrtoutput):
        """
        """
        self.gradwrtoutput=gradwrtoutput
        self.intermediate_gradwrtinput = self.conv2d.backward(self.gradwrtoutput)
        self.gradwrtinput = self.nearestupsampling.backward(self.intermediate_gradwrtinput)
        return self.gradwrtinput
    
    def update_gradient_step(self,lr):
        """
        Gradient step - Only conv2d layer has parameters 
        """
        self.conv2d.update_gradient_step(lr)

    def zero_grad(self):
        """
        Reset all parameter gradients to 0 (only Conv2d layer is considered since NearestUpsampling has no parameters)
        """
        self.conv2d.zero_grad()
        self.conv2d.zero_grad()

    def param(self):
        return self.weight
