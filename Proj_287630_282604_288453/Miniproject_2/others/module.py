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
        :param device: 'cpu' or 'cuda'
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
        Forward pass of ReLU
        :param input: tensor of hidden size representing the input given to the layer
        :return: tensor of hidden_size shape containing the element-wise ReLU of the input tensor
        """
        self.input = input.to(self.device)
        self.output = self.input
        self.output[self.output<=0] = 0
        self.output[self.output>0] = 1
        return self.output

    def backward(self, gradwrtoutput):
        """
        Backward pass of ReLU
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

    def update_gradient_step(self):
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
        :param device: 'cpu' or 'cuda'
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
        Forward pass of Sigmoid
        :param input: tensor of hidden_size shape representing the input given to the layer
        :return: tensor of hidden_size shape containing the element-wise Sigmoid of the input tensor
        """
        self.input = input.to(self.device)
        self.output = 1 / (1 + (-self.input).exp())
        return self.output

    def backward(self, gradwrtoutput):
        """
        Backward pass of Sigmoid
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
        :param channels_in: integer, size of the layer's input channel dimension (dimension 0)
        :param channels_out: integer, size of the layer's output channel dimension (dimension 0)
        :param input_shape: list of int, dimensions size of the input
        :param scale_factor: scale factor (int) used for scaling in both 2 and 3 dimensions
        :param device: 'cpu' or 'cuda'
        """
        self.device = device
        self.out_channels = channels_in
        self.in_channels = channels_in
        self.scale_factor = scale_factor

        if(input_shape): 
            self.input_shape=input_shape
            self.gradwrtinput = FloatTensor(input_shape).zero_().to(self.device)
            self.input = FloatTensor(input_shape).zero_().to(self.device)
            self.gradwrtoutput = FloatTensor((input_shape[0],input_shape[1],input_shape[2]*self.scale_factor,input_shape[2]*self.scale_factor)).zero_().to(self.device)

    def forward(self, input):
        """
        Forward pass of NearestUpsampling
        :param input: tensor representing the input given to the layer
        :return: tensor containing the element-wise Sigmoid of the input tensor
        """
        self.input=input.to(self.device)
        self.input_shape=input.size()
        nearest_neighbors_v = self.input.repeat_interleave(self.scale_factor, dim=2)
        self.output = nearest_neighbors_v.repeat_interleave(self.scale_factor, dim=3)
        return self.output

    def backward(self, gradwrtoutput):
        """
        Backward pass of NearestUpsampling
        :param gradwrtoutput: tensor representing the gradient with respect to the output of the layer
        :return: tensor representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput
        unfolded_gradwrtoutput=unfold(self.gradwrtoutput, self.scale_factor, stride=self.scale_factor)#.to(self.device)
        #sum gradient of the loss wrt to output along kernels
        summed_gradient=unfolded_gradwrtoutput.reshape(self.gradwrtoutput.size(0),self.in_channels,self.scale_factor**2,-1).sum(2)
        #reshape the vector as the input vector of the upsampling
        self.gradwrtinput=summed_gradient.view(self.gradwrtoutput.size(0),self.input_shape[1],self.input_shape[2],self.input_shape[2])
        return self.gradwrtinput

    def update_gradient_step(self):
        """
        Nothing to do because no parameter. Method added for consistency with other layers
        """

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
        return []

class Conv2d(object):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, dilation=1, padding=0 ,input_shape=0, device='cpu'): 
        """
        :param channels_in: integer, size of the layer's input channel dimension (dimension 0)
        :param channels_out: integer, size of the layer's output channel dimension (dimension 0)
        :param kernel_size: integer, size of the kernel along both dimensions 
        :param input_shape: list of int, dimensions size of the input
        :param stride:
        :param dilation:
        :param padding:
        :param device: 'cpu' or 'cuda'
        """
        self.device = device
        self.hidden_size = (channels_out,channels_in,kernel_size,kernel_size)

        self.kernel_size = kernel_size
        self.out_channels = channels_out
        self.in_channels = channels_in
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        # Random initialisation of weights
        self.weight = rand(channels_out,channels_in,kernel_size,kernel_size).to(self.device)
        self.bias = rand(channels_out).to(self.device)

        # PyTorch like initialization of weights
        k=1/(self.in_channels*self.kernel_size**2)
        self.weight = self.weight.uniform_(-math.sqrt(k),math.sqrt(k))
        self.bias = self.bias.uniform_(-math.sqrt(k),math.sqrt(k))

        self.weight_grad = FloatTensor(channels_out,channels_in,kernel_size,kernel_size).zero_().to(self.device)
        self.bias_grad = FloatTensor(channels_out).zero_().to(self.device)
        if(not(input_shape)): self.gradwrtinput = FloatTensor(input_shape).zero_().to(self.device)


    def forward (self, input): 
        """
        Forward pass of Conv2d
        :param input: tensor representing the input given to the layer
        :return: tensor containing the convolution of the input tensor 
        with the kernels (nb of kernels defined by channels_out)
        """
        self.input = input.to(self.device)
        #input shape 
        self.input_shape = self.input.size()

        #output of convolution as a matrix product
        self.unfolded = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding).to(self.device)

        wxb = self.weight.view(self.out_channels,-1) @ self.unfolded + self.bias.view(1,-1,1)#.to(self.device)
        self.output = wxb.view(self.input_shape[0],self.out_channels, int((self.input_shape[2] - self.kernel_size + 2*self.padding)/self.stride +1), int((self.input_shape[2] - self.kernel_size +2*self.padding)/self.stride +1))#.to(self.device)
        return self.output

    def backward (self, gradwrtoutput):
        """
        Backward pass of the Conv2d
        :param gradwrtoutput: tensor representing the gradient with respect to the output of the layer
        :return: tensor representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput.to(self.device)

        gradwrtoutput_reshaped = self.gradwrtoutput.permute(1, 2, 3, 0).reshape(self.weight.size(0), -1)

        # weight gradient 
        dW = gradwrtoutput_reshaped @ self.unfolded.permute(1,2,0).reshape(1,self.unfolded.size(1),-1).transpose(1,2)
        self.weight_grad =dW.reshape(self.weight.size())
        
        # bias gradient is the input_gradient, needs to be summed + supporting batches
        self.bias_grad =  self.gradwrtoutput.sum(3).sum(2).sum(0).reshape(self.out_channels, -1)

        # derivative of the Loss with respect to input
        W_flat = self.weight.reshape(self.out_channels, -1)
        dX_col = W_flat.transpose(0,1) @ gradwrtoutput_reshaped

        dX_col = dX_col.reshape(dX_col.size(0),int(dX_col.size(1)/self.gradwrtoutput.size(0)),-1).permute(2,0,1)
        self.gradwrtinput = fold(dX_col, output_size=(self.input_shape[2],self.input_shape[3]), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        return self.gradwrtinput

    def update_gradient_step(self,lr):
        """
        Gradient step
        """
        testavant=self.weight
        self.weight = self.weight - lr * self.weight_grad
        self.bias[0] = self.bias_grad[0] - lr * self.bias_grad[0]
        self.bias[1] = self.bias_grad[1] - lr * self.bias_grad[1]
        self.bias[2] = self.bias_grad[2] - lr * self.bias_grad[2]

    def __call__(self,*input):
        self.forward(input[0])
        return self.output

    def zero_grad(self):
        """
        Re-initializes all gradients with respect to parameters to zero
        """
        #print("avant zero_grad", self.weight_grad)
        self.weight_grad.zero_()
        self.bias_grad.zero_()
        #print("après zero grad", self.weight_grad)

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
        :param channels_in: integer, size of the layer's input channel dimension (dimension 0)
        :param channels_out: integer, size of the layer's output channel dimension (dimension 0)
        :param kernel_size: integer, size of the kernel along both dimensions 
        :param input_shape: list of int, dimensions size of the input
        :param scale_factor: used as stride for Conv2d, as scale_factor for NearestUpsampling
        :param dilation:
        :param padding:
        :param device: 'cpu' or 'cuda'
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

        if(input_shape): 
            self.gradwrtinput = FloatTensor(input_shape).zero_().to(self.device)
            self.input = FloatTensor(input_shape).zero_().to(self.device)
            self.gradwrtoutput = FloatTensor((input_shape[0],input_shape[1],input_shape[2]*self.scale_factor,input_shape[2]*self.scale_factor)).zero_().to(self.device)
            self.nearestupsampling=NearestUpsampling(self.in_channels, self.out_channels, self.scale_factor, input_shape=self.input.size())
            self.kernel_size = kernel_size

    def forward(self, input=0):
        """
        Forward pass of Upsampling
        :param input: tensor representing the input given to the layer
        :return: tensor containing the convolution of the input tensor 
        with the kernels (nb of kernels defined by channels_out)
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
        Backward pass of the Upsampling
        :param gradwrtoutput: tensor representing the gradient with respect to the output of the layer
        :return: tensor representing the gradient with respect to the input of the layer
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
        Re-Initializes all gradients with respect to parameters to zero (only Conv2d layer is considered since NearestUpsampling has no parameters)
        """
        self.conv2d.zero_grad()
        self.conv2d.zero_grad()

    def param(self):
        """
        :return: Parameter tensor and a gradient tensor of same size, of the Conv2d modules (the only one that has parameters in Upsampling layer)
        """
        return self.weight
