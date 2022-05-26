from torch import FloatTensor, rand, random, zeros, bool
from torch.nn.functional import fold, unfold

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
    def __init__(self, input_size=0):
        """
        :param input_size: integer, input size of the activation layer (equivalent to output size in activation layers)
        """
        if(input_size): 
            self.hidden_size = input_size
            self.input = FloatTensor(input_size).zero_()
            self.output = FloatTensor(input_size).zero_()
            self.gradwrtinput = FloatTensor(input_size).zero_()

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
        :param gradwrtoutput: tensor of hidden_size shape representing the gradient with respect to the output of the layer
        :return: tensor of hidden_size shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput
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
        super().__init__()
        if(input_size): 
            self.hidden_size = input_size
            self.input = FloatTensor(input_size).zero_()
            self.output = FloatTensor(input_size).zero_()
            self.gradwrtinput = FloatTensor(input_size).zero_()
       
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
        :param gradwrtoutput: tensor of hidden_size shape representing the gradient with respect to the output of the layer
        :return: tensor of hidden_size shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput
        self.gradwrtinput = gradwrtoutput * ( 1 - self.output) * self.output
        return self.gradwrtinput 

    def param(self):
        """
        :return: empty list since the activation layers have no parameters
        """
        return []


######## Convolutionnal layers ########

class NearestUpsampling(Module):
    def __init__(self, channels_in, channels_out, dilation=1, scale_factor=1, padding=0, input_shape=0):
        """
        """
        self.out_channels = channels_in
        self.in_channels = channels_in
        #stride is equivalent to scale_factor
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.padding = padding

        if(input_shape): 
            self.gradwrtinput = FloatTensor(input_shape).zero_()
            self.input = FloatTensor(input_shape).zero_()
            self.gradwrtoutput = FloatTensor((input_shape[0],input_shape[1],input_shape[2]*self.scale_factor,input_shape[2]*self.scale_factor)).zero_()

    def forward(self, input):
        """
        """
        self.input=input
        print(" NearestUpsampling input",self.input.size())
        # autre methode: ? neighbors=torch.nn.functional.unfold(self.input, kernel_size=1, stride=self.stride, dilation=self.dilation)
        nearest_neighbors_v = self.input.repeat_interleave(self.scale_factor, dim=2)
        self.output = nearest_neighbors_v.repeat_interleave(self.scale_factor, dim=3)
        return self.output

    def backward(self, gradwrtoutput):
        """
        """
        self.gradwrtoutput = gradwrtoutput
        kernel_mask=zeros(self.kernel.size(), dtype=bool)
        kernel_mask[int(self.kernel.size(0)-self.kernel.size(0)/2),int(self.kernel.size(1)-self.kernel.size(1)/2)]=True
        print(kernel_mask)
        mask=kernel_mask.repeat(1,1,int(self.gradwrtoutput.size(2)/self.kernel.size(0)),int(self.gradwrtoutput.size(3)/self.kernel.size(1))).repeat(1,self.out_channels,1,1) 
        print(mask)
        print(mask)
        output= self.gradwrtoutput.masked_select(mask)
        self.gradwrtinput=output.view(int(self.gradwrtoutput.size(0)/self.kernel.size(0)),int(self.gradwrtoutput.size(1)/self.kernel.size(1)))
        return self.gradwrtinput

    def param(self):
        """
        """
        return []

class Conv2d(object):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, dilation=1, padding=0 ,input_shape=0): 
        """
        :param channels_in: integer, size of the layer's input channel dimension (dimension 0)
        :param channels_out: integer, size of the layer's output channel dimension (dimension 0)
        :param kernel_size: integer, size of the kernel along both dimensions 
        :param input_shape: list of int, dimensions size of the input
        """
        self.hidden_size = (channels_out,channels_in,kernel_size,kernel_size)
        #random initialisation of weights
        self.weight = rand(channels_out,channels_in,kernel_size,kernel_size) 
        self.bias = rand(channels_out)
        self.kernel_size = kernel_size
        self.out_channels = channels_out
        self.in_channels = channels_in
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        self.weight_grad = FloatTensor(channels_out,channels_in,kernel_size,kernel_size).zero_()
        self.bias_grad = FloatTensor(channels_out).zero_()
        if(not(input_shape)): self.gradwrtinput = FloatTensor(input_shape).zero_()

    def forward (self, input): 
        """
        Forward pass.
        :param input: tensor of shape (1, channels_in, H, W)
        :return: tensor of shape (channels_out, H-kernel_size+1, W-kernel_size+1) containing the convolution of the input tensor 
        with the kernels (nb of kernels defined by channels_out)
        """
        self.input = input
        #input shape 
        self.input_shape = self.input.size()
        print("forward conv2d input ",self.input_shape)
        #output of convolution as a matrix product
        unfolded = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation, padding=self.padding)
        #print(unfolded.size())
        #print(self.weight.size())
        #print((self.weight.view(self.out_channels,-1) @ unfolded).size())
        #print(self.bias.view(1,-1,1).size())
        wxb = self.weight.view(self.out_channels,-1) @ unfolded + self.bias.view(1,-1,1)
        self.output = wxb.view(1,self.out_channels, int((self.input_shape[2] - self.kernel_size)/self.stride +1 +2*self.padding), int((self.input_shape[2] - self.kernel_size)/self.stride +1+2*self.padding))
        print("forward conv2d output ",self.output.size())
        return self.output

    def backward (self, *gradwrtoutput):
        """
        Backward pass.
        :param gradwrtoutput: tensor of (...) shape representing the gradient with respect to the output of the layer
        :return: tensor of (...) shape representing the gradient with respect to the input of the layer
        """
        self.gradwrtoutput = gradwrtoutput

        ### dL/dF (F being the kernels, weights, L being the loss)

        #convolution 
        unfolded_input = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        self.weight_grad = self.gradwrtoutput.view(self.in_channels,-1) @ unfolded_input
        #self.weight_grad 
        unfolded = unfold(self.input, kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation)
        dF = self.gradwrtoutput.view(self.in_channels,-1) @ unfolded
        #pas sûre du view
        self.weight_grad = dF.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # bias gradient is the input_gradient. 
        self.bias_grad = self.gradwrtoutput.sum(self.out_channels)

        ### dL/dX (X being the input, L being the loss)
        unfolded_kernel = unfold(self.weight.flip([2]).flip([1]),kernel_size=self.kernel_size)
        dX = self.gradwrtoutput.view(self.out_channels,-1) @ unfolded_kernel
        #pas sûre du view
        self.gradwrtinput = dF.view(1,self.in_channels, int((self.input_shape[2] - self.kernel_size)/self.stride +1), int((self.input_shape[2] - self.kernel_size)/self.stride +1))
        return self.gradwrtinput

    def __call__(self,*input):
        print("call conv2d")
        self.forward(input[0])
        return self.output

    def param(self):
        """
        :return: A list of pairs, each composed of a parameter tensor, and a gradient tensor of same size.
        """
        #pas sûre
        return [(self.weight[i, :, :, :], self.weight_grad[i, :, :, :]) for i in range(self.hidden_size[0])] \
               + [(self.bias, self.bias_grad)]


class Upsampling(Module):
    def __init__(self, channels_in, channels_out, kernel_size, dilation=1, scale_factor=1, padding=0, input_shape=0):
        """
        """
        self.out_channels = channels_out
        self.in_channels = channels_in
        #stride is equivalent to scale_factor
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.padding = padding
        self.kernel_size = kernel_size

        #to change
        self.hidden_size = (channels_out,channels_in,kernel_size,kernel_size)
        
        if(input_shape): 
            self.gradwrtinput = FloatTensor(input_shape).zero_()
            self.input = FloatTensor(input_shape).zero_()
            self.gradwrtoutput = FloatTensor((input_shape[0],input_shape[1],input_shape[2]*self.scale_factor,input_shape[2]*self.scale_factor)).zero_()
            self.kernel_size = kernel_size

    def forward(self, input=0):
        """
        """
        self.input = input
        print("forward upsampling, input ",self.input.size())
        #NNUpsampling
        self.intermediate_output= NearestUpsampling(self.in_channels, self.out_channels, self.dilation, self.scale_factor, input_shape=self.input.size()).forward(self.input)
        #Conv2d
        self.output = Conv2d(self.in_channels, self.out_channels, self.kernel_size, dilation=self.dilation, padding=self.padding, input_shape=self.intermediate_output.size()).forward(self.intermediate_output)
        return self.output
        #channels_in, channels_out, kernel_size, stride=1, dilation=1, input_shape=0

    def __call__(self,input):
        self.output = self.forward(input)
        return self.output

    def backward(self, gradwrtoutput):
        """
        """
        self.gradwrtoutput=gradwrtoutput
        self.intermediate_gradwrtinput = Conv2d.backward(gradwrtoutput)
        self.gradwrtinput = NearestUpsampling.backward(self.intermediate_gradwrtinput)
        return self.gradwrtinput

    def param(self):
        """
        """
        return []

