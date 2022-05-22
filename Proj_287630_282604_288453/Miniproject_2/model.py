import torch
import __init__


class MSE(input):
    def forward(self, input, target):
        self.input = input #general output y of the network
        self.target = target
        #(input-target).pow(2).sum()/input.size()[0]
        return ((input-target)**2).mean()
    def backward(self, gradwrtoutput):
        return 2*(self.input-self.target).sum()
    def param(self):
        return []

class ReLU(input):
    def forward(self, input):
        self.input = input
        return torch.maximum(input, 0) 
    def backward(self, gradwrtoutput):
        deriv = self.input
        deriv[deriv<=0] = 0
        deriv[deriv>0] = 1
        return gradwrtoutput*deriv
    def param(self):
        return []
        
class Sigmoid(input):
    def forward(self, input):
        self.output = 1 / (1 + torch.exp(-input))
        return self.output
    def backward(self, gradwrtoutput):
        return gradwrtoutput * ( 1 - self.output) * self.output
    def param(self):
        return []

class NearestUpsampling(input):
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []

class SGD(input):
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []

class Sequential(input):
    def forward(self, input):
        raise NotImplementedError
    def backward(self, gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []

        