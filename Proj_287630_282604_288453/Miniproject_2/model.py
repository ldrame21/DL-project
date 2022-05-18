import torch

    

class ReLU(input):
    def forward(self, input):
        return torch.maximum(input, 0) 
    def backward(self, gradwrtoutput):
        return torch.maximum(input, 0)
        