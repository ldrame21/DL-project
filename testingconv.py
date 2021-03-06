
import torch
import torch.nn as nn
from Proj_287630_282604_288453.Miniproject_2.others.module import *
from Proj_287630_282604_288453.Miniproject_2 import model

x = torch.randn(5, 3, 32, 32)

'''Test forward'''
print('torch implementation')
torchconv= nn.Conv2d(3, 3, 5, stride=1, padding=2)
#print(torchconv.weight.size())
#print(torchconv.bias.size())
output = torchconv(x)
print(output.size())

print('our implementation')
ourconv= Conv2d(3, 3, 5, stride=1, padding=2)
ourconv.weight=torchconv.weight
ourconv.bias=torchconv.bias
#print(ourconv.weight.size())
#print(ourconv.bias.size())
output_our = ourconv(x)
print(output_our.size())


print(torch.allclose(output_our, output))

model = model.Model()
print(model.state_dict().keys())

'''Test backward
print('torch implementation')
torchconv= nn.Conv2d(3, 3, 3, stride=1)
out = torchconv(x)
loss = out.sum()
loss.backward()
print(torchconv.weight.grad)
print(torchconv.bias.grad)
#print(x.grad.size())

print('our implementation')
ourconv= Conv2d(3, 3, 3, stride=1)
ourconv.weight=torchconv.weight
ourconv.bias=torchconv.bias
out_our = ourconv(x)
loss = out.sum()
loss.backward()
print(ourconv.weight_grad.size())
print(ourconv.bias_grad.size())
print(ourconv.gradwrtinput.size())
print(ourconv.gradwrtoutput.size())

'''
