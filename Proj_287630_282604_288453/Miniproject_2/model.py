from torch import FloatTensor, cuda, device, save, load, cat
import Proj_287630_282604_288453.Miniproject_2.__init__
import matplotlib.pyplot as plt
from Proj_287630_282604_288453.Miniproject_2.others.module import Module,ReLU,Sigmoid,Conv2d,Upsampling

import random
from collections import OrderedDict

######## Verbose ########
verbose=0 #verbose variable useful for monitoring accumulated loss and 

######## Loss ########
class MSE():
    def __init__(self, device='cpu'):
        self.device=device 

    def compute_loss(self, predicted, target):
        """
        Compute the Mean-Squarred-Error loss btw predicted tensor and target tensor
        :param predicted: tensor of shape (...) consisting of the output of the network, the predictions 
        :param target: tensor of shape (...) consisting of the targets
        :return: loss
        """
        return ((predicted-target)**2).mean().to(self.device)

    def compute_backward_pass_gradient(self, predicted, target):
        """
        Compute the gradient of the loss wrt to the predicted tensor
        :param predicted: tensor of shape (...) consisting of the output of the network, the predictions 
        :param target: tensor of shape (...) consisting of the targets
        :return: gradient of the loss wrt to the predictions
        """
        return 2*(predicted-target)/predicted.reshape(-1).size(0)

######## Optimizer: Stochastic Gradient Descent ########
class SGD():
    def __init__(self, *layers, lr=0.001, mini_batch_size=1, criterion=MSE()):#net
        self.learning_rate = lr
        self.batch_size = mini_batch_size
        self.criterion = criterion
        self.layers=layers
        random.seed(0)

    def gradient_step(self):
        """
        Perfoms gradient step for each layer of the network
        """
        for layer_in_net in self.layers:
            layer_in_net.update_gradient_step(self.learning_rate)

######## Container ########
class Sequential(Module):
    def __init__(self, *layers): 
        self.layers = layers # empty until the layers of the network are given

    def __call__(self,*input):
        self.model_input = input[0]
        self.forward(self.model_input)
        return self.model_output
   
    def forward(self, input):
        """
        Performs the forward pass, layer by layer 
        :param input: input given to the first layer for the forward pass
        """
        self.model_input = input
        for layer in self.layers:
            input = layer.forward(input)
        self.model_output=input
        return self.model_output
    
    def zero_grad(self):
        """
        Re-Initializes all gradients with respect to parameters to zero
        """
        for layer in self.layers:
            layer.zero_grad()

    def backward(self, gradwrtoutput):
        """
        Performs the forward pass, layer by layer 
        :param gradwrtoutput: tensor, gradient of the loss wrt to predictions given as entry of the backward pass
        """
        self.gradwrtoutput=gradwrtoutput
        for layer in reversed(list(self.layers)):
            gradwrtoutput = layer.backward(gradwrtoutput)
        
        self.gradwrtinput=gradwrtoutput

######## Model #########

class Model():
    def __init__(self, mini_batch_size=10, lr=0.001):
        """
        Instantiate model + optimizer + loss function 
        """
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        self.net = Sequential(
            Conv2d(3, 3, 3, stride=2, padding=1, device=self.device), 
            ReLU(),
            Conv2d(3, 3, 3, stride=2, padding=1, device=self.device),
            ReLU(),
            Upsampling(3, 3, 3, scale_factor=2, padding=1, device=self.device),
            ReLU(),
            Upsampling(3, 3, 3, scale_factor=2, padding=1, device=self.device),
            Sigmoid()
        )
        self.criterion = MSE()
        self.optimizer = SGD(self.net.layers[0], lr=lr, mini_batch_size=mini_batch_size, criterion=MSE())
        self.mini_batch_size = mini_batch_size
        self.lr=lr

    def __call__(self,*x):
        self.x = x[0]
        self.y=self.forward(self.x)
        return self.y
   
    def forward(self, x):
        """
        Performs the forward pass, layer by layer 
        :param x: input given to the model for the forward pass
        """
        self.x = x
        self.y = self.net(self.x)
        return self.y

    def load_pretrained_model(self, SAVE_PATH ='Proj_287630_282604_288453/Miniproject_2/bestmodel.pth'):
        """
        Loads the parameters saved in bestmodel.pth into the model
        :param SAVE_PATH: string, containing the path to the bestmodel.pth
        """
        if cuda.is_available():
            self.load_state(load(SAVE_PATH))
        else:
            self.load_state(load(SAVE_PATH, map_location=device('cpu')))

    def load_state(self, saved_model):
        '''
        Reads model parameters saved in .pth format and loads parameters accordingly in the model
        :param saved_model: string containing the path of .pth file to load parameters from
        '''
        for i in range(4):
            if(i==0 or i==1):
                self.net.layers[i*2].weight = saved_model.get('conv2d.'+str(2*i)+'.weight')
                self.net.layers[i*2].bias = saved_model.get('conv2d.'+str(2*i)+'.bias')
            else:
                self.net.layers[i*2].conv2d.weight = saved_model.get('conv2d.'+str(2*i)+'.weight')
                self.net.layers[i*2].conv2d.bias = saved_model.get('conv2d.'+str(2*i)+'.bias')

    def state_dict(self):
        '''
        Return weights and bias of layers, in a ordered dict
        '''
        dict = OrderedDict()
        for i in range(4): 
            if(i==0 or i==1):
                dict['conv2d.'+str(2*i)+'.weight']=self.net.layers[i*2].weight
                dict['conv2d.'+str(2*i)+'.bias']=self.net.layers[i*2].bias
            else:
                dict['conv2d.'+str(2*i)+'.weight']=self.net.layers[i*2].conv2d.weight
                dict['conv2d.'+str(2*i)+'.bias']=self.net.layers[i*2].conv2d.bias
        return dict
        
    def train(self, train_input, train_target, nb_epochs=10, verbose=0,  SAVE_PATH='Proj_287630_282604_288453/Miniproject_2/bestmodel.pth'):
        """
        :param train_input: tensor of size (N, C, H, W) containing a noisy version of the images.
        :param train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.
        """

        train_loss = []

        for e in range(nb_epochs):
            acc_loss = 0

            # Moving training data to device
            train_input = train_input.to(self.device)
            train_target = train_target.to(self.device)

            # SGD - shuffle datasamples for stochastic gradient descent
            random.shuffle(train_input)

            for b in range(0, train_input.size(0), self.mini_batch_size):
                
                output = self.forward(train_input.narrow(0, b, self.mini_batch_size)).to(self.device)
                self.loss = self.criterion.compute_loss(output, train_target.narrow(0, b, self.mini_batch_size))
                acc_loss = acc_loss + self.loss

                self.net.zero_grad()

                # Backward-pass
                grad_wrt_output = self.criterion.compute_backward_pass_gradient(output, train_target.narrow(0, b, self.mini_batch_size))
                self.net.backward(grad_wrt_output)
               
                # Gradient step
                self.optimizer.gradient_step()
            
            # Useful for plotting the loss
            train_loss.append(acc_loss)

            if verbose: print(e, acc_loss)

        #Saving the model
        if SAVE_PATH is not None : save(self.state_dict(), SAVE_PATH)

        #If verbose mode is active, we return the loss for plotting
        if verbose: 
            plt.figure(figsize=(8,6))
            plt.plot(train_loss, '-o')
            plt.title('Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            return train_loss


    def predict(self, test_input, mini_batch_size=1):
        """
        :param test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        :return: tensor of the size (N1, C, H, W)
        """
        losses = []
        model_outputs = []
        for b in range(0, test_input.size(0), mini_batch_size):
            output = self(test_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output)
        model_outputs = cat(model_outputs, dim=0)
        return model_outputs