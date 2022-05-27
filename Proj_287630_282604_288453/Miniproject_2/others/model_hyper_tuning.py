from torch import FloatTensor, cuda, device, save, load, cat, no_grad, save
import Proj_287630_282604_288453.Miniproject_2.__init__
import matplotlib.pyplot as plt
from Proj_287630_282604_288453.Miniproject_2.others.module import Module,ReLU,Sigmoid,Conv2d,Upsampling
import torch
from torch.utils.data import random_split
import random
from collections import OrderedDict

######## Verbose ########
verbose=1

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
        return ((predicted-target)**2).mean()

    @staticmethod
    def compute_backward_pass_gradient(predicted, target):
        """
        :param predicted: tensor of shape (...) consisting of the output of the network, the predictions 
        :param target: tensor of shape (...) consisting of the targets
        :return: gradient of the loss wrt to the predictions
        """
        return 2*(predicted-target)/predicted.reshape(-1).size(0)

######## Optimizer: Stochastic Gradient Descent ########
class SGD():
    def __init__(self, lr=0.001, mini_batch_size=1, criterion=MSE(), *layers):
        # paramètre nb_epoch ?
        # tolerance?
        self.learning_rate = lr
        self.batch_size = mini_batch_size
        self.criterion = criterion
        self.layers=layers
        random.seed(0)

    def gradient_step(self):
        # generate random int for the stochastic choice of image in the mini_batch_size
        idx = random.randint(0, self.batch_size-1)
        picker=self.layers[0].input[idx,:,:,:]
        for layer_in_net in self.layers:
            print(layer_in_net)
            layer_in_net.update_gradient_step(self.learning_rate)

    def param(self):
        """
        """
        return []


######## Container ########

class Sequential(Module):
    def __init__(self, *layers): #loss
        """
        :param loss: class instance with methods compute_loss and compute_grad (in our case always MSE())
        :param input_size: size of input samples of the network
        """

        self.layers = layers # empty until the layers of the network are given

    def __call__(self,*input):
        self.model_input = input[0]
        #print("call seq, net")
        self.forward(self.model_input)
        return self.model_output
   
    def forward(self, input):
        """
        """
        #print("forward seq, net")
        self.model_input = input
        for layer in self.layers:
            input = layer.forward(input)
        self.model_output=input
        return self.model_output
    
    def zero_grad(self):
        """
        """
        for layer in self.layers:
            layer.zero_grad()

    def backward(self, gradwrtoutput):
        """
        """
        for layer in reversed(list(self.layers)):
            gradwrtoutput = layer.backward(gradwrtoutput)
        self.gradwrtinput=gradwrtoutput
        return self.gradwrtinput

    def param(self):
        """
        """
        return []

######## Model #########

class Model():
    def __init__(self, mini_batch_size=10, lr=0.001):
        """
        Instantiate model + optimizer + loss function 
        """

        super().__init__()

        self.device = device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Sequential(
            Conv2d(3, 3, 3, stride=2, device=self.device), #padding 1
            ReLU(),
            Conv2d(3, 3, 3, stride=2, device=self.device), #padding 1 
            ReLU(),
            Upsampling(3, 3, 3, scale_factor=2, padding=2, device=self.device), #pas de padding
            ReLU(),
            Upsampling(3, 3, 3, scale_factor=2, padding=1,device=self.device), #pas de padding
            Sigmoid()
        )
        
        self.criterion = MSE()
        self.optimizer = SGD(lr, mini_batch_size, MSE(), self.net.layers[0])
        self.mini_batch_size = mini_batch_size
        self.lr=lr

        #move the model & criterion to the device (CPU or GPU)
        #device = device('cuda' if cuda.is_available() else 'cpu')
        #self.to(device)
        #self.criterion.to(device)

    def forward(self, x):
        """
        xxx
        """
        #print("forward model")
        self.x = x
        self.y = self.net(self.x)
        return self.y

    def __call__(self,*x):
        #print("call model")
        self.x = x[0]
        self.y=self.forward(self.x)
        return self.y
   
    def load_pretrained_model(self, SAVE_PATH ='Proj_287630_282604_288453/Miniproject_2/bestmodel.pth'):
        """
        Loads the parameters saved in bestmodel.pth into the model
        """
        if cuda.is_available():
            self.load_state(load(SAVE_PATH))
        else:
            self.load_state(load(SAVE_PATH, map_location=device('cpu')))

    def load_state(self, saved_model):
        '''
        Reads .pth and loads weights
        '''
        for i in range(4):
            self.net.layers[i*2].weight = saved_model.get('conv2d.'+str(2*i)+'.weight')
            self.net.layers[i*2].bias = saved_model.get('conv2d.'+str(2*i)+'.bias')

    def state_dict(self):
        '''
        return weights and bias of layers, in a ordered dict
        '''
        dict = OrderedDict()
        for i in range(4): 
            dict['conv2d.'+str(2*i)+'.weight']=self.net.layers[i*2].weight
            dict['conv2d.'+str(2*i)+'.bias']=self.net.layers[i*2].bias

        return dict
        
    def train(self, train_input, train_target, nb_epochs=10, verbose=0,  SAVE_PATH='Proj_287630_282604_288453/Miniproject_2/bestmodel.pth'):
        """
        :param train_input: tensor of size (N, C, H, W) containing a noisy version of the images.
        :param train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.
        """
        print('patience 3')
        #move the model, criterion & data to the device (CPU or GPU)
        device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device_)
        self.criterion.to(device_)
        
        #creating the training and validation dataset as well as trainloaders
        #Subsets
        training_dataset = torch.utils.data.TensorDataset(train_input, train_target)
        length = int(0.8*len(training_dataset))
        train_subset, val_subset = random_split(training_dataset, lengths=[length, len(training_dataset) - length])
        #Loaders
        train_loader = torch.utils.data.DataLoader(train_subset,
                                                batch_size=self.mini_batch_size,
                                                shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset,
                                                batch_size=self.mini_batch_size,
                                                shuffle=True)                                       
        #training_dataset = torch.utils.data.TensorDataset(train_input, train_target)
        #Batching the data
        #training_generator = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=10, shuffle=True)     
        train_loss = []
        val_loss = []

        for e in range(nb_epochs):
            acc_loss_train = 0
            acc_loss_val = 0


            for i, data in enumerate(val_loader, 0):
                #No need for gradient calculation:
                with no_grad():
                    val_input, val_target = data
                    val_input, val_target = val_input.to(device_), val_target.to(device_)

                    output = self.forward(val_input)
                    self.loss = self.criterion(output, val_target)
                    acc_loss_val = acc_loss_val + self.loss.item()

                
            
            for i, data in enumerate(train_loader, 0):

                train_input, train_target = data
                train_input, train_target = train_input.to(device), train_target.to(device)

                output = self.forward(train_input)
                self.loss = self.criterion(output, train_target)
                acc_loss_train = acc_loss_train + self.loss.item()
                self.net.zero_grad()
                # Backward-pass
                grad_wrt_output = self.criterion.compute_backward_pass_gradient(output, train_target.narrow(0, b, self.mini_batch_size))
                self.net.backward(grad_wrt_output)
                # Gradient step
                self.optimizer.gradient_step()
            
            #for plotting the loss
            train_loss.append(acc_loss_train/len(train_subset))
            val_loss.append(acc_loss_val/len(val_subset))

            if verbose: print(f"Epoch #{e}: lr = {self.lr}, mini_batch_size = {self.mini_batch_size} Training loss = {acc_loss_train/len(train_subset)} ----- Validation loss = {acc_loss_val/len(val_subset)} ")


        #Saving the model
        if SAVE_PATH is not None : save(self.state_dict(), SAVE_PATH)

        #If verbose mode is active, we return the loss for plotting
        if verbose: 
            plt.figure(figsize=(8,6))
            plt.plot(train_loss[1:], '-o')
            plt.plot(val_loss[1:], '-o')
            plt.title('Training and validation losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Training loss', 'Validation loss'])
            return train_loss, val_loss


    def predict(self, test_input, mini_batch_size=1):
        """
        :param test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        :return: tensor of the size (N1, C, H, W)
        """
        #device = device('cuda' if cuda.is_available() else 'cpu')
        #test_input = test_input.to(device)

        losses = []
        model_outputs = []
        for b in range(0, test_input.size(0), mini_batch_size):
            output = self(test_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output)
        model_outputs = cat(model_outputs, dim=0)
        return model_outputs