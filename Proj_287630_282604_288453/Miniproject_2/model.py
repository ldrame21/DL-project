from torch import FloatTensor, random
import torch
import Proj_287630_282604_288453.Miniproject_2.__init__
import matplotlib.pyplot as plt
from Proj_287630_282604_288453.Miniproject_2.others.module import Module,ReLU,Sigmoid,Conv2d

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



######## Optimizer: Stochastic Gradient Descent ########
class SGD():
    def __init__(self, *parameters, lr=0.001):#batch_size=1, random_state=None):
        # paramètre nb_epoch ?
        # tolerance?
        self.parameters = parameters
        self.learning_rate = lr
        #self.batch_size = batch_size ?
        random.seed()
        
    def param(self):
        """
        :return:
        """
        return []



######## Container ########

class Sequential(Module):
    def __init__(self, *layers): #loss, input_size):
        """
        :param loss: class instance with methods compute_loss and compute_grad (in our case always MSE())
        :param input_size: size of input samples of the network
        """
        #self.input_size = input_size
        #self.loss = loss
        self.layers = layers # empty until the layers of the network are given

    def __call__(self,*input):
        self.model_input = input[0]
        print("call seq, net")
        self.forward(self.model_input)
        return self.model_output
   
    def forward(self, input):
        """
        """
        print("forward seq, net")
        self.model_input = input
        for layer in self.layers:
            input = layer.forward(input)
        self.model_output=input
        return self.model_output

    def backward(self, gradwrtoutput):
        """
        """
        raise NotImplementedError

    def param(self):
        """
        """
        return []

######## Model #########

class Model(Module):
    def __init__(self, mini_batch_size=1):
        """
        Instantiate model + optimizer + loss function 
        """
        super().__init__()
        self.net = Sequential(
            Conv2d(3, 3, 3, stride=2),
            ReLU(),
            Conv2d(3, 3, 3, stride=2), 
            ReLU(),
            #NearestUpsampling(),
            #Conv2d(3, 3, 3, stride=2),
            #ReLU(),
            #NearestUpsampling(),
            #Conv2d(3, 3, 3, stride=2),
            Sigmoid()
        )
        self.optimizer = SGD(self.net.param(), lr=0.001)
        self.criterion = MSE()
        self.mini_batch_size = mini_batch_size

        #move the model & criterion to the device (CPU or GPU)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.to(device)
        #self.criterion.to(device)

    def forward(self, x):
        """
        xxx
        """
        print("forward model")
        self.x = x
        self.y = self.net(self.x)
        return self.y

    def __call__(self,*x):
        print("call model")
        self.x = x[0]
        self.y=self.forward(self.x)
        return self.y
   
    def load_pretrained_model(self, SAVE_PATH ='Proj_287630_282604_288453/Miniproject_2/bestmodel.pth'):
        """
        Loads the parameters saved in bestmodel.pth into the model
        """
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(SAVE_PATH))
        else: 
            self.load_state_dict(torch.load(SAVE_PATH, map_location=torch.device('cpu')))
        
    def train(self, train_input, train_target, nb_epochs=10, verbose=0,  SAVE_PATH='Proj_287630_282604_288453/Miniproject_2/bestmodel.pth'):
        """
        :param train_input: tensor of size (N, C, H, W) containing a noisy version of the images.
        :param train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.
        """
    
        #move data to the device (CPU or GPU)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #train_input, train_target = train_input.to(device), train_target.to(device)

        #creating the dataset
        #training_dataset = torch.utils.data.TensorDataset(train_input, train_target)
        #Batching the data
        #training_generator = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=10, shuffle=True)     
        train_loss = []

        for e in range(nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.forward(train_input.narrow(0, b, self.mini_batch_size))
                print("output size", output.size())
                print("target size", train_target.narrow(0, b, self.mini_batch_size).size())
                self.loss = self.criterion.compute_loss(output, train_target.narrow(0, b, self.mini_batch_size))
                acc_loss = acc_loss + self.loss

                #self.zero_grad() ???

                # Backward-pass
                grad_wrt_output = self.criterion.compute_backward_pass_gradient(output, train_target.narrow(0, b, self.mini_batch_size))
                self.net.backward(grad_wrt_output)
                # Gradient step
                self.gradient_step(step_size)
            

            step_size = step_size * 0.9
            #for plotting the loss
            train_loss.append(acc_loss)

            if verbose: print(e, acc_loss)

        #Saving the model
        #if SAVE_PATH is not None : torch.save(self.state_dict(), SAVE_PATH)

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

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #test_input = test_input.to(device)

        print("predict model")
        losses = []
        model_outputs = []
        for b in range(0, test_input.size(0), mini_batch_size):
            output = self(test_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output)
        model_outputs = torch.cat(model_outputs, dim=0)
        return model_outputs