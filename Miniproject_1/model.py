import math
import torch
from torch import optim
from torch import Tensor
from torch import nn
import matplotlib.pyplot as plt

mini_batch_size=100

class Model(torch.nn.Module):
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need 
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size = 3, padding=3//2),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size = 3, padding=3//2), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, kernel_size = 3, padding=3//2),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 3, kernel_size = 3, padding=3//2)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay = 1e-8)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded

    def load_pretrained_model(self, SAVE_PATH ='./bestmodel.pth'):
        ## This loads the parameters saved in bestmodel.pth into the model
        if torch.cuda.is_available():
            print('if')
            self.load_state_dict(torch.load(SAVE_PATH))
        else: 
            print('else')
            self.load_state_dict(torch.load(SAVE_PATH, map_location=torch.device('cpu')))
        


    def train(self, train_input, train_target, nb_epochs=10, verbose=0,  SAVE_PATH='./bestmodel.pth',):
        #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images.
        #:train_target: tensor of size (N, C, H, W) containing another noisy version of the same images, which only differs from the input by their noise.

        #move the model, criterion & data to the device (CPU or GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.criterion.to(device)
        train_input, train_target = train_input.to(device), train_target.to(device)

        #creating the dataset
        #training_dataset = torch.utils.data.TensorDataset(train_input, train_target)
        #Batching the data
        #training_generator = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=10, shuffle=True)     
        train_loss = []

        for e in range(nb_epochs):
            acc_loss = 0

            for b in range(0, train_input.size(0), mini_batch_size):
                output = self.forward(train_input.narrow(0, b, mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                acc_loss = acc_loss + loss.item()
                self.zero_grad()
                loss.backward()
                self.optimizer.step()
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


    def predict(self, test_input):
        #:test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1, C, H, W)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_input = test_input.to(device)
        
        losses = []
        model_outputs = []
        for b in range(0, test_input.size(0), mini_batch_size):
            output = self(test_input.narrow(0, b, mini_batch_size))
            model_outputs.append(output.cpu())
             # Calculating the loss function
            loss = self.criterion(output, test_input.narrow(0, b, mini_batch_size))
        model_outputs = torch.cat(model_outputs, dim=0)
        return model_outputs