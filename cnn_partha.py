import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import torch.nn.functional as F

#Download training data
training_data = datasets.MNIST(
    root = "MNIST",
    train = True,
    download = True,
    transform = ToTensor(),
    #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.MNIST(
    root = "MNIST",
    train = False,
    download = True,
    transform = ToTensor(),
    #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cpu"
print("Using {} device".format(device))

# Adaptor class added onto the code
class Adaptor(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 256)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        #nn.ReLU(),
        self.fc1 = nn.Linear(256, 128)
        #nn.ReLU(),
        self.fc2 = nn.Linear(128, 64)
        #nn.ReLU(),
        self.fc3 = nn.Linear(64 ,10)
        

    def forward(self, x, adaptor):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x,1)
        x = x.view(x.size(0), -1)

        ### ADAPTOR MODULE ###
        # If there exists an adaptor, apply it
        if (adaptor != -1):
            x = adaptor(x)
            for param in self.parameters():
                param.requires_grad = False
        ### ADAPTOR MODULE ###

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork()
adaptor = Adaptor()
# adaptor = -1 
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

if (adaptor != -1):
    adap_optim = optimizer = torch.optim.SGD(adaptor.parameters(), lr = 0.001)
else:
    adap_optim = -1

def train(dataloader, model, loss_fn, optimizer, adaptor, adap_optim):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        #prediction error
        pred = model(X, adaptor)
        loss = loss_fn(pred, y)

        #Backpropagation
        if (adaptor != -1):
            optimizer.zero_grad()
            adap_optim.zero_grad()
            loss.backward()
            optimizer.step()
            adap_optim.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:<7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, adaptor):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X, adaptor)
            test_loss += loss_fn(pred, y).item()
            pred = model(X, adaptor)
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")

epochs = 5

model.load_state_dict(torch.load("MNIST_sd.pth"))

for t in range(epochs):
    print(f"Epoch {t+1}\n------------")
    train(train_dataloader, model, loss_fn, optimizer, adaptor, adap_optim)
    test(test_dataloader, model, loss_fn, adaptor)
print("Done!")

torch.save(model.state_dict(), "ADAPTER.pth")
# test(test_dataloader, model, loss_fn, adaptor)

# torch.save(model.state_dict(), "MNIST_RETRAIN.pth")
# print("Saved PyTorch Model State to MNIST.pth")