import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

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

def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):           
        if dataset.targets[i] in class_name:        #Get the index of all data belonging to the class of interest
            indices.append(i)
    return indices

train_labels = [0,1,2,3,4,5,6]
test_labels = [0,1,2,3,4,5,6]
idx_train = get_indices(training_data, train_labels)       #change second argument for different classes
idx_test = get_indices(test_data, test_labels)

train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler = SubsetRandomSampler(idx_train))   #samples from the modified dataset
test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler = SubsetRandomSampler(idx_test))

print(len(idx_train))
print(len(idx_test))

# for idx, (data, target) in enumerate(train_dataloader):
#     print(target)

#print(train_dataloader)
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cpu"
print("Using {} device".format(device))

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
        

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x,1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        #prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%500 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:<7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, idx_test):
    size = len(dataloader.dataset)
    print(size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = model(X)
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= len(idx_test)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn, idx_test)
print("Done!")

FILE = "MNIST_sd.pth"
torch.save(model.state_dict(), FILE)
print("Saved PyTorch Model State to MNIST.pth")