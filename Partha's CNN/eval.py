import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
#from script1 import NeuralNetwork


test_data = datasets.MNIST(
    root = "MNIST",
    train = False,
    download = True,
    transform = ToTensor(),
    #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        #nn.Flatten(),
        self.fc1 = nn.Linear(256, 256)
        #nn.ReLU(),
        self.fc2 = nn.Linear(256, 256)
        #nn.ReLU(),
        self.fc3 = nn.Linear(256, 128)
        #nn.ReLU(),
        self.fc4 = nn.Linear(128, 64)
        #nn.ReLU(),
        self.fc5 = nn.Linear(64 ,10)

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x,1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

#model = NeuralNetwork()
model = torch.load("MNIST.pth")

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')