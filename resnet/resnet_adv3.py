import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
learning_rate = 0.001
adaptor_on = 1
batch_size = 50

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())
# Data Loader, and Sorting
def get_indices(dataset,class_name):
    indices = []
    for i in range(len(dataset.targets)):           
        if dataset.targets[i] in class_name:        #Get the index of all data belonging to the class of interest
            indices.append(i)
    return indices

train_labels = [0,1,2,3,4,5,6]
test_labels = [0,1,2,3,4,5,6]

idx_train = get_indices(train_dataset, train_labels)       #change second argument for different classes
idx_test = get_indices(test_dataset, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = SubsetRandomSampler(idx_train))   #samples from the modified dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler = SubsetRandomSampler(idx_test))

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class Adaptor(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvAdaptor(nn.Module):
    def __init__(self, out_channels):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = conv3x3(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)

        ## Insert Adaptor Module Here ###
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, adaptor, c_adaptor):

        if (adaptor != -1 or c_adaptor != -1):
            for param in self.parameters():
                param.requires_grad = False

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)

        if (c_adaptor != -1):
            out = c_adaptor(out)

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        if (adaptor != -1):
            out = adaptor(out)

        out = self.fc(out)
        return out

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model.load_state_dict(torch.load("ResNet_pt.pth"))

if (adaptor_on == 1):
    adaptor = Adaptor()
    adap_optim = torch.optim.Adam(adaptor.parameters(), lr=learning_rate)
    c_adaptor = ConvAdaptor(16)
    c_adap_optim = torch.optim.Adam(c_adaptor.parameters(), lr=learning_rate)
else:
    adaptor = -1
    adap_optim = -1
    c_adaptor = -1
    c_adap_optim = -1

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(train_loader, model, criterion, optimizer, adaptor, adap_optim, c_adaptor, c_adap_optim):
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images, adaptor, c_adaptor)
        loss = criterion(outputs, labels)

        # Backward and optimize
        if (adaptor == -1 and c_adaptor == -1):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif (adaptor == -1):
            optimizer.zero_grad()
            c_adap_optim.zero_grad()
            loss.backward()
            optimizer.step()
            c_adap_optim.step()
        elif (c_adaptor == -1):
            optimizer.zero_grad()
            adap_optim.zero_grad()
            loss.backward()
            optimizer.step()
            adap_optim.step()
        else:
            optimizer.zero_grad()
            adap_optim.zero_grad()
            c_adap_optim.zero_grad()
            loss.backward()
            optimizer.step()
            adap_optim.step()
            c_adap_optim.step()

        if (i+1) % (len(train_loader)/5) == 0:
            print ("Loss: {:.4f}".format(loss.item()))

# Test the model
def test(test_loader, model, criterion, adaptor, c_adaptor):
    with torch.no_grad():
        test_loss, correct = 0, 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, adaptor, c_adaptor)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader)
        correct /= total
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")

for t in range(num_epochs):
    print(f"Epoch [{t+1}/{num_epochs}]\n------------")
    train(train_loader, model, criterion, optimizer, adaptor, adap_optim, c_adaptor, c_adap_optim)
    test(test_loader, model, criterion, adaptor, c_adaptor)

# test(test_loader, model, criterion, adaptor, c_adaptor)
# Save the model checkpoint
# torch.save(model.state_dict(), 'retrain.pth')