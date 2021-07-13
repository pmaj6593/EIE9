#from __future__ import print_function
#import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
from torch.utils.data.sampler import SequentialSampler
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import  BinarizeLinear,BinarizeConv2d
from binarized_modules import  Binarize,HingeLoss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 15
learning_rate = 0.005
lr_decay = 3

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
                                             download=True
                                             )

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor()
                                            )

def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):           
        if dataset.targets[i] == class_name:        #Get the index of all data belonging to the class of interest
            indices.append(i)
    return indices

def get_portion_of_data(dataset, labels, div):
    total_indices = []
    for j in labels:                             #Get all indices for each label of interest
        indices = get_indices(dataset, j)   
        indices = indices[0:len(indices)//div]      #Halve the list for that label
        total_indices+=indices                      #Add the halved list to the main main list of indices
    return total_indices

label_nums = [0,1,2,3,4,5,6,7,8,9]

idx_train = get_portion_of_data(train_dataset, label_nums, 1)

random.shuffle(idx_train)                           #Shuffle data so we don't have to use a random sampler (otherwise data will contain each label in order)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=10, 
                                           shuffle=False,
                                           sampler = SequentialSampler(idx_train))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=50, 
                                          shuffle=False)

label_nums = [0,1,2,3,4,5,6,7,8,9]

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,do_bntan=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        #self.conv2 = Binaryconv3x3(planes, planes)
        #self.tanh2 = nn.Hardtanh(inplace=True)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.do_bntan=do_bntan
        self.stride = stride

    def forward(self, x):

        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        #out = self.conv2(out)


        if self.downsample is not None:
            # if residual.data.max()>1:
                # import pdb; pdb.set_trace()
            residual = self.downsample(residual)

        out += residual
        # if self.do_bntan:
        #     out = self.bn2(out)
        #     out = self.tanh2(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block):
        super(ResNet, self).__init__()
        self.inflate = 2
        self.inplanes = 16*self.inflate
        depth = 18
        num_classes = 10
        n = int((depth - 2) / 6)
        self.conv1 = BinarizeConv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.maxpool = lambda x: x
        self.bn1 = nn.BatchNorm2d(16*self.inflate)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.layer1 = self._make_layer(block, 16*self.inflate, n)
        self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2, do_bntan = False)
        # self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2,do_bntan=False)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.bn2 = nn.BatchNorm1d(2*64*self.inflate)
        self.bn3 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax(-1)
        self.fc = BinarizeLinear(128*self.inflate, num_classes)

#in_channels = planes
#out_channels = planes * block_expansion

    def _make_layer(self, block, planes, blocks, stride=1,do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinarizeConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes,do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(torch.clone(x))
        x = self.layer2(torch.clone(x))
        #x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)

        return x

# class ResNet_cifar10(ResNet):

    # def __init__(self, num_classes=10,
    #              block=BasicBlock, depth=18):
    #     super(ResNet_cifar10, self).__init__()
    #     self.inflate = 5
    #     self.inplanes = 16*self.inflate
    #     n = int((depth - 2) / 6)
    #     self.conv1 = BinarizeConv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1,
    #                            bias=False)
    #     self.maxpool = lambda x: x
    #     self.bn1 = nn.BatchNorm2d(16*self.inflate)
    #     self.tanh1 = nn.Hardtanh(inplace=True)
    #     self.tanh2 = nn.Hardtanh(inplace=True)
    #     self.layer1 = self._make_layer(block, 16*self.inflate, n)
    #     self.layer2 = self._make_layer(block, 32*self.inflate, n, stride=2)
    #     self.layer3 = self._make_layer(block, 64*self.inflate, n, stride=2,do_bntan=False)
    #     self.layer4 = lambda x: x
    #     self.avgpool = nn.AvgPool2d(8)
    #     self.bn2 = nn.BatchNorm1d(64*self.inflate)
    #     self.bn3 = nn.BatchNorm1d(10)
    #     self.logsoftmax = nn.LogSoftmax()
    #     self.fc = BinarizeLinear(64*self.inflate, num_classes)

    #     init_model(self)        

model = ResNet(ResidualBlock).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        torch.autograd.set_detect_anomaly(True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    # if (epoch+1) % 20 == 0:
    #     curr_lr /= lr_decay
    #     update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')


