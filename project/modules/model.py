import torch.nn as nn
import torch.nn.functional as F

# hyperparams
DROPOUT_1 = .5
DROPOUT_2 = .5

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        padding_size = 3

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, padding=padding_size)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, padding=padding_size)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=7, padding=padding_size)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, padding=padding_size)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=7, padding=padding_size)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=7, padding=padding_size)
        self.bn6 = nn.BatchNorm2d(512)

        # Fully connected layers
        self.fc1 = nn.Linear(512*3*3, 1024) 
        self.dropout1 = nn.Dropout(DROPOUT_1)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(DROPOUT_2)
        self.fc3 = nn.Linear(512, 128)

    def forward(self, x):
        
        # Convolutional layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, kernel_size=2)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activations and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  

        return x