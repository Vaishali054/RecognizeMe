import torch
import torch.nn as nn

class FER2013CNN(nn.Module):
    def __init__(self):
        super(FER2013CNN, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

        # Layer 4
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.bn4 = nn.BatchNorm1d(1024)

        # Layer 5
        self.fc2 = nn.Linear(1024, 7)  
        self.dropout = nn.Dropout(0.3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
