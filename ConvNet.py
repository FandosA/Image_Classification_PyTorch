# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:47:30 2022

@author: andre
"""

import torch.nn as nn


class ConvNet(nn.Module):
    
    def __init__(self, num_classes=6):
        
        super(ConvNet, self).__init__()
        
        # Input shape = (batch_size, num_channels, 150, 150)
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1.out_channels)
        self.relu1 = nn.ReLU()
        # Shape = (batch_size, 16, 150, 150)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Output shape = (batch_size, 16, 75, 75)
        
        
        # Input shape = (batch_size, 16, 75, 75)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels,
                               out_channels=self.conv1.out_channels * 2,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.relu2 = nn.ReLU()
        # Shape = (batch_size, 32, 75, 75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Output shape = (batch_size, 32, 37, 37)
        
        
        # Input shape = (batch_size, 32, 37, 37)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels,
                               out_channels=self.conv2.out_channels * 2,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3.out_channels)
        self.relu3 = nn.ReLU()
        # Shape = (batch_size, 64, 37, 37)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # Output shape = (batch_size, 64, 18, 18)
        
        
        # Input shape = (batch_size, 64, 18, 18)
        self.flatten = nn.Flatten()
        # Output shape = (batch_size, 64*18*18) = (batch_size, 20736)
        
        
        # Input shape = (batch_size, 20736)
        self.fc1 = nn.Linear(in_features=self.conv3.out_channels * 18 * 18,
                             out_features=self.conv3.out_channels * 18 * 18 // 4)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        # Output shape = (batch_size, 5184)
        
        
        # Input shape = (batch_size, 5184)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features,
                             out_features=self.fc1.out_features // 4)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        # Output shape = (batch_size, 1296)
        
        
        # Input shape = (batch_size, 1296)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features,
                             out_features=self.fc2.out_features // 4)
        self.relu6 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        # Output shape = (batch_size, 324)
        
        
        # Input shape = (batch_size, 324)
        self.fc4 = nn.Linear(in_features=self.fc3.out_features,
                             out_features=num_classes)
        # Output shape = (batch_size, 6)
        
        
        
    def forward(self, x):
        
        output = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        output = self.maxpool2(self.relu2(self.bn2(self.conv2(output))))
        output = self.maxpool3(self.relu3(self.bn3(self.conv3(output))))
        
        output = self.flatten(output)
        
        output = self.dropout1(self.relu4(self.fc1(output)))
        output = self.dropout2(self.relu5(self.fc2(output)))
        output = self.dropout3(self.relu6(self.fc3(output)))
        output = self.fc4(output)
        
        return output