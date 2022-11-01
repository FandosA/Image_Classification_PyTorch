# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:20:14 2022

@author: andre
"""

import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import configargparse
from ConvNet import ConvNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import plot_loss_acc


def compute_accuracy(y_pred, y):
    
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    
    return acc



def create_model_folder(log_dir):
    
    os.mkdir(log_dir)
    
    checkpoints_path = 'checkpoints/'
    checkpoints_path = os.path.join(log_dir, checkpoints_path)
    os.mkdir(checkpoints_path)

    return checkpoints_path



def train(model, train_loader, validate_loader, loss_fn, optimiser, num_epochs,
          batch_size, learning_rate, device, log_dir, checkpoints_path):
    
    
    max_val_acc = 0.0
    
    train_accs = []
    train_losses = []
    
    val_accs = []
    val_losses = []
    
    bestEpoch = 0
    
    print('--------------------------------------------------------------')
    
    # Loop along epochs to do the training
    for i in range(num_epochs):
        
        print(f'EPOCH {i+1}')
        
        # Training loop
        train_accuracy = 0.0
        train_loss = 0.0
        model.train()
        iteration = 1
        
        print('\nTRAINING')
        
        for images, labels in train_loader:
            
            print('\rEpoch[' + str(i+1) + '/' + str(num_epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(train_loader)), end='')
            iteration += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimiser.zero_grad()
            
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            
            loss.backward()
            optimiser.step()
            
            train_accuracy += compute_accuracy(predictions, labels).item()
            train_loss += loss.item()
        
        
        # Validation loop
        val_accuracy = 0.0
        val_loss = 0.0
        model.eval()
        iteration = 1

        print('')
        print('\nVALIDATION')
        
        for images, labels in validate_loader:
            
            print('\rEpoch[' + str(i+1) + '/' + str(num_epochs) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(validate_loader)), end='')
            iteration += 1
            
            images, labels = images.to(device), labels.to(device)
            
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            
            val_accuracy += compute_accuracy(predictions, labels).item()
            val_loss += loss.item()
        
        # Save loss and accuracy values
        train_accs.append(train_accuracy / len(train_loader))
        val_accs.append(val_accuracy / len(validate_loader))
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(validate_loader))
        
        print('\n')
        print(f'- Train Acc: {(train_accuracy / len(train_loader))*100:.2f}%')
        print(f'- Val Acc: {(val_accuracy / len(validate_loader))*100:.2f}%')
        print(f'- Train Loss: {train_loss / len(train_loader):.3f}')
        print(f'- Val Loss: {val_loss / len(validate_loader):.3f}')
        
            
        # Save the model every 10 epochs
        if i % 10 == 0:
            torch.save(model.state_dict(), checkpoints_path + "/checkpoint_" + str(i) + ".pth")
            
        # Save the best model
        if (val_accuracy / len(validate_loader)) > max_val_acc:
            
            # Remove previous best model and save current best model
            if i == 0:
                torch.save(model.state_dict(), checkpoints_path + "checkpoint_" + str(i) + "_best.pth")
            else:
                os.remove(checkpoints_path + "checkpoint_" + str(bestEpoch) + "_best.pth")
                torch.save(model.state_dict(), checkpoints_path + "checkpoint_" + str(i) + "_best.pth")
                
                # Remove previous loss and accuracy files to update the txt files with the best epoch
                os.remove(log_dir + '/train_accuracies_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt')
                os.remove(log_dir + '/val_accuracies_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt')
                os.remove(log_dir + '/train_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt')
                os.remove(log_dir + '/val_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                            '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt')
            
            print(f'\nAccuracy increased ({max_val_acc*100:.6f}% ---> {(val_accuracy / len(validate_loader))*100:.6f}%) \nModel saved')
            
            # Update parameters with the new best model
            max_val_acc = val_accuracy / len(validate_loader)
            bestEpoch = i
            
            
        print("--------------------------------------------------------------")
        
        
        # Save losses and accuracies
        np.savetxt(log_dir + '/train_accuracies_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(train_accs))
        np.savetxt(log_dir + '/val_accuracies_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(val_accs))
        np.savetxt(log_dir + '/train_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(train_losses))
        np.savetxt(log_dir + '/val_losses_epochs' + str(num_epochs) + '_bs' + str(batch_size) +
                    '_lr' + str(learning_rate) + '_bestEpoch' + str(bestEpoch) + '.txt', np.array(val_losses))
        
    
    # Plot losses and accuracy curves
    plot_loss_acc(log_dir)
    
    

if __name__ == "__main__":
    
    # Select parameters for training
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_train', type=str, default='seg_train', help='Dataset train path.')
    p.add_argument('--dataset_val', type=str, default='seg_test', help='Dataset validation path.')
    p.add_argument('--log_dir', type=str, default='image_classification_ConvNet', help='Name of the folder to save the model.')
    p.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    p.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    p.add_argument('--num_epochs', type=int, default=20, help='Number of epochs.')
    p.add_argument('--device', type=str, default='gpu', help='Choose the device: "gpu" or "cpu"')
    opt = p.parse_args()
    
    assert not (os.path.isdir(opt.log_dir)), 'The folder log_dir already exists, remove it or change it'
    
    
    # Select device
    if opt.device == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Device assigned: GPU (' + torch.cuda.get_device_name(device) + ')\n')
    else:
        device = torch.device("cpu")
        if not torch.cuda.is_available() and opt.device == 'gpu':
            print('GPU not available, device assigned: CPU\n')
        else:
            print('Device assigned: CPU\n')
            
        
    transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        # formula: output[channel] = (input[channel] - mean[channel]) / std[channel]
        transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1]
                            [0.5,0.5,0.5])
    ])
    
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(opt.dataset_train, transform=transformer),
        batch_size=opt.batch_size, shuffle=True
    )
    
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(opt.dataset_val, transform=transformer),
        batch_size=opt.batch_size, shuffle=True
    )
    
    checkpoints_path = create_model_folder(opt.log_dir)  # Create model folder to save checkpoints
    model = ConvNet(num_classes=len(os.listdir(opt.dataset_train))).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    train(model, train_loader, test_loader, loss_fn, optimiser, opt.num_epochs,
          opt.batch_size, opt.learning_rate, device, opt.log_dir, checkpoints_path)