import re
import os
import numpy as np
import matplotlib.pyplot as plt



def plot_loss_acc(log_dir):
    
    network_files = os.listdir(log_dir)
    
    train_acc_file = [string for string in network_files if 'train_accuracies' in string]
    train_accs = np.loadtxt(log_dir + '/' + train_acc_file[0])
    
    validation_acc_file = [string for string in network_files if 'val_accuracies' in string]
    validation_accs = np.loadtxt(log_dir + '/' + validation_acc_file[0])
    
    train_loss_file = [string for string in network_files if 'train_losses' in string]
    train_losses = np.loadtxt(log_dir + '/' + train_loss_file[0])
    
    validation_loss_file = [string for string in network_files if 'val_losses' in string]
    validation_losses = np.loadtxt(log_dir + '/' + validation_loss_file[0])
    
    bestEpoch = validation_acc_file[0].split('_')
    bestEpoch = bestEpoch[-1]
    bestEpoch = bestEpoch.split('.')
    bestEpoch = bestEpoch[0]
    bestEpoch = int(re.search(r'\d+', bestEpoch).group())
    
    epochs = np.arange(train_losses.shape[0])
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss", c='b')
    plt.plot(epochs, validation_losses, label="Validation loss", c='r')
    plt.plot(bestEpoch, validation_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.01, validation_losses[bestEpoch]+.01, str(bestEpoch) + ' - ' + str(round(validation_losses[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(log_dir + '/loss.png')
    
    plt.figure()
    plt.plot(epochs, train_accs, label="Training accuracy", c='b')
    plt.plot(epochs, validation_accs, label="Validation accuracy", c='r')
    plt.plot(bestEpoch, validation_accs[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+.001, validation_accs[bestEpoch]+.001, str(bestEpoch) + ' - ' + str(round(validation_accs[bestEpoch], 3)), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(log_dir + '/accuracy.png')
    
    plt.show()
