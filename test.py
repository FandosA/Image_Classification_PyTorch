import os
import json
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import configargparse
from ConvNet import ConvNet
from torchvision.transforms import transforms


def predict(dataset_pred, img_name, transformer):
    
    image = Image.open(dataset_pred + '/' + img_name)
    image_tensor = torch.unsqueeze(transformer(image), dim=0)
    image_tensor = image_tensor.to(device)
    
    predictions = model(image_tensor)
    
    softmax = nn.Softmax(dim=1)
    predictions_softmax = softmax(predictions)
    predictions_softmax = torch.squeeze(predictions_softmax)
    predictions_softmax = predictions_softmax.cpu().detach().numpy()
    
    max_index = np.argmax(predictions_softmax)
    probability = predictions_softmax[max_index]
    predicted_class = classes[max_index]
    
    return predicted_class, probability, predictions_softmax



if __name__ == "__main__":
    
    # Select parameters for predictions
    p = configargparse.ArgumentParser()
    p.add_argument('--dataset_pred', type=str, default='seg_pred', help='Dataset predictions path.')
    p.add_argument('--log_dir', type=str, default='image_classification_ConvNet', help='Name of the folder to load the model.')
    p.add_argument('--checkpoint', type=str, default='checkpoint_18_best.pth',help='Checkpoint path')
    p.add_argument('--device', type=str, default='gpu', help='Choose the device: "gpu" or "cpu"')
    opt = p.parse_args()
    
    assert os.path.isdir(opt.log_dir), 'The folder log_dir does not exists'
    
    
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
            
            
    classes = os.listdir("seg_test")
    images = os.listdir(opt.dataset_pred)
    num_images = len(images)
    
    transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        # formula: output[channel] = (input[channel] - mean[channel]) / std[channel]
        transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1]
                            [0.5,0.5,0.5])
    ])
    
    model = ConvNet(num_classes=len(classes)).to(device)
    state_dict = torch.load(opt.log_dir + "/checkpoints/" + opt.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    predictions_dict={
        # Name of the image
        "image": [],
        
        # Class predicted for the image
        "class": [],
        
        # Probability that the image is the predicted class
        "probability": [],
        
        # Probabilities predicted for each class
        "probabilities": []
    }

    for i, img_name in enumerate(images):
        
        print('\rPredicting images... [' + str(i+1) + '/' + str(num_images) + ']', end='')
        
        predicted_class, probability, predictions = predict(opt.dataset_pred, img_name, transformer)
        
        predictions_dict["image"].append(img_name)
        predictions_dict["class"].append(predicted_class)
        predictions_dict["probability"].append(str(probability))
        predictions_dict["probabilities"].append(predictions.tolist())
        
        
    with open(opt.log_dir + '/predictions.json', "w") as fp:
        json.dump(predictions_dict, fp, indent=4)
