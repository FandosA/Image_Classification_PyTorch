# Image classification with PyTorch
This is a simple convolutional neural network to classify images with PyTorch. The model has been designed to classify images using the Intel Image Classification dataset, which can be found in Kaggle [here](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). This dataset contains around 25k images (around 14k images in the Training Set, 3k in Test Set and 7k in the Prediction Set) of size 150x150 distributed under 6 categories:

- buildings → 0,
- forest → 1,
- glacier → 2,
- mountain → 3,
- sea → 4,
- street → 5

## Download the dataset
To download the dataset you need to be registered on the Kaggle website. Once registered, you can download the dataset. Then download this repository and unzip the dataset file downloaded to the repository folder. At the end, in the repository folder you should have the next files and folders

![image](https://user-images.githubusercontent.com/71872419/199296300-536e07e8-161e-4403-b5f4-51f8a5f00540.png)

## Train the model
When this is all done, just run the file ```train.py```. You can change the hyperparameters modifying them in the code. After training the model, a new folder with the name set in the parameters will have been created (see image below).

![folder_model](https://user-images.githubusercontent.com/71872419/199300085-b30fa407-1d77-42db-bf08-637c50312792.png)

In this folder you will find the checkpoints (saved every 10 epochs), the images showing the loss and accuracy curves along the epochs, and the text files with the values of accuracy and loss at each epoch. In addition, the checkpoint with the highest accuracy is also saved.

![image](https://user-images.githubusercontent.com/71872419/199299935-a2d31178-6ff9-427c-896f-1452ee0e7e51.png)

