# Image classification with PyTorch
This is a simple convolutional neural network to classify images with PyTorch. The model has been designed to classify images using the Intel Image Classification dataset, which can be found in Kaggle [here](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). This dataset contains around 25k images (around 14k images in the Training Set, 3k in Test Set and 7k in the Prediction Set) of size 150x150 distributed under 6 categories:

- buildings → 0,
- forest → 1,
- glacier → 2,
- mountain → 3,
- sea → 4,
- street → 5

## Download and prepare the dataset
To download the dataset you need to be registered on the Kaggle website. Once registered, you can download the dataset. Then download this repository and unzip the dataset file downloaded to the repository folder. At the end, in the repository folder you should have the next files and folders (as shown in the image below):
- seg_pred (folder)
- seg_test (folder)
- seg_train (folder)
- ConvNet.py (file)
- test.py (file)
- train.py (file)
- utils.py (file)

![image](https://user-images.githubusercontent.com/71872419/199296300-536e07e8-161e-4403-b5f4-51f8a5f00540.png)
