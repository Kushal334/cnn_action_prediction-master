# Action and Action Class Prediction: Project Overview

- Aim of the project is to develop Convolutional Neural Network for a multi-output predictions.
- Obtained 3030 training and 2100 testing images containing 21 different actions under 5 action classes.
- Since the number of training samples is small of approximate 1940 images, data augmentation is used which is a technique to randomly transform the images to artificially expand the training size.
- Model developed comprised of two parts: one is used as the feature extractor with Transfer Learning that is made up of **MobileNetV2** blocks, and the other is the classifier part which is made up of the fully connected layers and the output layer with Softmax as the chosen activation function for the final layer. 
- Model achieved 81% in predicting **action** and 92% in predicting **action class** on an Identical and Independent Dataset (IID).


## Reference

**Python Version:** 3.6 Google Colab <br/>
**GPU:** NVIDIA Tesla K80 GPU  <br/>
**Packages:** numpy, pandas, seaborn, matplotlib, tensorflow, keras, Image <br/>
**CNN Article:** [F-beta Score in Keras](https://towardsdatascience.com/f-beta-score-in-keras-part-i-86ad190a252f) <br/>
**Metric Article:** [A Simple CNN: Multi Image Classifier](https://towardsdatascience.com/a-simple-cnn-multi-image-classifier-31c463324fa) <br/>
**Data Generators:** [Keras data generators and how to use them](https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c) <br/>

# Introduction

- Aim of this project is to develop a deep convolutional neural network (CNN) for a multi-output classifier that can identify the actions of a person from still images.
- Before carrying out any investigation, it is clear a custom data/ image generator is required as there is insufficient memory to load the dataset.
- There are 3030 training and 2100 testing images containing 21 different actions under 5 action classes.
- The analysis is first done by carrying out Exploratory Data Analysis where images generated from the data generator is visualized.

# Data Preprocessing

- Data preprocessing is done on the dataset to encode all the categorical variables of actions and their action classes using Keras. 
- Further pre-processing is done by splitting the train dataset into 20% test and 80% train set. 
- The train set is then further split into 80% training and 20% validation data which will be used for model evaluations and improvements.

# Exploratory Data Analysis 

![](https://github.com/roywong96/cnn_action_prediction/blob/master/images/diffSizeimages.png)

Several Observations can be made from the intitial exploration of these images.

- Images have good similarity to common natural laguage dataset like imagenet. Transfer learning is an option.
- Images have different shapes. These images requires a common shape for transformation.
- Some exmaple images are ambiguous. The final performance for the model may be affected.
- In some images, the important information and features is toward a corner of the image. Data Augmentation needs to be done carefully.

As observed, we can augment the images within the data generator.

<p float="left">
    <img src="https://github.com/roywong96/cnn_action_prediction/blob/master/images/augmented_images.png" width="50%" height="50%">
</p>


# Model Building


- Before any neural network is being developed, 
- **Baseline model** is developed with Keras functional API that has a VGG-type network which have two convolutional layers with 3x3 filters along with a max pooling layer.
- Since model contains more complexity and features within each images, transfer learning with pre-trained model is required for model building.
- As a result, **MobileNetV2** is selected for feature extraction because of its appropriate complexity for the task at hand.
- The model development is followed by adding fully connected layers for classifying the action and the action class.

<p allign="left">
    <img src="https://github.com/roywong96/cnn_action_prediction/blob/master/images/model.png" width="50%" height="50%">
</p>

## Experiments and Tuning

The goal of this project is to achieve a 70% accuracy in prediction both classes.

- Augmented and non-augmented images were first fitted with **Baseline Model** under 100 epochs. 
- The results revealed unrepresentative validation dataset. It implies that validation set did not provide sufficient information to evaluate the ability for the model to generalize.


<p float="left">
    <img src="https://github.com/roywong96/cnn_action_prediction/blob/master/images/aug_train_loss.png" width="30%" height="30%">
    <img src="https://github.com/roywong96/cnn_action_prediction/blob/master/images/ori_train_loss.png" width="30%" height="30%">
</p>


# Error Analysis


<p float="left">
    <img src="https://github.com/roywong96/cnn_action_prediction/blob/master/images/action_error.png" width="50%" height="50%">
    <img src="https://github.com/roywong96/cnn_action_prediction/blob/master/images/action_class_error.png" width="40%" height="40%">
</p>
