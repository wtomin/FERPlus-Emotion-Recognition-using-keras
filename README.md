# FERPlus-Emotion-Recognition-using-keras
This is a repository about using keras to do emotion recognition on FERPlus dataset

## FERPlus dataset
The [FERPlus (FER+)](https://github.com/microsoft/FERPlus) annotations provide a set of new labels for the standard Emotion [FER dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), which is thought to be more accuract than the original FER2013 annotations.

The new label file is named [fer2013new.csv]((https://github.com/microsoft/FERPlus)) and contains the same number of rows as the original [fer2013.csv]((https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)) label file with the same order, so that you infer which emotion tag belongs to which image. 

## The main purpose

The main purpose of this repository is to replicate the pretrained CNN models' performances in this [blog](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html), which has the model weights in Pytorch and MatConvNet, but not in tensorflow or keras.
