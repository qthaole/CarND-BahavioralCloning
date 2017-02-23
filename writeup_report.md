#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[brightness]: brightness.png "Brightness augmentation"
[translation]: translation.png "Translation augmentation"
[architecture]: architecture.png "Architecture Image"
[preprocessed]: preprocessed.png "Preprocessed Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 177-187).

Data normalization is applied in the model by using a Keras lamda layer (model.py code line 173).

In the model, RELU layers are used to introduce nonlinearity (model.py between code lines 178-211).

Fully-connected and Dropout layers are also present in the model.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 180-212). 

The orignal data set is split into training and validation sets. The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). 

The model was tested by using it to run the simulator in autonomous mode and the result showed that the car could stay on the track.

####3. Model parameter tuning

I used the adam optimizer (so the learning rate was not tuned manually) and mse loss function in the model (model.py line 216).

####4. Appropriate training data

For training, I used the Udacity data to generate more training data. For some techniques of data augmentation and their implementation, I got my inspiration from Vivek Yadav who very kindly shared his approach on this [article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.7vaxw7z36)

For details about how I created the training data are discussed the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a model complex enough to modelize the driving behaviors,  avoid overfitting and require an acceptable computaional cost. 

My first step was to use a convolution neural network model similar to the LeNet architecture: convoluation layers followed by full-connected layers. This architecture proves to be working well on the previous project.

When trained without Dropout layers, the model perormed well on the training set but poorly on the validation set. This is a sign of overfitting. To oversome the overfitting problem, I added Dropout layers the model and this proved to help the model perform better on the validation data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track: tough left turn after the bridge, and tough right turn ater that. Improve the driving behavior in these cases, I tried several kernel sizes for convolution layers and found that a kernel size of (3,3) worked well for me. Also, I tuned the parameters used in my data augmentation techniques to further improve the overall performance of the model.

At the end of the process, the vehicle is able to drive autonomously around track one without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 171-214) consisted of a convolution neural network with the following layers and layer sizes:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 62, 62, 32)    896         lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 62, 62, 32)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 31, 31, 32)    0           elu_1[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 31, 31, 32)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 29, 29, 64)    18496       dropout_1[0][0]                  
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 29, 29, 64)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 14, 14, 64)    0           elu_2[0][0]                      
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 14, 14, 64)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 12, 12, 128)   73856       dropout_2[0][0]                  
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 12, 12, 128)   0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 6, 6, 128)     0           elu_3[0][0]                      
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 6, 6, 128)     0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4608)          0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           2359808     flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 512)           0           elu_4[0][0]                      
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           65664       dropout_4[0][0]                  
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 128)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 128)           0           elu_5[0][0]                      
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 64)            8256        dropout_5[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 64)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 64)            0           elu_6[0][0]                      
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 32)            2080        dropout_6[0][0]                  
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 32)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 32)            0           elu_7[0][0]                      
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 16)            528         dropout_7[0][0]                  
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 16)            0           dense_5[0][0]                    
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 16)            0           elu_8[0][0]                      
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 1)             17          dropout_8[0][0]                  
====================================================================================================
Total params: 2,529,601
Trainable params: 2,529,601
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

To have the training set, I used the training data provided by Udacity to generate abundant data to improve the generialization of the model. The techniques used are: image translation, brightness modification, usage of images captured by left and right cameras.

The fact that input images are translated horizontally helps simulate scenarios in which the car is in different position of the road, teach the car know to recover to the center. Below are some examples of translated images:

![alt text][translation]

Brightness modification further generalizes level of the model by simulating different light conditions (day, night, sunny, cloudy, etc...). Some examples of data generated by this technique:

![alt text][brightness]

Finally, the image is cropped at the top and the bottom to remove irrevelant information such as background trees and other objects, sky.... and resized to 64x64. At 64x64, information loss is acceptable, the training images remains clear and their visual quality is suffficent for training.

Here are some final preprocessing results:

![alt text][preprocessed]

For each epoch, I generated 25600 training data points.

Before running the training process, I randomly shuffled the data set and put 0.2% of the data into a validation set. The validation set helped determine if the model was over or under fitting. 

After 4 epochs, the model proved to perform well enough on the validation set: 0.0355. I decided to stop at 4 epochs since the trained model was already able to drive the car successfully on the track one.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
