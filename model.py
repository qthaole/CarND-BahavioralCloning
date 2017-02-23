import numpy as np
import os
import csv
import cv2
import sklearn
from sklearn.utils import shuffle
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras import initializations

#Reading the data lines from the driving log file used to train the model
samples = [] #data lines will be fed into samples
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Parameters for preprocessing the data and for training the model
model_name = 'model.h5' #the file name of the trained model to save
img_size_x, img_size_y = (64, 64) #the size of the preprocessed image as input of the model
init = 'lecun_uniform' #initialization method
#batch size and epoch
batch_size = 256
epoch_number = 4
#Translation parameters for autmenting the training images with image translation
translate_x_range = 40 #translation limit on x axis
translate_y_range = 10 #translation limit on y axis
angle_change_per_pixel = 0.025 # angle change per pixel when the input image is translated on x axis
camera_left_right_angle_correction = 0.25 # angle correction when using images from left and right cameras

top, bottom = (30, 15) #offsets to crop the image at the top and the bottom

#Randomly change the input input's brightness
def augment_brightness(image):
    img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .3 + np.random.uniform()

    img[:,:,2] = img[:,:,2]*random_bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

#Randomly translate the image and adapt the steering angle
def translate_image(img, angle):
    translate_x = np.random.uniform(-1, 1) * translate_x_range
    translate_y = np.random.uniform(-1, 1) * translate_y_range

    angle = angle  + angle_change_per_pixel * translate_x

    trans_m = np.float32([[1,0,translate_x],[0,1,translate_y]])
    image_translated = cv2.warpAffine(img,trans_m,(img.shape[1],img.shape[0]))

    return image_translated, angle

#The function performs augmentation on the input image
def augment_imge(img, angle):
    img = augment_brightness(img)

    img, new_angle = translate_image(img, angle)

    return img, new_angle

#This functions is used to preprocess training set
def preprocessing_training(data_line):
    angle = float(data_line[3])

    #Randonly choose image from the center, left and right cameras
    image_select_index = np.random.randint(3)
    #update the steering angle accodringly
    if image_select_index == 0:#center
        angle = angle
    elif image_select_index == 1:#left camera
        angle += camera_left_right_angle_correction
    elif image_select_index == 2: #right camera
        angle -= camera_left_right_angle_correction

    #get the image file name
    name = './data/IMG/'+data_line[image_select_index].split('/')[-1]

    #read the image
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #augment
    img, angle = augment_imge(img, angle)

    #crop and resize
    shape = img.shape
    img = img[top:shape[0]-bottom, 0:shape[1]]
    img = cv2.resize(img, (img_size_x, img_size_y), interpolation=cv2.INTER_AREA)

    return img, angle

#This functions is used preprocess validation data
def preprocessing_validation(data_line):
    name = './data/IMG/'+data_line[0].split('/')[-1]
    angle = float(data_line[3])
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #crop and resize
    shape = img.shape
    img = img[top:shape[0]-bottom, 0:shape[1]]
    img = cv2.resize(img, (img_size_x, img_size_y), interpolation=cv2.INTER_AREA)

    return img, angle



#This generator is used to generate the data set for training at each batch
#It randomly picks data lines from the data
def generator_training(samples, batch_size=32):
    while 1: # Loop forever so the generator never terminates
        images = []
        angles = []
        batch_no = 0
        while batch_no < batch_size:
            #randomly pick the data
            index = np.random.randint(len(samples))
            sample = samples[index]

            name = './data/IMG/'+sample[0].split('/')[-1]
            if name == './data/IMG/center':
                    continue;

            #preprocess, augment the image
            processed_image, angle = preprocessing_training(sample)

            images.append(processed_image)
            angles.append(angle)
            batch_no += 1

        yield np.array(images), np.array(angles)

#This generator is used to generate the data set for validation
def generator_validation(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        images = []
        angles = []
        batch_no = 0
        while batch_no < batch_size:
            #randomly pick the data
            index = np.random.randint(len(samples))
            sample = samples[index]

            name = './data/IMG/'+sample[0].split('/')[-1]
            if name == './data/IMG/center':
                    continue;

            #preprocess the image
            processed_image, angle = preprocessing_validation(sample)

            images.append(processed_image)
            angles.append(angle)
            batch_no += 1

        yield np.array(images), np.array(angles)


#The model architecture
def get_model(time_len=1):
  ch, row, col = 3, img_size_y, img_size_x

  model = Sequential()

  model.add(Lambda(lambda x: x/255. - 0.5,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))

  model.add(Convolution2D(32, 3, 3, border_mode="valid", init=init))
  model.add(ELU())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(.5))

  model.add(Convolution2D(64, 3, 3, border_mode="valid", init=init))
  model.add(ELU())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(.5))

  model.add(Convolution2D(128, 3, 3, border_mode="valid", init=init))
  model.add(ELU())
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(.5))

  model.add(Flatten())

  model.add(Dense(512, init=init))
  model.add(ELU())
  model.add(Dropout(.5))

  model.add(Dense(128, init=init))
  model.add(ELU())
  model.add(Dropout(.5))

  model.add(Dense(64, init=init))
  model.add(ELU())
  model.add(Dropout(.5))

  model.add(Dense(32, init=init))
  model.add(ELU())
  model.add(Dropout(.5))

  model.add(Dense(16, init=init))
  model.add(ELU())
  model.add(Dropout(.5))

  model.add(Dense(1, init=init))

  model.compile(optimizer="adam", loss="mse")

  return model


# compile and train the model using the generator function
#shuffle the data
samples = shuffle(samples)

#split into train and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#generators from training and validation
train_generator = generator_training(train_samples, batch_size=batch_size)
validation_generator = generator_validation(validation_samples, batch_size=batch_size)

#get the compiled model
model = get_model()

#train the model
history = model.fit_generator(train_generator,
                    samples_per_epoch=25600,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=epoch_number)

#save the model
model.save(model_name)
