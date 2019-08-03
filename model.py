## Import libraries
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle # Shuffle arrays in a consistent way
from sklearn.model_selection import train_test_split # Library to split train and validation datasets

## Initialize variables
batch_size = 32  # Set the batch size

###### Data Pre-processing #############
rows = []             # Initialize an empty array to load each row in csv file
#with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile: # Open csv file
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile: # Open csv file    
    reader = csv.reader(csvfile) # Read the data from the csv file
    next(reader, None) # Skip the first row (header) in the csv file
    for row in reader:
        rows.append(row) # Append each row in csv file into an array

# Split the training and validation dataset. 80% data is split into training dataset and 20% data is split into validation dataset.
training_data, validation_data = train_test_split(rows, test_size=0.2) 

#print('length of training data=', len(training_data))
#print('length of validation data=', len(validation_data))

##### Data Augmentation #####

# Define a function to generate the training and validation data using data augmentation techniques
def data_generator(samples, batch_size):
    num_samples = len(samples) # Total number of training/validation data
    while 1: # Loop forever so the generator never terminates
        # Shuffle the data to generalize the model 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size): # Loop to process data with batches
            batch_samples = samples[offset:offset+batch_size] # Get the data in batch size
            
            images = [] # Initialize an empty array to store the images
            steering_angles = [] # Initialize an empty array to store the steering angles
            for batch_sample in batch_samples: # Process each row from batch samples
                for i in range(0, 3): # loop to process center, left and right paths from csv
                    name = '/opt/carnd_p3/data/IMG/'+ batch_sample[i].split('/')[-1] # Get the name of the image from image URL (Split using '/' and return last item i.e., image name)
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) # Convert image from BGR to RGB since it is processed as RGB in drive.py file.
                    images.append(center_image)
                    
                    steering_center = float(batch_sample[3]) # Get the steering angle value and convert it from string to float
                    
                    if(i==0):
                        correction = 0; # If center image, add no correction factor
                    elif(i==1):
                        correction = 0.2; # If left image, add 0.2 correction factor
                    elif(i==2):
                        correction = -0.2; # If right image, add -0.2 correction factor
 
                    steering_angles.append(steering_center + correction)
                        
                    # Flip the image for data augmentation
                    images.append(cv2.flip(center_image,1))
                    steering_angles.append((steering_center + correction)*-1)
              
            X_samples = np.array(images)
            y_samples = np.array(steering_angles)
                        
            yield sklearn.utils.shuffle(X_samples, y_samples)                       

# Generator function to work with large amount of data with batches
training_data_generator = data_generator(training_data, batch_size)
validation_data_generator = data_generator(validation_data, batch_size)

###### Model Architecture
# The NVIDIA network achitecture is chosen
# It consists of a Normalization layer, 5 Convolutional layers and 4 fully connected layers
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
                        
model = Sequential()

# Preprocessing the data using Normalization and Mean Centering the data
# For Normalization, Lamba layer is added. Normalization is done by dividing each element by 255 which is the max value of image pixel.
# When the image is normalized to range between 0 and 1, mean centering of image is done by subtracting pixels by 0.5.
# Input shape = 160x320x3
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Cropping portion of image to impove model accuracy
# Input shape = 160x320x3, cropping=160-(70+25)=65
# Output shape = 65x320x3
model.add(Cropping2D(cropping=((70,25),(0,0))))           

#output_height = [(input_height - filter_height + 2 * padding) / vertical_stride] + 1
#output_width = [(input_width - filter_width + 2 * padding) / vertical_stride] + 1
#output_depth = number of filters

# Layer 1: Convolutional Layer
# Input shape = 65x320x3, Number of filters = 24, Filter size = 5x5, Vertical Stride=2x2
# output_height = [(65-5+2*0)/2]+1 = 31.
# output_width = [(320-5+2*0)/2]+1 = 158.
# output_depth = filter_depth = 24
# Output shape = 31x158x24
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="elu"))

# Layer 2: Convolutional Layer
# Input shape = 31x158x24, Number of filters = 36, Filter size = 5x5, Vertical Stride=2x2
# output_height = [(31-5+2*0)/2]+1 = 14.
# output_width = [(158-5+2*0)/2]+1 = 77.
# output_depth = filter_depth = 36
# Output shape = 14x77x36
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="elu"))

# Layer 3: Convolutional Layer
# Input shape = 14x77x36, Number of filters = 48, Filter size = 5x5, Vertical Stride=2x2
# output_height = [(14-5+2*0)/2]+1 = 5.
# output_width = [(77-5+2*0)/2]+1 = 37.
# output_depth = filter_depth = 48
# Output shape = 5x37x48
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="elu"))

# Layer 4: Convolutional Layer
# Input shape = 5x37x48, Number of filters = 64, Filter size = 3x3, Vertical Stride=1x1
# output_height = [(5-3+2*0)/1]+1 = 3.
# output_width = [(37-3+2*0)/1]+1 = 35.
# output_depth = filter_depth = 64
# Output shape = 3x35x64
model.add(Convolution2D(64, 3, 3, activation="elu"))

# Layer 5: Convolutional Layer
# Input shape = 3x35x64, Number of filters = 64, Filter size = 3x3, Vertical Stride=1x1
# output_height = [(3-3+2*0)/1]+1 = 1.
# output_width = [(35-3+2*0)/1]+1 = 33.
# output_depth = filter_depth = 64
# Output shape = 1x33x64
model.add(Convolution2D(64, 3, 3, activation="elu"))

# Flatten the output
# Input shape = 1x33x64, Output shape = 2112
model.add(Flatten())

# Layer 6: Fully Connected Layer 1
# Input shape = 2112, Output shape = 100
model.add(Dense(100))
model.add(Activation('elu'))

# Added a dropout layer to avoid overfitting. 25% training data is dropped at this stage.
model.add(Dropout(0.25))

# Layer 7: Fully Connected Layer 2
# Input shape = 100, Output shape = 50
model.add(Dense(50))
model.add(Activation('elu'))

# Layer 8: Fully Connected Layer 3
# Input shape = 50, Output shape = 10
model.add(Dense(10))
model.add(Activation('elu'))

# Layer 9: Fully Connected Layer 4. This is final layer and it contains only one output bcz this is a regression model.
# The output is the steeing angle.
# Input shape = 10, Output shape = 1
model.add(Dense(1))

# Mean Squared Error loss function is used as it is best for regression models
model.compile(loss='mse',optimizer='adam')

# Number of epochs= 5
model.fit_generator(training_data_generator, samples_per_epoch= len(training_data), validation_data=validation_data_generator,   nb_val_samples=len(validation_data), nb_epoch=5, verbose=1)

# Save the model
model.save('model.h5')

print('Model Saved!')

# Print the Model Summary
model.summary()