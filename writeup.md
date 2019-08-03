# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Capture.PNG "Model Architecture"
[image2]: ./examples/data.PNG "driving_log.csv"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md file summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

#### Strategies for Collecting Data
We need to come up a with right strategy to collect data from simulator in order to ensure a successful model.
- the car should stay in the center of the road as much as possible
- if the car veers off to the side, it should recover back to center
- driving counter-clockwise can help the model generalize
- flipping the images is a quick way to augment the data
- collecting data from the second track can also help generalize the model
- we want to avoid overfitting or underfitting when training the model
- knowing when to stop collecting more data

#### Data Visualization
Simulator data consists of images. Driving data is recorded/logged into driving_log.csv file. Each line in this file represents a point in time during my training lap. Each line has 7 tokens.
- The first 3 tokens in each line are paths to images, one each for cameras mounted on center, left and right of the vehicle's windshield. 
- The next 4 tokens are measurements for steering, throttle, break and speed. The steering measurements range between -1 and +1. 
- The throttle measurements range between 0 and +1.
- The break measurements seem be to zero all the time.
- Speed measurement range between 0 to 30.
The throttle, break and speed data could all be useful for training network, but ignored in this project.
![alt text][image2]
I have used images as 'feature' set and steering measurements as 'label' set. Then I used images to train the network to predict steering measurements.

#### Data Pre-processing
I used the dataset provided by Udacity for training and validating the model. I then split 80% data into training dataset and 20% data into validation dataset. As the training data is large in size, model is trained as batches. A function called 'generator' is used(defined) for this purpose.

Data Augmentation Techniques were chosen to generate more data which helps to generalize the model and avoid over fitting. The techniques used for data augmentation are as follows,
- Correction factor of +/-0.2 is added to steering angle measurement for left and right images so that they can be treated as center image.
- For each left and right images measure the corresponding steering angle as follows, steering_angle=steering_center+correction (left image) and steering_angle=steering_center-correction (right image)
- Flip the images and measure the steering angle as follows, steering_angle=(steering_center+correction)*-1

Next step followed in data pre-processing is to shuffle the data. It is very important to shuffle the training data otherwise ordering of data might have huge effect on how the network trends (Neural Network training).

```
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
```

### Model Architecture and Training Strategy
As a first step, I chose LeNet architecture to train the model. LeNet architecture takes 32x32x1 image. But in this case we have 160x320x3 image shape. However, Convolution NN work with wide range of input images. As this is a very powerful NN, I trained it over 5 epochs. The loss continuosly started decreasing although very slightly.
As suggested in the course, I used The NVIDIA network architecture to train the model. It consists of a Normalization layer, 5 Convolutional layers followed by 4 fully connected layers.
![alt text][image1]
##### Normalization and Mean Centering
Initially, the data was pre-processed using Normalization and Mean Centering the data. For Normalization, Lamba layer is added. Normalization is done by dividing each element by 255 which is the max value of image pixel. When the image is normalized to range between 0 and 1, mean centering of image is done by subtracting pixels by 0.5.

**Dimensionality**: The number of neurons of each layer in our CNN can be calculated by using below formula,
```
output_height = [(input_height - filter_height + 2 * padding) / vertical_stride] + 1
output_width = [(input_width - filter_width + 2 * padding) / vertical_stride] + 1
output_depth = number of filters
```

#### Model Architecture
```
Lambda Layer: Apply Normalization and Mean Square Centering
              Input shape = 160x320x3

Cropping: Cropping portion of image to impove model accuracy
          Input shape = 160x320x3, cropping=160-(70+25)=65
          Output shape = 65x320x3
          
Layer 1: Convolutional Layer
         Input shape = 65x320x3, Number of filters = 24, Filter size = 5x5, Vertical Stride=2x2
         Output_height = [(65-5+2*0)/2]+1 = 31.
         Output_width = [(320-5+2*0)/2]+1 = 158.
         Output_depth = filter_depth = 24
         Output shape = 31x158x24
         
Layer 2: Convolutional Layer
         Input shape = 31x158x24, Number of filters = 36, Filter size = 5x5, Vertical Stride=2x2
         Output_height = [(31-5+2*0)/2]+1 = 14.
         Output_width = [(158-5+2*0)/2]+1 = 77.
         Output_depth = filter_depth = 36
         Output shape = 14x77x36

Layer 3: Convolutional Layer
         Input shape = 14x77x36, Number of filters = 48, Filter size = 5x5, Vertical Stride=2x2
         Output_height = [(14-5+2*0)/2]+1 = 5.
         Output_width = [(77-5+2*0)/2]+1 = 37.
         Output_depth = filter_depth = 48
         Output shape = 5x37x48
         
Layer 4: Convolutional Layer
         Input shape = 5x37x48, Number of filters = 64, Filter size = 3x3, Vertical Stride=1x1
         output_height = [(5-3+2*0)/1]+1 = 3.
         output_width = [(37-3+2*0)/1]+1 = 35.
         output_depth = filter_depth = 64
         Output shape = 3x35x64

Layer 5: Convolutional Layer
         Input shape = 3x35x64, Number of filters = 64, Filter size = 3x3, Vertical Stride=1x1
         Output_height = [(3-3+2*0)/1]+1 = 1.
         Output_width = [(35-3+2*0)/1]+1 = 33.
         Output_depth = filter_depth = 64
         Output shape = 1x33x64

Flattening: Flatten the output
            Input shape = 1x33x64, Output shape = 2112

Layer 6: Fully Connected Layer 1
         Input shape = 2112, Output shape = 100

Dropout Layer: Added a dropout layer to avoid overfitting. 25% training data is dropped at this stage.

Layer 7: Fully Connected Layer 2
         Input shape = 100, Output shape = 50

Layer 8: Fully Connected Layer 3
         Input shape = 50, Output shape = 10

Layer 9: Fully Connected Layer 4. This is final layer and it contains only one output bcz this is a regression model.
         The output is the steeing angle.
         Input shape = 10, Output shape = 1
```
##### Hyperparameters
- EPOCH variable is used to tell the TensorFlow how many times to run our training data through the network. More number of EPOCHS results in better model training but it takes a longer time to train the network. The number of epochs chosen = 5.
- 'verbose = 1' parameter that tells Keras to output loss metrics as the model trains.

The model used an adam optimizer, so the learning rate was not tuned manually. Mean Squared Error loss function is used for model compilation as it is best for regression models.
```
model.compile(loss='mse',optimizer='adam')
```
Save the model architecture as 'model.h5'.


```
model.save('model.h5')
```


```
_________________________________________________________________
Layer (type)                 Output Shape              Param =================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         _________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
activation_2 (Activation)    (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
activation_3 (Activation)    (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
activation_4 (Activation)    (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
activation_5 (Activation)    (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
activation_6 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_7 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_8 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```

