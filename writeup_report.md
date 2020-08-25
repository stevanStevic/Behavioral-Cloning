# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_driving.jpg "Center Driving"
[image3]: ./examples/steeer_to_edge.jpg "Recovery Image 1"
[image4]: ./examples/close_to_edge.jpg "Recovery Image 2"
[image5]: ./examples/steer_away2.jpg "Recovery Image 3"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"
[image8]: ./examples/neural_network.png "Network Summary"
[image9]: ./examples/training_loss_25-08-2020-20-33-40.png "Network Summary"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

Example of driving the first track can be see in `video.mp4`.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 57-73) 

Beside ELU the model includes RELU layers to introduce nonlinearity (code line 58 and 60), and the data is normalized in the model using a Keras lambda layer (code line 54). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 63, 66, 69, 72). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving laps in both directions.
I've put emphasis on recovery maneuvers to try and keep vehicle on track better.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reuse existing network architecture.

As suggested in the lecures I first implemented LeNet, and tried loading and preprocessing images, to confirm that this part works.

Preprocessing of the image includes croping out the part of the image that is not relevant, like sky and hood. Next is converting to YUV color space and bluring the image to smooth it out.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set where 20% of data was used for validation. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I used model implemented by NVIDIA for self driving car. It is explained [here](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) in more details. I thought this model might be appropriate because it used in real vehicles and could prorably handle simulator well.

For this I also added resizing of the image to fit default image size for this nerual network. I've also added Lambda layer to normalize the image.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.

To improve the driving behavior in these cases, I created generator that would load and generate data in batches. This also incldued shuffling of input data in order to have random training data sequence so that model would generlize better. For this I also included augmentation of image by randomly flipping it over Y axis.

I loaded all recorded images (center, left and right) but adjusted steering angles of the side images so that angle increases in order to combat recovery maneuvers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 53-72) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture

![alt text][image1]

And here is the summary of the neural network 

![alt text][image8]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on track and steer on curves. These images show what a recovery looks like where I steer towards the edge of the track, and then hard turn to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images over Y axis and angles thinking that this would help nerual network to generalize better. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Image above is also preprocessed with croped unrelevant parts and blured with Gaussian filter.

After the collection process, I had ~30k samples. This includes center, left and right images. Since I loaded all of them during training I count all of them

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs for me was 15 as evidenced by image below

![alt text][image9]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Possible improvements

Possible improvements would include additional methods for image augmentation, like resizing, interpolating, using custom parameters, changing brightness and etc.
Might be good to experiment more with epchos and batch sizes. I tried only few combinations.
Changing activation functions in neural network could provide some benfits.
Of course using differnet architecture at all and many other.



