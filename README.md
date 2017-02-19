# Behavioral Cloning

This project is about teaching a car how to control the steering by example. The car will be manually driven and let a deep learning algorithm clone the driving behavior so that it may drive the car by itself.


### Collecting Data

The input data is collected by manually driving the car in the simulator.  The collected data consist of images (center, left, and right images) from 3 cameras facing the road, the applied steering angle, throttle, brake, and current speed.


### Preprocessing

We are trying to predict the steering angle given an image of a road. To make training of the model easier, we will only be including relevant information from our dataset: images (center, left, and right) and steering angle.

The images will be used as our training input, and steering angle will be our target output.

For every set of images (center, left, right), there is only 1 corresponding steering angle. This angle is assigned to the center image, while the steering angles for left and right images are synthesized by adding 6.25 degrees for the left image and subtracting 6.25 degrees for the right image. ± 6.25 degrees were chosen to simulate the steering wheel as if it is moving towards the center.

To add more data, the images are flipped horizontally and the sing of the corresponding steering angle is reversed. This doubles the data and the same time reduces biases towards left and right turns.

It would only require the images of the road to train. Unnecessary information is minimized by cropping the images by cropping 60 pixels from the top, to remove the anything above the horizon, and 20 pixels from the bottom, to remove the visible dashboard. The image is then resized to 32x32 pixels to be compatible with the model.

See ![img_22.jpg](https://github.com/elbernante/behavioral-cloning/blob/master/images/img_22.jpg "img_22.jpg") and ![img_23.jpg](https://github.com/elbernante/behavioral-cloning/blob/master/images/img_23.jpg "img_23.jpg") for samples of processed images.

During training, the dataset will be 0 centered and normalized to have a range between -1 and 1. This will help the learning process to converge better.

The dataset was then split into 90% training set and 10% validation set. To test the performance of the model, the simulator is used in autonomous mode using the model.


### Model Architecture


The model is loosely based on NVIDIA architecture as described on this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). However, since the model is expected to run only in a very limited simulator environment, the architecture was stripped down dramatically as small and as simple as possible. This removes the unnecessary complexity and huge size of the model. It also makes the model easier to train given its smaller size.

The model consists of 2 convolutional layers and 2 dense layers. It accepts 32x32 images with 3 channels as input.

The first convolutional layer uses 8 5x5 filters followed by 2x2 max-pooling layer.
The second convolutional layer uses 16 3x3 filters followed by 2x2 max-pooling layer.

The output of the second convolution layer is flattened before being fed to dense layers, which consists of 256 and 128 hidden units. 

All convolutional and dense layers use RELU activations.

To avoid overfitting, dropout layers are introduced in the second convolutional layer, as well as in the 2 dense layers.

The output layer is a dense layer with linear activation, which would spit out the predicted steering angle.


### Model Training

Images for training were collected as described in “Collecting Data” section. Additionally, more images were collected that simulates when the car runs off track and adjust the steering angle accordingly to recover to the middle lane. This is done so by deliberately making the car go off-lane and then going back to the middle of the lane. However, only the images when the car was recovering to be back on track were recorded. The part were the car was running off-track were not recorded. This is done so to prevent the model from learning to wander off track, but allow the model to learn on how to recover back on-track.

Some of the recorded images can be seen here:

![center_2016_12_01_13_32_52_350.jpg](https://github.com/elbernante/behavioral-cloning/blob/master/images/center_2016_12_01_13_32_52_350.jpg "Sample Image 1")
![center_2016_12_01_13_34_20_804.jpg](https://github.com/elbernante/behavioral-cloning/blob/master/images/center_2016_12_01_13_34_20_804.jpg "Sample Image 2")

The images were then preprocessed to be compatible with our model (See “Preprocessing” section for more details). Some of the preprocessed images can be seen here:

![img_22.jpg](https://github.com/elbernante/behavioral-cloning/blob/master/images/img_22.jpg "Sample Preprocessed Image 1")
![img_23.jpg](https://github.com/elbernante/behavioral-cloning/blob/master/images/img_23.jpg "Sample Preprocessed Image 2")


Since this is a regression problem, mean squared error was chosen as the loss function.

The number of layers and filter size where chosen from bigger sizes and gradually reducing over several tryouts.

The images were read from disk on the fly using python generator. This helps conserving memory and use only what is needed at a given time.

The batch size of 128 was chosen in order for the training batch to fit comfortably in the memory.

Adam optimizer was used. Initially, the default 0.001 learning rate was used but found out that the loss dropped too rapidly and the car never seemed to drive properly. It was then reduced to 0.0001 and turned out to be a good learning rate for this model.

The model was tested in autonomous mode in the simulator after every epoch. After the 5th epoch, there seems to be no significant improvement of the driving behavior of the car.
