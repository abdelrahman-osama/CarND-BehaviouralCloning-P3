# **Behavioral Cloning** 


---

**Behavioral Cloning Project**


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/right.jpg "Recovery Image"
[image4]: ./examples/left.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used nVidia model architecture witht the following parameters:
```
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```
#### 2. Attempts to reduce overfitting in the model

I used early stopping to reduce overfitting.The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I collected a big dataset that contains 8 laps, 4 of them in the reverse direction. 

I made some cropping on data to remove data that might create distortion to the model such as the upper part of the image.
I also added a correction factor of 0.2 to the steering angle for the left and right images, which made the model drive smoother and I made sure to shuffle the data to make sure that the model does not overfit or memorize the images in order.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first thing I thought of was to search for a good model which can run on my current hardware. I found nVidia model as it is known to be one of the best models, I also thought about GoogLeNet but I think it was too complex and heavy for this project. After that I started thinking about the data generation and processing. I flipped the right and left images and added them to the dataset and I added a correction angle for them but I am not really sure if it affected the model or not. What I believe made difference and helped me achieve a good final result was two things, changing the input dataset color channel from BGR to RGB and using a bigger dataset that I made.

#### 2. Final Model Architecture

The final model architecture that was used was the nVidia model with 5 convolutional layers and 4 dense layers, I did cropping to images to make sure that unimportant data are not distracting the model output.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. And other four laps on track one but in reverse. Here are example images from the dataset:

![alt text][image2]

To augment the data set, I also flipped images and angles thinking that this would prevent the model from driving on the right lane all the time as the right turns are more than the left turns so the data is unbalanced.

After the collection process, I had about 13K data points. I then preprocessed this data by changing the color channel from BGR to RGB, flipping the right and left images and finally adding the correction factor.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used early stopping to prevent overfitting the model. I also used an adam optimizer so that manually training the learning rate wasn't necessary.
