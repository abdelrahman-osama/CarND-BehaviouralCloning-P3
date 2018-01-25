import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

lines = []
with open('./t1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
data_train, data_validation = train_test_split(lines, test_size=0.2)

def data_generator(samples, batch_size = 32):
    no_samples = len(samples)

    while (1):
        shuffle(samples)

        for j in range(0, no_samples, batch_size):
            batch = samples[j:j+batch_size]
            images = []
            measurements = []
            for b in batch:
                for i in range(3):
                    source_path = b[i]
                    filename = source_path.split('/')[-1]
                    current_path = './t1/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    images.append(cv2.flip(image, 1))
                    measurement = float(b[3])
                    if(i == 1): #left image
                        measurement+=0.2
                    if(i == 2): #right image
                        measurement-=0.2
                    measurements.append(measurement)
                    measurements.append(measurement*-1.0)
        X_train = np.array(images)
        y_train = np.array(measurements)
        yield shuffle(X_train, y_train)

train_generator = data_generator(data_train)
valid_generator = data_generator(data_validation)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers import Input, Dropout, BatchNormalization, Cropping2D
from keras.models import Model
from keras.callbacks import EarlyStopping



model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
#model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0001,
                                   patience=0,
                                   verbose=0,
                                   mode='auto')
model.fit_generator(train_generator, samples_per_epoch = len(data_train),validation_data=valid_generator, nb_val_samples=len(data_validation), nb_epoch=10, callbacks=[early_stop])
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save('model.h5')