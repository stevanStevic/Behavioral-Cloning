
import cv2
import numpy as np
import tensorflow as tf 
import csv
import math
import matplotlib.pyplot as plt

import image_processing as ip

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D, ELU
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

def generator(samples, batch_size=32, augment=False):
    num_samples = len(samples)

    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images, angles = ip.read_data(batch_samples, augment)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield (X_train, y_train)

batch_size=100
dropouts = 0.35

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size, augment=True)
validation_generator = generator(validation_samples, batch_size=batch_size, augment=False)

# Model definition
model = Sequential()

# Preprocessing
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (66, 200, 3)))

# Model
model.add(Conv2D(24, (5, 5), strides = (2, 2), activation = 'elu', input_shape = (66, 200, 3)))
model.add(Conv2D(36, (5, 5), strides = (2, 2), activation = 'relu'))
model.add(Conv2D(48, (5, 5), strides = (2, 2), activation = 'elu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'elu'))
model.add(Flatten())
model.add(Dropout(dropouts))
model.add(Dense(100))
model.add(ELU())
model.add(Dropout(dropouts))
model.add(Dense(50))
model.add(ELU())
model.add(Dropout(dropouts))
model.add(Dense(10))
model.add(ELU())
model.add(Dropout(dropouts))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,                                              \
            steps_per_epoch = math.ceil(len(train_samples) / batch_size),         \
            validation_data = validation_generator,                               \
            validation_steps = math.ceil(len(validation_samples) / batch_size),   \
            epochs=10,  \
            verbose=1)

model.save('model.h5')

# Plot training data
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')

# Save plot
plt.savefig('docs/training_loss_' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))