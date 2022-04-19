# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import Dropout
classifier = Sequential()
#
# classifier.add(TimeDistributed(Convolution2D(32, (3,3), padding='same', strides=2), input_shape=(None, 64, 64, 1)))
# classifier.add(Activation('relu'))
# classifier.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
# classifier.add(TimeDistributed(Convolution2D(64, (3, 3), padding='same', strides = 2)))
# classifier.add(Activation('relu'))
# classifier.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
# classifier.add(TimeDistributed(Convolution2D(64, (3, 3), padding='same', strides = 2)))
# classifier.add(Activation('relu'))
# classifier.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
# classifier.add(TimeDistributed(Convolution2D(128, (3, 3), padding='same')))
# classifier.add(Activation('relu'))
# classifier.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
#
#
# classifier.add(TimeDistributed(Flatten()))
# classifier.add(LSTM(512, return_sequences=True))
# classifier.add(LSTM(512))
# classifier.add(Dense(128))
# classifier.add(Dense(3))
#
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#

classifier.add(TimeDistributed(Convolution2D(32, (3,3), strides=(4,4), activation='relu'), input_shape=(None, 64, 64, 1)))
classifier.add(TimeDistributed(MaxPooling2D((3, 3), strides = (2,2))))
classifier.add(TimeDistributed(BatchNormalization()))
classifier.add(TimeDistributed(ZeroPadding2D(padding = (2,2))))
classifier.add(TimeDistributed(Convolution2D(128, (5, 5), activation='relu')))
classifier.add(TimeDistributed(MaxPooling2D((3, 3), strides = (2,2))))
classifier.add(TimeDistributed(BatchNormalization()))
classifier.add(TimeDistributed(ZeroPadding2D(padding = (1,1))))

classifier.add(TimeDistributed(Convolution2D(384, (3, 3), activation='relu')))
classifier.add(TimeDistributed(ZeroPadding2D(padding = (1,1))))

#model.add(layers.MaxPooling2D((2, 2)))
classifier.add(TimeDistributed(Convolution2D(192, (3, 3), activation='relu')))
classifier.add(TimeDistributed(ZeroPadding2D(padding = (1,1))))

classifier.add(TimeDistributed(Convolution2D(128, (3, 3), activation='relu')))
classifier.add(TimeDistributed(MaxPooling2D((3, 3), strides = (2,2))))

#model.add(layers.MaxPooling2D((2, 2)))
classifier.add(TimeDistributed(Flatten()))
classifier.add(TimeDistributed(Dense(4096, activation='relu')))

classifier.add(LSTM(256, return_sequences = True))

classifier.add(TimeDistributed(Dense(4096, activation='relu')))
classifier.add(TimeDistributed(Dropout(0.5)))

classifier.add(LSTM(256, return_sequences = True))

classifier.add(Dense(101, activation='relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Step 2 - Preparing the train/test data and training the model

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./64,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./64)

training_set = train_datagen.flow_from_directory('train_frames_rs',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('test_frames_rs',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical')
classifier.fit(
        training_set,
        steps_per_epoch=800, # No of images in training set
        epochs=5,
        validation_data=training_set,
        validation_steps=30
        )# No of images in test set


# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')
