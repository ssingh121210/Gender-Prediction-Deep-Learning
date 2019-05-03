#Importing required libraries 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
#Setting up the neural network
classifier=Sequential()
#Step-1 Convolution
#Choose 64 as input_shape due to less CPU power you can set it to 128 or 256 based on your requirement.
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#Step-2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
# 2-nd Convolution Layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Step-3 Flattening
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Setting up the training and testing set of image.
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Location of your training set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'Location f Your Test Set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        workers=1,
        use_multiprocessing=False,
        
        steps_per_epoch=300,
        epochs=25,
        validation_data=test_set,
        validation_steps=50)
				
#Predicitng a single image
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('Location of your image ',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    print("MALE")
else:
    print("FEMALE")
