#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam #optimizer is use to minimize the loss function
from keras.callbacks import ModelCheckpoint #When training deep learning models, the checkpoint is the weights of the model.                                            
import matplotlib.pyplot as plt                #These weights can be used to make predictions


# In[3]:


# this is the augmentation configuration we will use for training data
# It generate more images using below parameters
#Rescale : One of many augmentation parameters, adjusts the pixel values of our image,
#Setting rescale=1./255 will adjust our pixel values to be between 0–1.
training_datagen = ImageDataGenerator(rescale=1./255, 
                                      rotation_range=40, #Int, Degree range for random rotations
                                      width_shift_range=0.2, #its 20 % width shift range
                                      height_shift_range=0.2,
                                      shear_range=0.2, #Shear angle in counter-clockwise direction in degrees)
                                      zoom_range=0.2,
                                      horizontal_flip=True, #Boolean. Randomly flip inputs horizontally
                                      fill_mode='nearest') #One of {"constant", "nearest", "reflect" or
                                                           # "wrap"}. Default is 'nearest'. 
                                                           


# In[4]:


# this is a generator that will read pictures found 
#at train_data_path, and indefinitely generate batches of augmented image data.
training_data = training_datagen.flow_from_directory('data/train/', # this is the target directory
                                      target_size=(150, 150), #all images will be resized to 150x150
                                      batch_size=32, #Number of images to include in each batch of training data 
                                      class_mode='binary')  # since we use binary_crossentropy loss, 
                                                            #we need binary labels.
 
training_data.class_indices


# In[5]:


# this is the augmentation configuration we will use for validation:
# here, we have done only rescaling. 
valid_datagen = ImageDataGenerator(rescale=1./255)
 
# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory('data/val/',
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')


# In[6]:


def plotImages(images_arr): #then we created a function for plotting our argumented images
    fig, axes = plt.subplots(1, 5, figsize=(20, 20)) #here we're plotting 5 images of size 20x20
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# In[7]:


# showing augmented images
images = [training_data[0][0][0] for i in range(5)] #in this we can see some diseased/non diseased cotton leaf images
plotImages(images) #calling our function that we have created earlier 


# In[8]:


# save best model using vall accuracy
model_path = 'model/v6_pred_cott_dis.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[9]:



#Building cnn model
cnn_model = keras.models.Sequential([
                                    keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150, 150, 3]),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(filters=64, kernel_size=3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),
                                    keras.layers.Conv2D(filters=128, kernel_size=3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),                                    
                                    keras.layers.Conv2D(filters=256, kernel_size=3),
                                    keras.layers.MaxPooling2D(pool_size=(2,2)),

                                    keras.layers.Dropout(0.5),                                                                        
                                    keras.layers.Flatten(), # neural network beulding
                                    keras.layers.Dense(units=128, activation='relu'), # input layers
                                    keras.layers.Dropout(0.1),                                    
                                    keras.layers.Dense(units=256, activation='relu'),                                    
                                    keras.layers.Dropout(0.25),                                    
                                    keras.layers.Dense(units=4, activation='softmax') # output layer
])


# compile cnn model
cnn_model.compile(optimizer = Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[10]:


cnn_model.summary()


# In[11]:


# train cnn model
history = cnn_model.fit(training_data, 
                          epochs=100,  
                          verbose=1, 
                          validation_data= valid_data,
                          callbacks=callbacks_list)


# In[14]:


history.history
