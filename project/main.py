from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,  Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tkinter as tk
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dense
from keras.preprocessing.image import load_img, img_to_array
import os

# ZAMAN KONTROL FONKSİYONLARI
start_time = 0
def start_timer(process_name='Process'):
    print(process_name + ' Started')
    start_time = time()

def stop_timer(process_name='Process'):
    print(process_name + " Finished ({:.2f} seconds)".format(round((time() - start_time), 2)))

# DEFINE
SEED = 10
EPOCHS = 15
RANGE_EPOCH = range(1, EPOCHS + 1)

BATCH_SIZE = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)

# VERİSETİ KONUMU
DATASET_DIR = r"E:\finished_projects\machine_learning\samed_enes\preprocessed_data"
TEST_DIR = DATASET_DIR + "\\test\\"
TRAIN_DIR = DATASET_DIR + "\\train\\"
VALIDATION_DIR = DATASET_DIR + "\\val\\"

start_timer('Dataset Importing')
# TRAIN DATA IMPORT
train_data_generator = ImageDataGenerator(
    validation_split = 0.2,
    preprocessing_function = preprocess_input
)
train_data = train_data_generator.flow_from_directory(
    TRAIN_DIR,
    target_size = IMG_SIZE,
    shuffle = True,
    seed = SEED,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    batch_size = BATCH_SIZE,
    subset='training'
)

# TRAIN DATA IMPORT
val_data_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    validation_split = 0.2
)
valid_data = val_data_generator.flow_from_directory(
    VALIDATION_DIR,
    target_size = IMG_SIZE,
    shuffle = False,
    seed = SEED,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    batch_size = BATCH_SIZE,
    subset = 'validation'
)

# TEST DATA IMPORT
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = test_generator.flow_from_directory(
    TEST_DIR,
    target_size = IMG_SIZE,
    shuffle = False,
    seed = SEED,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    batch_size = BATCH_SIZE
)
stop_timer('Dataset Importing')

classes = list(train_data.class_indices.keys())
num_classes = len(classes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=IMG_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = Adam(),
    metrics = ['accuracy']
)

start_timer('Training')
hist = model.fit( 
    train_data,
    steps_per_epoch = train_data.samples // BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = valid_data,
    verbose = 1,
    validation_steps = valid_data.samples // BATCH_SIZE
)
stop_timer('Training')

accuracy = np.array(hist.history['accuracy'])
val_accuracy = np.array(hist.history['val_accuracy'])
loss = np.array(hist.history['loss'])
val_loss = np.array(hist.history['val_loss'])

plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.plot(RANGE_EPOCH, accuracy, label='Training Accuracy')
plt.plot(RANGE_EPOCH,val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(RANGE_EPOCH, loss, label='Training Loss')
plt.plot(RANGE_EPOCH, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

def recogout():
    root=tk.Tk()
    root.withdraw()
    #img_path = filedialog.askopenfilename()
    image_paths = os.listdir(TEST_DIR)
    for image_path in image_paths:
        img=load_img(os.path.join(TEST_DIR, image_path), color_mode = 'grayscale', target_size=IMG_SIZE)
        img_array=img_to_array(img)
        img_array=tf.expand_dims(img_array, 0)
        predictions=model.predict(img_array)
        score=tf.nn.softmax(predictions[0])
        print(image_path + " This image most likely belongs to {}"
            .format(classes[np.argmax(score)]))

recogout()