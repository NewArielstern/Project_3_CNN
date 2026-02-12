import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    GlobalAveragePooling2D,
    Dense, Dropout, Input,Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

ZIP_PATH = r"C:\Users\USER\Desktop\Study\ML_True\Project_3_CNN_2\emotions_dataset.zip"
EXTRACT_PATH = r"C:\Users\USER\Desktop\Study\ML_True\Project_3_CNN_2\dataset"

if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
        print('Connected Successfully')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    os.path.join(EXTRACT_PATH, "train"),
    target_size=(64,64),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb'
)

test_set = test_datagen.flow_from_directory(
    os.path.join(EXTRACT_PATH, "test"),
    target_size=(64,64),
    batch_size=64,
    class_mode='categorical',
    color_mode='rgb'
)
print(training_set[0][0].shape)
print(training_set.class_indices)

model = Sequential([
    Input(shape=(64, 64, 3)),

    Conv2D(32, 3, activation='relu', padding='same'),
    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPooling2D(2),

    Conv2D(128, 3, activation='relu', padding='same'),
    MaxPooling2D(2),

    Conv2D(128, 3, activation='relu', padding='same'),

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(5, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0003),loss='categorical_crossentropy',metrics=['accuracy'])


model_path = "emotions_model3_cnn_5.keras"

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    training_set,
    validation_data=test_set,
    epochs=50,
    callbacks = [early_stop]
)


model.summary()
model.save(model_path)
print(f'Model {model_path} Saved Successfully!')

# 378/378 ━━━━━━━━━━━━━━━━━━━━ 144s 382ms/step - accuracy: 0.6610 - loss: 0.8680 - val_accuracy: 0.6591 - val_loss: 0.8602