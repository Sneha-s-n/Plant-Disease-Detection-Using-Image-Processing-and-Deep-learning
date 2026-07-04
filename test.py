import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

# ====================================
# DATASET PATHS
# ====================================

TRAIN_DIR = r'dataset/Train'
VALID_DIR = r'dataset/valid'
TEST_DIR = r'dataset/test'

# ====================================
# IMAGE SETTINGS
# ====================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

# ====================================
# DATA PREPROCESSING
# ====================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ====================================
# LOAD DATASET
# ====================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(

    VALID_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical',

    shuffle=False
)

# ====================================
# CLASSES
# ====================================

num_classes = len(train_generator.class_indices)

print("Classes Found:")

print(train_generator.class_indices)

# ====================================
# TRANSFER LEARNING MODEL
# ====================================

base_model = MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)
)

base_model.trainable = False


inputs = Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.5)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# ====================================
# COMPILE MODEL
# ====================================

model.compile(

    optimizer=Adam(learning_rate=0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# ====================================
# MODEL SUMMARY
# ====================================

model.summary()


# ====================================
# TRAIN MODEL
# ====================================

history = model.fit(

    train_generator,

    epochs=8,

    validation_data=valid_generator
)
# ====================================
# CREATE FOLDERS
# ====================================

os.makedirs('model', exist_ok=True)

os.makedirs('static/images', exist_ok=True)

# ====================================
# SAVE MODEL
# ====================================

model.save('model/plant_disease_model.h5')

print("Model Saved Successfully")

# ====================================
# EVALUATE MODEL
# ====================================

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Accuracy:", test_accuracy)

# ====================================
# PREDICTIONS
# ====================================

predictions = model.predict(test_generator)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

# ====================================
# CONFUSION MATRIX
# ====================================

conf_matrix = confusion_matrix(

    true_classes,

    predicted_classes
)

plt.figure(figsize=(10, 8))

sns.heatmap(

    conf_matrix,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=class_labels,

    yticklabels=class_labels
)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

# ====================================
# CLASSIFICATION REPORT
# ====================================

print(classification_report(

    true_classes,

    predicted_classes,

    target_names=class_labels
))

# ====================================
# ACCURACY GRAPH
# ====================================

plt.figure(figsize=(12, 5))

# Accuracy

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

# Loss

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

# ====================================
# SAVE GRAPH
# ====================================

plt.savefig('static/images/graph.png')

plt.close()