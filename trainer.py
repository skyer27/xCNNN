import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

# Constants
EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
MODELS_DIR = "models"
TRAINING_DIR = "dataset/train"
MODEL_NAME = "models/cnn.vgg16.keras"

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# Data augmentation and normalization
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
dataset = generator.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define number of classes
NUM_CLASSES = len(dataset.class_indices)

# Model architecture
model = Sequential()

# In a Convolutional Layer, a NxM kernel slides to each possible location in the 2D input.
# The kernel (of weights) performs an element-wise multiplication and summing all values into one.
# The N-kernels will generate N-maps with unique features extracted from the input image.
# Kernel Sizes: 3x3 (common), 5x5 (suitable for small features), 7x7 or 9x9 (appropriate for larger features)

# Rectified Linear Unit ReLU(x) = max(0, x)
# Any negative value becomes zero, addressing the gradients/derivatives
#   from becoming very small and providing less effective learning
# ReLU sets all negative values in the feature maps to zero
#   introducing non-linearity to help in learning complex patterns and relationships
    
# Batch Normalization helps accelerate and stabilize the training process
#   by normalizing the activation after the Convolutional Layer.
# Each feature map is independently normalized.
    
# Max Pooling is used to downsample and reducing spatial dimensions of feature maps.
# It divides the feature map into non-overlapping regions and chooses the maximum value for each.
# Simply, it looks for the most important parts and reduces the data size for improved processing.
# Larger pool size may lead to less detailed feature maps.
# Pool Size: 2x2 and 3x3 (most common)

model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())


# The complex feature maps must be flattened
#   before feeding to the Dense layers, since it only accepts 1D arrays.
model.add(Flatten())

# Dense is a layer where each neuron is fully connected to the previous layer.
# It means that each neuron accepts the full output of the previous layer.
model.add(Dense(4096, activation="relu"))
model.add(Dense(4096, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Dropout is typically applied after the fully connected layers.
# Value range from 0.2-0.5, with 0.5 as ideal to avoid overfitting in smaller datasets.

# Output Layer
# Sigmoid and SoftMax are applicable, but sigmoid is prefered for binary classifications.
model.add(Dense(NUM_CLASSES, activation="softmax"))

# Compile the model
# AdaM / Adaptive Moment Estimation
# AdaM tends to reach an optimal solution faster.
# Binary Cross-Entropy is more optimal for Binary Classification
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
# An epoch is the full iteration on a dataset
# There is no hard rules, but early stopping is essential
#   especially when reaching desired accuracy or
#   when it is deteriorating.
# An example value would be 100/200/1000, but must be changed
#   based on the dataset, HW resources, actual dataset, etc.
model.fit(dataset, epochs=EPOCHS)

# Save the model
model.save(MODEL_NAME)