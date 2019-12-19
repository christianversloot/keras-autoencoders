import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dropout
import matplotlib.pyplot as plt
import numpy as np
from vis.utils import utils
from vis.visualization import visualize_activation, get_num_filters
from vis.input_modifiers import Jitter
import random
import math
import os.path

# Model configuration
img_width, img_height = 28, 28
batch_size = 1000
no_epochs = 15
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Reshape data
input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Create the model
model = Sequential()
model.add(Conv2D(56, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape, name='conv_layer'))
model.add(Dropout(0.5))
model.add(Conv2DTranspose(56, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', name='conv_t_layer'))
model.add(Dropout(0.5))
model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

# Compile and fit data
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(input_train, input_train,
                epochs=no_epochs,
                batch_size=batch_size,
                validation_split=validation_split)

# Define the folder name to save into
folder_name = 'filter_visualizations'

# Iterate over multiple layers
for layer_nm in ['conv_layer', 'conv_t_layer']:

  # Find the particular layer
  layer_idx = utils.find_layer_idx(model, layer_nm)

  # Get the number of filters in this layer
  num_filters = get_num_filters(model.layers[layer_idx])

  # Draw filters randomly
  rows = 10
  cols = 5
  total = rows * cols
  drawn_filters = random.choices(np.arange(num_filters), k=total)

  # Visualize each filter
  plots = []
  for filter_id in drawn_filters:
    img = visualize_activation(model, layer_idx, filter_indices=filter_id, input_modifiers=[Jitter(16)])[:, :, 0]
    plots.append(img)
    print(f'Saved layer {layer_nm}/{filter_id} to file!')

  # Matplotlib plotting
  fig, axes = plt.subplots(rows, cols)
  for i in range(0, len(plots)):
    row = math.floor(i/cols)
    col = i % cols
    print(f'Row/col {row},{col}')
    axes[row, col].set_facecolor('white')
    axes[row, col].imshow(plots[i], cmap='gray')
  fig.suptitle(f'Filter visualizations, Dropout, layer  = {layer_nm}')
  plt.show()