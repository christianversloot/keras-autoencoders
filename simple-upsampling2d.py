import keras
from keras.models import Sequential
from keras.layers import UpSampling2D
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
input_flattened = np.arange(0, 2, 0.5)
input_image = np.reshape(input_flattened, (2, 2, 1))
input_image_shape = np.shape(input_image)
input_image_shape = (input_image_shape[0], input_image_shape[1], 1)

# Create the model
model = Sequential()
model.add(UpSampling2D((32, 32), input_shape=input_image_shape, interpolation='bilinear'))
model.summary()

# Perform upsampling
model_inputs = np.array([input_image])
outputs_upsampled = model.predict(model_inputs)

# Get output
output_upsampled = outputs_upsampled[0]

# Visualize input and output
fig, axes = plt.subplots(1, 2)
axes[0].imshow(input_image[:, :, 0]) 
axes[0].set_title('Original image')
axes[1].imshow(output_upsampled[:, :, 0])
axes[1].set_title('Upsampled input')
fig.suptitle(f'Original and upsampled input, method = bilinear interpolation')
plt.show()