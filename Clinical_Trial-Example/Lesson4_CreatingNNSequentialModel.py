# Tensorflow modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense


print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Sequential model - most simple, basically a linear stack of layers, as shown:
model = Sequential([  # Variable "model"-List of layers
    # The 1st layer is the input data. The input data is the layer itself
    Dense(units=16, input_shape=(1,), activation='relu'), # 2nd layer - 1st hidden layer - 16 nodes/neurons
    Dense(units=32, activation='relu'),  # Dense layer=Fully connected layer - relu: activation function
    Dense(units=2, activation='softmax')  # Output of the probabilities for each class - 2 nodes because there are only
    # two possible outputs: experience side-effects (1) or not (0) - softmax: activation function for the output layer,
    # so a classifier
])

model.summary()  # visual summary of the architecture of the model created above

