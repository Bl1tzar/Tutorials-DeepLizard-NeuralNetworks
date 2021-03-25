# For more about Fine-tuning: https://deeplizard.com/learn/video/5T-iXNNiwIs

# -- Building a Fine-tuned VGG16 model --

from importing_modules import *

# Download the model with weights - For the first time, it will download it. For the rest, it will use the copy downloaded
# From: https://github.com/fchollet/deep-learning-models/releases
vgg16_model = tf.keras.applications.vgg16.VGG16()

# vgg16_model.summary()

# The model was trained to output 1000 different predictions. Following the objective, we only want 2 (cats and dogs)

# We also want to work, for now, with a sequential model. The VGG16 is a "Model" model. The "Model" is a model from the Keras functional API
# So, we will convert the original VGG16 model into a sequential model. For that, we will create a new variable called model and setting it equal to a sequential object
# We will loop into each layer from the VGG16 model (except the last one, the output) and adding it to the new sequential model
vgg16_model_sequential = Sequential()
for layer in vgg16_model.layers[:-1]:
    vgg16_model_sequential.add(layer)

# vgg16_model_sequential.summary()

# We define each layer as not being trainable. This means that it will freeze the weights and bias of each layer so we used the trained vgg16 model
for layer in vgg16_model_sequential.layers:
    layer.trainable = False

# Defining the output layer to 2 predictions and which will be the only one trainable
vgg16_model_sequential.add(Dense(units=2, activation='softmax'))

# Only 8194 parameters trainable, which are from the output layer created. If the other layers weren't frozen, we would train 134.268.738 parameters!
vgg16_model_sequential.summary()