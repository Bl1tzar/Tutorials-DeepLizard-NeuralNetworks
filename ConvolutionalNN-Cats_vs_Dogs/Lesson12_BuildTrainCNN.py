from importing_modules import *
from Lesson10_DataPreparation import train_batches, valid_batches, plotImages

# -- CNN using the Keras sequential model --
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

# 1st hidden Layer (with an implicit input layer): Conv2D layer - standard convolutional layer that will accept image data
## filters = 32 is arbitrary with a common size of 3 by 3 for images without padding (no change in size)
# 2nd hidden layer: MaxPool2D - Pooling action
## In this case, it will cut the size of the images in half
# 3rd hidden layer: Conv2D - it is common to increase the number of filters as we go in the number of layers
# 5th hidden layer: Flatten () - flatten all of it into a 1D tensor before passing it to the dense output layer
# 6th output layer: output 2 nodes (cat and dog)

# Network architecture
# model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# -- Training the CNN --

model.fit(x=train_batches,  # Since data is stored as a generator (with the labels), there is no y for the target data
          steps_per_epoch=len(train_batches),  # indication of how many batches of samples from the training set should
          # be passed to the model before declaring one epoch complete
          # steps_per_epoch = (nº of samples from the data set/batch size)
          validation_data=valid_batches,
          validation_steps=len(valid_batches),  # the same for steps_per_epoch is applied here for the validation batches
          epochs=10,
          verbose=2
          )

# NOTE: THIS MODEL HAS OVERFITTING (COMPARE THE ACCURACY WITH THE VALIDATION ACCURACY)
# To solve this problem, we need to be doing fine tuning.