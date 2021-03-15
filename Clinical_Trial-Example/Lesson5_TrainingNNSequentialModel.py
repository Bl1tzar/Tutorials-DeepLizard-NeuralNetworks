from Lesson4_CreatingNNSequentialModel import model
from Lesson3_DataProcessing import scaled_train_samples, train_labels

# Used not when building the module, rather when training it
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# Function "compile" to prepare the model for training. Get's everything in order that's needed before training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training of the model with "fit" function

model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2)

# batch_size = number of samples per gradient update
# epochs = the model is going to process/train on all of the data in the dataset 30 times before completing the total
# training process
# shuffle = default is true - the data is being shuffled when we pass it to the network - GOOD! no order in the data
# verbose = type of output to let us see when we run the fit function
# See more for the definition of each parameter in TensorFlow API-tf.keras-sequential


