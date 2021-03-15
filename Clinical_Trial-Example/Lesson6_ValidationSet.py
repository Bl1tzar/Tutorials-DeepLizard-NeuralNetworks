from Lesson4_CreatingNNSequentialModel import model
from Lesson3_DataProcessing import scaled_train_samples, train_labels

# Used not when building the module, rather when training it
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# Function "compile" to prepare the model for training. Get's everything in order that's needed before training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# For the Validation set, we add the parameter "validation_split"

model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True,
          verbose=2)

# validation_split = splits the last portion (percentage) of the training set into a validation set
# the data we split is no longer in the training data!
# NOTE! The data is split into the validation set before the data is shuffled, even if the se state of "shuffle=true"
# NOTE! In this example, the data was shuffled before passing the fit function

# The validation set is helpful to confirm that the model is not overfiting (aka the model is not only useful to one
# set of data, but generic)
