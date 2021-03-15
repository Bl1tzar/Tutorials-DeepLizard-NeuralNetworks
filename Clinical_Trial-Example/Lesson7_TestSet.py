# We want the model to be generic, so if we train it in one data set, we want it to be able to be used in random data
# Since our model as already been trained and validated, we now want to test it! -> This is called inference


import numpy as np  # library from numpy
from random import randint  # library from Python - function randint
from sklearn.utils import shuffle  # library from scikit-learn - function shuffle
from sklearn.preprocessing import MinMaxScaler  # library from scikit-learn - Class MinMaxScaler

from Lesson5_TrainingNNSequentialModel import model
from Lesson3_DataProcessing import scaler

# ---CREATION OF THE TEST DATA---

test_labels = []
test_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

# scaler = MinMaxScaler(feature_range=(0, 1))  # Specifying the range - Scale down the data from [13, 100] to [0, 1]
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

# ---USING THE TRAINED MODEL CREATED BEFORE AND PREDICT ON THE TEST DATA---

# Evaluating the test set
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

# Print out two columns: they contain the probabilities for each possible output - experienced or didn't experience# side effects

#for i in predictions:
#    print(i)

# Know which output has the highest probability and print it
rounded_predictions = np.argmax(predictions, axis=-1)
#for i in rounded_predictions:
#    print(i)