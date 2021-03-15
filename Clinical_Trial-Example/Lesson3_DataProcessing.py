import numpy as np  # library from numpy
from random import randint  # library from Python - function randint
from sklearn.utils import shuffle  # library from scikit-learn - function shuffle
from sklearn.preprocessing import MinMaxScaler  # library from scikit-learn - Class MinMaxScaler

train_samples = []  # Variable - list for the training labels (input)
train_labels = []  # Variable - list for the training labels (output)

# dummy data - not real, just as an example - numerical data processing
for i in range(50):  # 100 (50x2) of the 2100 total (5%)
    # The ~5% of YOUNGER individuals who did experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)  # 1 means they experienced side-effects

    # The ~5% of OLDER individuals who did not experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)  # 0 means they didn't Experience side-effects

for i in range(1000):  # 2000 (1000x2) of the 2100 total (95%)
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# for i in train_samples:  # test print of the list "train_samples" created in the for loop above
#    print(i)

# for i in train_labels:
#    print(i)

# Converting the lists into numpy array's from the fit function from TF (see documentation)
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# Shuffle the labels and samples to get rid of any order that was imposed on the dataset during the creation process
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0, 1))  # Specifying the range - Scale down the data from [13, 100] to [0, 1]
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))  # fit_transform function - Transform the
# training samples from 13-100 to 0-1. Reshape (1, -1) function is just a formality

for i in scaled_train_samples:
    print(i)
