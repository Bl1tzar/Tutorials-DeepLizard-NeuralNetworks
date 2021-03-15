# The confusion matrix is used to visualize prediction results from a neural network during inference
# Basically, a visualization of the prediction of the model on test data
import matplotlib
from Lesson7_TestSet import rounded_predictions, test_labels, np

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# Using scikit-learn to create the confusion matrix
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)


# Function directly taken from the scikit-learn website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Labels for the confusion matrix
cm_plot_labels = ['no_side_effects', 'had_side_effects']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
plt.show()
# NOTE! Light Blue - Number of incorrect predictions | Dark blue - Number of correct predictions