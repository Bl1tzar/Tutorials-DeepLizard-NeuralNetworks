from importing_modules import *
from Lesson10_DataPreparation import test_batches, plotImages
from Lesson12_BuildTrainCNN import model

# -- Use of a test set to predict on the model trained --
test_imgs, test_labels = next(test_batches)

print(test_labels)
# plotImages(test_imgs)

# It is very likely that the test labels are the same category, since the test set is not shuffled
# This allows to map directly the unshuffled labels (if it was shuffled, we wouldn't get the correct mapping between the labels and samples)
# We care about the correct mapping because, after getting the predictions, we want to plot our prediction into a confusion matrix

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
# Steps specifies how many batches to yield from the test set before declaring one prediction round complete!

# Print of the predictions (highest probability) for each sample!
predictions_print = np.round(predictions)
print(predictions_print)

# -- Confusion Matrix --
cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
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


test_batches.class_indices
cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
plt.show()