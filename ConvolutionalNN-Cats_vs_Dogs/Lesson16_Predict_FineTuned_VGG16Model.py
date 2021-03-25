from importing_modules import *
from Lesson10_DataPreparation import test_batches
from Lesson15_Train_FineTuned_VGG16Model import vgg16_model_sequential
from Lesson13_TestSet_theModel import plot_confusion_matrix

print('ATUM!')

predict = vgg16_model_sequential.predict(x=test_batches, verbose=0)

print('CEBOLA!')

#test_batches.classes



#cmatrix = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predict, axis=-1))



#test_batches.class_indices

#cmatrix_plot_labels = ['cat', 'dog']



#plot_confusion_matrix(cm=cmatrix, classes=cmatrix_plot_labels, title='Confusion Matrix :)')


