# Training in the dataset created of cats and dogs

from Lesson14_FineTuned_VGG16Model import vgg16_model_sequential
from importing_modules import *
from Lesson10_DataPreparation import train_batches, valid_batches

print('ARROZ!')

vgg16_model_sequential.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print('GRAO!')

vgg16_model_sequential.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

print('FEIJAO!')
