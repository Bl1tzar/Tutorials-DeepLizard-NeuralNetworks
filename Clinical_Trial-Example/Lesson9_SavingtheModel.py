from Lesson5_TrainingNNSequentialModel import model

if os.path.isfile('Tutorials-DeepLizard/Clinical_Trial-Example/DeepLizard_MedicalTrial_Example.h5') is False:
    model.save('Tutorials-DeepLizard/Clinical_Trial-Example/DeepLizard_MedicalTrial_Example.h5')
print('Saved model to disk')