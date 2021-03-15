from importing_modules import *

warnings.simplefilter(action='ignore', category=FutureWarning)

# -- Identifying the GPU --
# gpu = len(tf.config.list_physical_devices('GPU')) > 0
# print("GPU is", "available" if gpu else "NOT AVAILABLE")

# -- Organize data into train, valid, test, directories! --
os.chdir('Data_Set')
if os.path.isdir('train/dog') is False:  # Makes sure the directories that we are about to make don't already exist
    os.makedirs('train/dog')  # Train set
    os.makedirs('train/cat')
    os.makedirs('valid/dog')  # Validation Set
    os.makedirs('valid/cat')
    os.makedirs('test/dog')  # Test Set
    os.makedirs('test/cat')

    # Moving randomly 500 (out of 12500) cat images to the train/cat
    for i in random.sample(glob.glob('cat*'), 500):
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 500):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 100):
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 100):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 50):
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 50):
        shutil.move(i, 'test/dog')

os.chdir('../')

# Processing the data

test_path = 'Data_Set/test'
train_path = 'Data_Set/train'
valid_path = 'Data_Set/valid'

# -- When we train a model, the model (in our case, a sequential model with the data passing threw the fit function) --

# expects data in a certain format - We are putting our images into a Keras generator. each variable here will return
# a directory iterator. TL;DR: It will create batches of data from the directories where our data set resides. Then,
# this batches will be able to pass to the sequential model using the fit function!

# Using the image processing of the VGG16 model
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
# ImageDataGenerator(-): Function that will apply some type of image preprocessing on the images before they pass to
# the NN - specifically, this image processing is corresponding to the vgg16 model that will be used
# .flow_from_directory(-): this is where we are passing in the data and specifying how we want the data to be processed
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10,
                         shuffle=False)
# The test data has shuffle = false because, when we use the test_batch in inference, we want to see the predicted
# results in a confusion matrix, so we want to access the unshuffled labels for our test set


# -- Visualizing the Data! --

# Grabbing a single batch of images and the corresponding labels from the train_batches (which are 10 images and labels in this case)
imgs, labels = next(train_batches)


# Plotting the images (function from TensorFlow website)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


print(len(train_batches))
print(labels)  # Vector of the labels: Left-Cat | Right-Dog
plotImages(imgs)  # Image data that we pass through the model!

