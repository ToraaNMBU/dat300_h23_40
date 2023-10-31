# %% [markdown]
# # Compulsory Assignment 3: Semantic segmentation
# 
# Please fill out the the group name, number, members and optionally the name below.
# 
# **Group number**: \
# **Group member 1**: \
# **Group member 2**: \
# **Group member 3**: \
# **Group name (optional)**: 
# 
# 
# # Assignment Submission
# To complete this assignment answer the relevant questions in this notebook and write the code required to implement the relevant models. This is the biggest assignemnt of the semester, and therefore you get two weeks to work on it. However, we reccomend that **you start early**. This assignment has three semi-big sections, each of which build upon the last. So if you delay the assignment until the day before submission, you will most likely fail. This assignment is completed by doing the following. 
# * Submit notebook as an .ipynb file to canvas.
# * Submit notebook as an .pdf file to canvas.
# * Submit the python script you run on ORION to canvas.
# * Submit the SLURM script you run on ORION to canvas.
# * Submit at least one of your model predictions to the Kaggle leaderboard, and attain a score that is higher than the *BEAT ME* score. 
# 
# NOTE: Remember to go through the rules given in the lecture "Introduction to compulsory assignments", as there are many do's and dont's with regard to how you should present the work you are going to submit.
# 

# %% [markdown]
# 
# # Introduction 
# This assignment will center around semantic segmentation of the dataset in the TGS salt identification challenge. Several of the Earths accumulations of oil and gas **also** have huge deposits of salt, which is easier to detect than the actual hydrocarbons. However, knowing where the salt deposits are precisely is still quite difficult, and segmentation of the seismic images is still something that requires expert interpretation of the images. This leads variable, and highly subjective renderings. To create more accurate, objective segmentations TGS (the worlds leading geoscience data company) have created this challenge to determine if a deep learning model is up to the task. 
# 
# ## Dataset
# In this assigmnet you will be given 3500 annotated images. The image, and mask dimensions are 128x128 pixels. With each image there follows an annotation mask where each pixel is classified as `1` (salt deposit) or `0` not salt deposit. The test-dataset contains 500 images, where no ground truth masks are given. To evualuate your model on the test dataset, submit your predictions to the Kaggle leaderboard.
# 
# ## Assignment tasks
# 
# 1. Implement a U-net model, and train it to segment the dataset.
# 2. Implement a U-net model that uses a pre-trained backbone model of your choice (VGGnet, ResNet, DarkNet, etc.), and train it to segment the dataset.
# 3. Train one of the models from part 1 or 2 on Orion, and compare the training times and attained performances.
# 4. Submit the best model prediction on Kaggle learderboard.

# %% [markdown]
# 
# ## Submissions to the Kaggle leaderboard
# 
# Link to the Kaggle leaderboard will be posted in the Canvas assignment.
# 
# ```python
# y_pred      = model.predict(X_test)                       # Make prediction
# flat_y_pred = y_pred.flatten()                            # Flatten prediction
# flat_y_pred[flat_y_pred >= USER_DETERMINED_THRESHOLD] = 1 # Binarize prediction (Optional, depends on output activation used)
# flat_y_pred[flat_y_pred != 1]   = 0                       # Binarize prediction (Optional, depends on output activation used)
# submissionDF = pd.DataFrame()
# submissionDF['ID'] = range(len(flat_y_pred))              # The submission csv file must have a column called 'ID'
# submissionDF['Prediction'] = flat_y_pred
# submissionDF.to_csv('submission.csv', index=False)        # Remember to store the dataframe to csv without the nameless index column.
# ```

# %% [markdown]
# 
# # Library imports

# %%
import time
from tqdm import tqdm # Cool progress bar

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.keras as ks
import tensorflow as tf

SEED = 458 # Feel free to set another seed if you want to
RNG = np.random.default_rng(SEED) # Random number generator
tf.random.set_seed(SEED)

from utilities import *
from visualization import *

# %% [markdown]
# # Data loading
# 
# Load the data from the HDF5 file `student_TGS_challenge.h5` that is available on Canvas, and Kaggle.
# The data should be loaded in the same manner as in CA2. 

# %%
dataset_path = './student_TGS_challenge.h5'
with h5py.File(dataset_path,'r') as f:
    print('Datasets in file:', list(f.keys()))
    X_train = np.asarray(f['X_train'])
    y_train = np.asarray(f['y_train'])
    X_test = np.asarray(f['X_test'])
    print('Nr. train images: %i'%(X_train.shape[0]))
    print('Nr. test images: %i'%(X_test.shape[0]))



# %%
# print the resolution and the number of channels for the train and test images
print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape, X_test.shape)

# %% [markdown]
# # Visualization
# 
# Plot a few samples images and masks. Feel free to visualize any other aspects of the dataset that you feel are relevant. 

# %%
# plot a few sambles and masks from the training set


# %% [markdown]
# # Preprocessing
# 
# Preprocess the dataset in whatever ways you think are helpful. 

# %%


# %% [markdown]
# # Part 1: Implementing U-net
# 
# ## Intersection over Union
# 
# The IoU score is a popular metric in both segmentation and object detection problems. 
# 
# If you want to use the `plot_training_history` function in the `visualization.py` library remember to compile the model with the TP, TN, FP, FN metrics such that you can estimate the *Intersection-over-Union*. **However, it is voluntary to estimate IoU**
# 
# See example below:
# 
# ```python
# from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives
# from utilities import F1_score, 
# from visualization import plot_training_history, 
# ...
# model.compile(optimizer='Something', 
#                   loss='Something else', 
#                   metrics=[FalseNegatives(),
#                            FalsePositives(),
#                            TrueNegatives(),
#                            TruePositives(),
#                            F1_score,
#                            OtherMetricOfChoice])
# 
# training_history = model.fit(X_train, y_train, ...)
# plot_training_history(training_history)
# ```
# 
# You have also been provided with a custom F1-score metric in the `utilities.py` library, which is specific for image segmentation. **This is mandatory to use when compiling the model**.

# %% [markdown]
# 
# ## Task 1.1 Model implementation
# 
# Implement the classical U-net structure that you have learned about in the lectures. Feel free to experiment with the number of layers, loss-function, batch-normalization, etc. **Remember to compile with the F1-score metric**. 
# 

# %%
"""
Version of U-Net with dropout and size preservation (padding= 'same')
""" 
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True, n_classes = 2):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# %%
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives


input_img = Input(shape=(128,128,3))
model = get_unet(input_img, n_filters = 32, dropout = 0.0, batchnorm = True, n_classes = 1)

# %% [markdown]
# ## Task 1.2 Train the model, and plot the training history
# Feel free to use the `plot_training_history` function from the provided library `utilities.py`

# %%
batch_size = 64
epochs = 10

# Create a validation split
validation_split = 1/6

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=[FalseNegatives(),
                       FalsePositives(),
                       TrueNegatives(),
                       TruePositives(),
                       F1_score])
# Train the model
start_time = time.time()
history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_split=validation_split)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.5f} seconds.")

model.save("/mnt/users/dat300-h23-40/ca3/models/model.keras")


# %% [markdown]
# ## Task 1.3 Visualize model predictions
# 
# Make a plot that illustrates the original image, the predicted mask, and the ground truth mask. 

# %% [markdown]
# # Part 2: Implementing U-net with transfer learning
# 
# Implement a model with the U-net structure that you have learned about in the lectures, but now with a pre-trained backbone. There are many pre-trained back-bones to choose from. Pick freely from the selection here [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications), or here [Keras model scores](https://keras.io/api/applications/) (nicer table in the second link). Feel free to experiment with the number of layers, loss-function, batch-normalization, etc. Many of the backbones available are quite big, so you might find it quite time-consuming to train them on your personal computers. It might be expedient to only train them for 1-5 epochs on your PCs, and do the full training on Orion in Part 3. 
# 
# ## Task 2.1 Transfer learning model implementation
# 
# Implement a U-net model utilizing the pre-trained weights of a publically available network. **Remember to compile with the F1-score metric**.

# %%


# %% [markdown]
# ## Task 2.2 Train the transfer learning model and plot the training history
# 
# Feel free to use the `plot_training_history` function from the provided library `utilities.py`

# %%


# %% [markdown]
# # Part 3: Training your model Orion
# 
# Use the lecture slides from the Orion-lecture to get started.
# 1. Put one of your model implementations into a python script (`.py`)
# 2. Transfer that script to Orion.
# 3. Change the relevant path variables in your python script (path-to-data for example), and make sure that you record the time it takes to train the model in the script. This can be done using the `time` library for example.
# 4. Set up a SLURM-script to train your model, please use the example from the Orion lecture as a base.
# 5. Submit your SLURM job, and let the magic happen. 
# 
# If you wish to use a model trained on Orion to make a Kaggle submission, remeber to save the model, such that you can transfer it to your local computer to make a prediction on `X_test`, or test the model on Orion directly if you want to. 
# 
# ## Tips
# 
# If you compiled, trained and stored a model on Orion with a custom performance metric (such as F1-score), remember to specify that metric when loading the model on your computer again.
# 
# Loading a saved model:
# ```python
# trained_model = tf.keras.models.load_model('some/path/to/my_trained_model.keras', custom_objects={'F1_score': F1_score})
# ```
# 
# Loading a checkpoint:
# ```python
# trained_model = tf.keras.saving.load_model('some/path/to/my_trained_model_checkpoint', custom_objects={'F1_score': F1_score})
# ```

# %% [markdown]
# # Discussion
# 
# **Question 1: Which model architectures did you explore, and what type of hyperparameter optimization did you try?**
# 
# **Answer 1:**
# 
# **Question 2: Which of the model(s) did you choose to train on Orion, and how long did it take to train it on Orion?**
# 
# **Answer 2:**
# 
# **Question 3: What where the biggest challenges with this assignment?**
# 
# **Answer 3:**

# %% [markdown]
# # Kaggle submission
# 
# Evaluate your best model on the test dataset and submit your prediction to the Kaggle leaderboard.
# Link to the Kaggle leaderboard will be posted in the Canvas assignment.

# %%
y_pred = model.predict(X_test)

flat_y_pred = y_pred.flatten() # Flatten prediction
flat_y_pred[flat_y_pred >= THRESHOLD] = 1 # Binarize prediction (Optional, depends on output activation used)
flat_y_pred[flat_y_pred != 1]   = 0 # Binarize prediction (Optional, depends on output activation used)
submissionDF = pd.DataFrame()
submissionDF['ID'] = range(len(flat_y_pred)) # The submission csv file must have a column called 'ID'
submissionDF['Prediction'] = flat_y_pred
submissionDF.to_csv('submission.csv', index=False) # Remember to store the dataframe to csv without the nameless index column.

# %%



