# %% [markdown]
# # Compulsory Assignment 3: Semantic segmentation
# 
# Please fill out the the group name, number, members and optionally the name below.
# 
# **Group number**:  40 \
# **Group member 1**: Tor Erik Aasestad\
# **Group member 2**: Tage Andersen \
# **Group member 3**:  Elias Hartmarks \
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
import os
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
def plot_samples(X, y, num_samples=5):
    """Visualize sample images and their corresponding masks."""

    fig, ax = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))

    for i in range(num_samples):
        ax[i, 0].imshow(X[i, ...], cmap='gray')
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f"Sample {i+1}")

        ax[i, 1].imshow(y[i, ...], cmap='gray') 
        ax[i, 1].axis('off')
        ax[i, 1].set_title(f"Mask {i+1}")

    plt.tight_layout()
    plt.show()

plot_samples(X_train, y_train)


# %%
# Flatten y_train to make it 1D and then get unique counts
unique_labels, counts = np.unique(y_train, return_counts=True)

# Plot the distribution
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
sns.barplot(x=unique_labels, y=counts, ax=ax)

ax.set_xlabel('Label')
ax.set_ylabel('Label count')
ax.set_title('Label distribution in Train')
fig.tight_layout()
plt.show()

# %% [markdown]
# # Preprocessing
# 
# Preprocess the dataset in whatever ways you think are helpful. 

# %%
# Find the maximum pixel value from the training data
max_pixel_value = np.max(X_train)

# Normalize the images to [0,1] using the max pixel value from the training data
X_train = X_train.astype("float32") / max_pixel_value
X_test = X_test.astype("float32") / max_pixel_value



# %%
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives

batch_size = 32
epochs = 2
validation_split = 1/5


# %% [markdown]
# ## Task 1.2 Train the model, and plot the training history
# Feel free to use the `plot_training_history` function from the provided library `utilities.py`

# %%


# Get the current working directory
current_directory = os.getcwd()

# Define the substring you want to check for
substring = "dat300-h23-40"

# Check if the substring is in the current working directory
if substring in current_directory:
    MODEL_PATH = "/mnt/users/dat300-h23-40/ca3/models/model.keras"
    PICTURE_PATH = "/mnt/users/dat300-h23-40/ca3/models/model.png"
    batch_size = 128
    epochs = 40
else:
    MODEL_PATH = "../models/model.keras"
    PICTURE_PATH = "../models/model.png"


pre_trained_model = ks.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3))

for layer in pre_trained_model.layers:
    layer.trainable = False

model_with_pretrain = ks.models.Sequential()
model_with_pretrain.add(pre_trained_model)
model_with_pretrain.add(ks.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(512, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(512, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(256, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(256, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(128, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(128, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(64, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(64, 3, activation='relu', padding='same'))
model_with_pretrain.add(ks.layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same'))
model_with_pretrain.add(ks.layers.Conv2D(1, 1, activation='sigmoid', padding='same'))

# Compile the model with appropriate loss function and metrics
model_with_pretrain.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[FalseNegatives(),
                    FalsePositives(),
                    TrueNegatives(),
                    TruePositives(),
                    F1_score])


# %% [markdown]
# ## Task 2.2 Train the transfer learning model and plot the training history
# 
# Feel free to use the `plot_training_history` function from the provided library `utilities.py`

# %%
#Using the same number of epochs and batch size as before, train the model with the pre-trained weights. 
start_time = time.time()
history_pre_trained = model_with_pretrain.fit(X_train, y_train, 
                                                batch_size=batch_size, 
                                                epochs=epochs, 
                                                validation_split=validation_split,
                                                #mute the training output
                                                verbose=0)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training took {elapsed_time:.5f} seconds.")
print(f"Per epoch: {elapsed_time/epochs:.5f} seconds.")

model_with_pretrain.save(MODEL_PATH)

# %%
print(history_pre_trained.history.keys())
plot_training_history_and_return(history_pre_trained).savefig(PICTURE_PATH)
# %%

# Flatten prediction
y_pred      = model_with_pretrain.predict(X_test)                       # Make prediction
flat_y_pred = y_pred.flatten()                            # Flatten prediction
flat_y_pred[flat_y_pred >= 0.5] = 1 # Binarize prediction (Optional, depends on output activation used)
flat_y_pred[flat_y_pred != 1]   = 0                       # Binarize prediction (Optional, depends on output activation used)
submissionDF = pd.DataFrame()
submissionDF['ID'] = range(len(flat_y_pred))              # The submission csv file must have a column called 'ID'
submissionDF['Prediction'] = flat_y_pred


