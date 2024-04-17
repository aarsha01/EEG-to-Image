import tensorflow as tf
from tqdm import tqdm
import os
import shutil
import pickle
from glob import glob
from natsort import natsorted
import wandb
import numpy as np
import cv2
import math

clstoidx = {}
idxtocls = {}

for idx, item in enumerate(natsorted(glob('data/images/test/*')), start=0):
    clsname = os.path.basename(item)
    clstoidx[clsname] = idx
    idxtocls[idx] = clsname

# Load data from pickle file
with open('data/eeg/image/data.pkl', 'rb') as file:
    data = pickle.load(file, encoding='latin1')
    test_X = data['x_test']
    test_Y = data['y_test']

# Define the class name
class_name = "apple"

# Initialize a counter for the number of occurrences
z = 1

# Iterate over test data to find and store X_mobile
for X, Y in zip(test_X, test_Y):
    if idxtocls[np.argmax(Y)] == class_name:
        if z == 10:  # Adjust the index according to your requirement
            file_name = f"{class_name}.txt"
            with open(file_name, "w") as file:
                file.write("X_ = ")
                file.write(repr(X))

            print(f"Stored {class_name} data in '{file_name}'")
            break
        z += 1
