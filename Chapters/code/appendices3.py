#Make the code running on both python2 and python3
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

# Imports Libraries and dependencies
import os 
import cv2 
from sklearn.datasets import load_files 
from sklearn.model_selection import train_test_split 
raw_data = load_files(os.getcwd() + r'/Data',
                      shuffle=False)
files = raw_data['filenames']
targets = raw_data['target']

train_files,
test_files,
train_targets,
test_targets = train_test_split(files,
                                targets,
                                test_size=1/3,
                                random_state=191)

# Taking ~25% of the training data for validation
valid_files = train_files[300:]
valid_targets = train_targets[300:]

# Remaining data will be used for training the model
train_files = train_files[:300]
train_targets = train_targets[:300]

# Generic details about the data
print('Total number of videos:', len(files))
print('\nNumber of videos in training data:',
      train_files.shape[0])
print('Number of videos in validation data:',
      valid_files.shape[0])
print('Number of videos in test data:',
      test_files.shape[0])
