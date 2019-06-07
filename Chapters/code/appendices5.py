# Imports
import numpy as np
from keras.utils import to_categorical
from utils import Videos

reader = Videos(target_size=(50, 50), 
                to_gray=True, 
                max_frames=200, 
                extract_frames='middle', 
                normalize_pixels=(0, 1))

X_train = reader.read_videos(train_files)
y_train = to_categorical(train_targets,
                         num_classes=6)
print('Training data:',
      X_train.shape)
print('Training labels:',
      y_train.shape)

X_valid = reader.read_videos(valid_files)
y_valid = to_categorical(valid_targets,
                         num_classes=6)
print('Shape of validation data:',
      X_valid.shape)
print('Shape of validation labels:',
      y_valid.shape)

X_test = reader.read_videos(test_files)
y_test = to_categorical(test_targets,
                        num_classes=6)
print('Shape of testing data:',
      X_test.shape)
print('Shape of testing labels:',
      y_test.shape)
