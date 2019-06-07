# Imports
import numpy as np
import matplotlib.pyplot as plt
from utils import Videos
%matplotlib inline
sample_files = train_files[:1]
reader = Videos(target_size=None, 
                to_gray=False)
sample = reader.read_videos(sample_files)
print('\nShape of the sample data:', sample.shape)
plt.imshow(sample[0][300])
