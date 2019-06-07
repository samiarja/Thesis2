from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import keras
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt
%matplotlib inline

img_path = 'photopose.png'
img = image.load_img(img_path,
                     target_size=(224, 224)),
                    img = image.img_to_array(img)


fig = plt.figure(figsize=(20, 8))
plt.subplot(1,2,1)
plt.imshow(img.astype(np.uint8))

plt.subplot(1,2,2)
plt.hist(img[:,:,0].flatten(),
         bins=100, normed=True,
         color='r')
plt.hist(img[:,:,1].flatten(),
         bins=100, normed=True,
         color='g')
plt.hist(img[:,:,2].flatten(),
         bins=100, normed=True,
         color='b')
print(img.shape)
plt.savefig("ImagetoArray.png")

#load a complete VGG166, trained on ImageNet dataset
vgg = keras.applications.VGG16(include_top=True,
                               weights='imagenet')

# preprocess the image 
# create a batch of size 1 [N,H,W,C]
img_ = np.expand_dims(img, 0) 

img_ = preprocess_input(img_)
# apply the model to the pre-processed image : 
preds = vgg.predict(img_)
# print top 5 prediction 
pprint(decode_predictions(preds, top=5)[0])
vgg_layers = [ layer.name for layer in vgg.layers]
block3_pool_extractor = Model(inputs=vgg.input,
    outputs=vgg.get_layer('block3_pool').output)
block3_pool_featres = block3_pool_extractor.predict(img_)
# plot the first feature as gray-level image 
plt.imshow(block3_pool_featres[0, :, :, 0], cmap='gray')
plt.title("First Feature as gray-level")

def plot_feature_maps(feature_maps):
    height, width, depth = feature_maps.shape
    nb_plot = int(np.rint(np.sqrt(depth)))
    fig = plt.figure(figsize=(20, 20))
    for i in range(depth):
        plt.subplot(nb_plot, nb_plot, i+1)
        plt.imshow(feature_maps[:,:,i], cmap='gray')
        plt.title('feature map {}'.format(i+1))
        plt.savefig("All Feature map.png")
        plt.show()
























