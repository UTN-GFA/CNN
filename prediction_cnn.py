# make a prediction for a new image.
from cProfile import run
import os
from tkinter import font
from unittest import result
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np 

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dict = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
    }
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def run_example():
    # load the image
    img = load_image('sample_image.png')
    # load model
    model = load_model('final_model.h5')
    # predict the class
    result = model.predict(img)
    max_index_col = np.argmax(result, axis=1)[0]

    for key, value in dict.items():
        if key == max_index_col:
            categorie = value
               
    return categorie
    
run_example()

# Plot of the prediction

font = {'family': 'serif',
        'color':  'Black',
        'weight': 'normal',
        'size': 16,
        }

img = mpimg.imread('sample_image.png')
plt.figure(num="CNN Predition")
plt.title(f"The image was categorized as a: {run_example().capitalize()}", fontdict=font)
plt.axis('off')
imgplot = plt.imshow(img)
plt.show()