from ctypes import BigEndianStructure
from distutils.util import subst_vars
from unittest import result
from keras.models import load_model
from keras.models import Model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
from matplotlib import pyplot

# Selection of the model previously trained. 
model = load_model('final_model.h5')

successive_outputs = [layer.output for layer in model.layers[0:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = Model(inputs = model.input, outputs = successive_outputs)

#Load the input image
img = load_img('horse5.png', target_size=(32, 32))

# Convert ht image to Array of dimension (32,32,3)
x = img_to_array(img)      
x = x.astype('float32')

# normalize to range 0-1
x /= 255.0
x = x.reshape((1,) + x.shape)

# Let's run input image through our vislauization network
# to obtain all intermediate representations for the image.
successive_feature_maps = visualization_model.predict(x)

#print(f'feature maps shape: {len(successive_feature_maps)}')

for i in range(len(successive_feature_maps)):
    #print(f'Feature map [{i}]:{successive_feature_maps[i].shape}')
    pass

filters, biases = model.layers[0].get_weights()

print(f'biases: {biases[0]}')

def convolution(filter, biases, image, stride, number_filter): # stride -> Para poder generalizar. 
	
	# height = image.shape[0]
	# width = image.shape[1]
	# channels = image.shape[2]

	filter_height = filter.shape[0]
	filter_width = filter.shape[1]
	filter_channels = filter.shape[2]

	result = 0

	for h in range(filter_height):
		for w in range(filter_width):
			for c in range(filter_channels): #RGB Image input
				result += image[0,h,w+stride,c] * filter[h,w,c,number_filter] #number_filter = first filter of the layer. 
				print(f'column:{h}, row:{w}, channel:{c}, pixel value:{image[0,h,w+stride,c]}')


	print(f'Result of conv element 1 = {result+biases[0]}')

convolution(filters, biases, x, 0, 0)

# Operation Validation - First Pixel 
# print(successive_feature_maps[0][0,1,1,0]) 