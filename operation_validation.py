from distutils.util import subst_vars
from unittest import result
from keras.models import load_model
from keras.models import Model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np

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

print(successive_feature_maps[0][0,1,3,0])
# print(successive_feature_maps[1])

filters, biases = model.layers[0].get_weights()

# print('#####')
# print(f'filter 1: {filters[:,:,:,0]}')
# print('#####')

print(f'biases: {biases[0]}')

def convolution(filter, biases, image, stride, number_filter): # stride -> Para poder generalizar. 
	
	# height = image.shape[0]
	# width = image.shape[1]
	# channels = image.shape[2]

	filter_height = filter.shape[0]
	filter_width = filter.shape[1]
	filter_channels = filter.shape[2]

	result = 0

	for i in range(filter_height):
		for j in range(filter_width):
			for k in range(filter_channels): #RGB Image input
				result += image[0,i,j+2,k] * filter[i,j,k,number_filter] #number_filter = first filter of the layer. 

	print(f'Result of conv element 1 = {result+biases[0]}')

convolution(filters, biases, x, 0, 0)


# filter 1: 
# # R
# [[[ 0.5698619   0.40890396 -0.36380863]
#   [ 0.5614102   0.5993685   0.55676275]
#   [ 0.55084157  0.2543275   0.13960779]]

# # G
#  [[ 0.21123037 -0.43697175  0.26390883]
#   [ 0.5817639   0.06042234  0.01526073]
#   [ 0.05200297 -0.39285138 -0.33733857]]

# # B 
#  [[-0.03604303 -0.52488816 -0.2592365 ]
#   [-0.407283   -0.09848811 -0.19774196]
#   [-0.3290573   0.16400887 -0.4091162 ]]]


