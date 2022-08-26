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

# Operation Validation
# print(successive_feature_maps[0][0,1,1,0])

# print('Feature Map')
# print(successive_feature_maps[0].shape)
# print(successive_feature_maps[0][0,1:4,2:5,1:3])


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

	for h in range(filter_height):
		for w in range(filter_width):
			for c in range(filter_channels): #RGB Image input
				result += image[0,h,w+stride,c] * filter[h,w,c,number_filter] #number_filter = first filter of the layer. 
				print(f'column:{h}, row:{w}, channel:{c}, pixel value:{image[0,h,w+stride,c]}')


	print(f'Result of conv element 1 = {result+biases[0]}')

convolution(filters, biases, x, 0, 0)


print(x[0,0:3,0:3,0:3])

square = 3

for fmap in successive_feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()


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


# Image: 
# [[[0.8666667  0.8862745  0.7764706 ]
#   [0.827451   0.84705883 0.70980394]
#   [0.7647059  0.78039217 0.67058825]]

#  [[0.8352941  0.8627451  0.7176471 ]
#   [0.8039216  0.8352941  0.6509804 ]
#   [0.7882353  0.8235294  0.6509804 ]]

#  [[0.8156863  0.8627451  0.68235296]
#   [0.7607843  0.827451   0.60784316]
#   [0.77254903 0.84313726 0.63529414]]]