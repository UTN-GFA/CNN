import os
from time import sleep
from turtle import width

# evaluate the deep model on the test dataset
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical

# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import Model

# analysis of convolutional operation
import numpy as np
import cv2
from scipy.ndimage import convolve

# plot 
import matplotlib.pyplot as plt
font = {'family': 'serif',
        'color':  'Black',
        'weight': 'normal',
        'size': 16,
        }

# load train and test dataset
def load_dataset():
	
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	
	# return normalized images
	return train_norm, test_norm
 
# run the test harness for evaluating a model
def run_test_harness():
	
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	
	# load model
	model = load_model('final_model.h5')
	#model.summary()

	# evaluate model on test dataset
	_, acc = model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))
	
	# cannot easily visualize filters lower down
	# retrieve weights from the second hidden layer	
	filters, biases = model.layers[0].get_weights()
	
	# normalize filter values to 0-1 so we can visualize them
	f_min, f_max = filters.min(), filters.max()
	filters = (filters - f_min) / (f_max - f_min)

	# access to the first filter of the CNN (3, 3, 3, 32) 
	filters = filters[:,:,:,:]
	
	# plot first few filters
	n_filters, ix = 6, 1
	# for i in range(n_filters):
		
	# 	# get the filter
	# 	f = filters[:, :, :, i]
		
	# 	# plot each channel separately
	# 	for j in range(3):

	# 		# specify subplot and turn of axis
	# 		ax = plt.subplot(n_filters, 3, ix)
	# 		ax.set_xticks([])
	# 		ax.set_yticks([])
			
	# 		# plot filter channel in grayscale
	# 		plt.imshow(f[:, :, j], cmap='gray')

	# 		ix += 1	

	# plt.figure(num="Filters visualization")			
	# plt.show()
	
	# summarize feature map shapes
	for i in range(len(model.layers)):
		layer = model.layers[i]
		# check for convolutional layer
		if 'conv' not in layer.name:
			continue
		# summarize output shape
		# print(i, layer.name, layer.output.shape)
	
	# redefine model to output right after the first hidden layer
	ixs = [0, 1, 4, 6, 7]
	outputs = [model.layers[i].output for i in ixs]
	model = Model(inputs=model.inputs, outputs=outputs)
	
	# load the image with the required shape
	img = load_img('horse5.png', target_size=(32, 32))

	# convert the image to an array
	img = img_to_array(img)

	# expand dimensions so that it represents a single 'sample'
	img = np.expand_dims(img, axis=0)

	# prepare the image (e.g. scale pixel values for the vgg)
	img = preprocess_input(img)

	# get feature map for first hidden layer
	feature_maps = model.predict(img)

	print(len(feature_maps))

	# plot the output from each block
	square = 3
	for i in ixs:
		for j, fmap in enumerate(feature_maps):

			# plot all 64 maps in an 8x8 squares
			ix = 1
			for m in range(square):
				for n in range(square):
				
					# specify subplot and turn of axis
					ax = plt.subplot(square, square, ix)
					ax.set_xticks([])
					ax.set_yticks([])

					# plot filter channel in grayscale
					plt.imshow(fmap[0, :, :, ix-1], cmap='gray')

					ix += 1
					
		plt.show()

# entry point, run the test harness
run_test_harness()