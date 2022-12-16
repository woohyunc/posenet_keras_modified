import PIL.ImImagePlugin
from tqdm import tqdm
import numpy as np
import os.path
import sys
import random
import math
import cv2

directory = './NewData/'
dataset_train = 'dataset_train.txt'
dataset_test = 'dataset_test.txt'

def centeredCrop(img, set_size):

    h, w, c = img.shape

    if set_size > min(h, w):
        return img

    crop_width = set_size
    crop_height = set_size

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
       
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

def preprocess(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])
		if (int(X.shape[0]) != 512) and (int(X.shape[1]) != 512):
			X = centeredCrop(X, 540)
			X = cv2.resize(X, (224, 224)).astype("float32")
		else:
			X = cv2.resize(X, (224,224)).astype("float32")
		images_cropped.append(X)
	image_data = np.array(images_cropped)
	print(image_data.shape)
	image_data /= 255.0
	image_mean = image_data.mean()
	image_std = image_data.std()
	image_data -= image_mean
	image_data /= image_std
	images_out = image_data
	return images_out

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

def get_data(dataset):
	poses = []
	images = []
	with open(directory+dataset) as f:
		for line in f:
			fname, p0,p1,p2,p3,p4,p5,p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)
			poses.append((p0,p1,p2,p3,p4,p5,p6))
			images.append(fname)
	images_out = preprocess(images)
	return datasource(images_out, poses)

def getData():
	datasource_train = get_data(dataset_train)
	datasource_test = get_data(dataset_test)

	images_train = []
	poses_train = []
	images_test = []
	poses_test = []

	for i in range(len(datasource_train.images)):
		images_train.append(datasource_train.images[i])
		poses_train.append(datasource_train.poses[i])

	for i in range(len(datasource_test.images)):
		images_test.append(datasource_test.images[i])
		poses_test.append(datasource_test.poses[i])

	return datasource(images_train, poses_train), datasource(images_test, poses_test)
