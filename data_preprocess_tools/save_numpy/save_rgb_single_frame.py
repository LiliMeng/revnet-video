"""
Tools for converting HMDB-51 optical flow images to numpy array

HMDB-51 optical flow images can be downloaded from 
https://github.com/rohitgirdhar/ActionVLAD

"""


import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import os
import random
import cv2

img_list_file="../../data_preprocess_tools/hmdb51_data_list/img_list/test.list"
save_path = "/home/lili/Video/single_stream_revnet_video/revnet-video/resnet/dataset/test"

IMG_SIZE = 56
#dataset = "UCF-101"
dataset = "HMDB-51"


def read_img(img_list_file):
	with open(img_list_file) as f:
		lines_img = f.readlines()

	# list for storing data and label for all the videos

	all_imgs_rgb = []
	labels = []
	
	
	for i in range(len(lines_img)):
	
		
		label_img = int(lines_img[i].split()[1])

		labels.append(label_img)
		
		img_path_suffix = lines_img[i].split()[0]
		img_names = os.listdir(img_path_suffix)

		# storing data for each video img_uv.shape=[stack_frame_numx2, H, W]
		img_rgb_list=[]

		start_idx = random.randint(1, (len(img_names)-2))
		#print(start_idx)

		if dataset == "HMDB-51":
			path_img_rgb = os.path.join(img_path_suffix, 'image_' + str('%05d'%(start_idx)) + '.jpg')
		elif dataset == "UCF-101":
			path_img_rgb = os.path.join(img_path_suffix,  str('%05d'%(start_idx)) + '.jpg')
		else:
			raise Exception("Not implemented yet")
		img_rgb = cv2.imread(path_img_rgb, 3)
		
		img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) 
		img_rgb = np.array(img_rgb)
			
		
		img_rgb=np.array(img_rgb)	
	
		all_imgs_rgb.append(img_rgb)


	all_imgs_rgb = np.array(all_imgs_rgb)
	labels = np.array(labels)
	
	if dataset == "HMDB-51":
		# change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
  		all_imgs_rgb = np.transpose(all_imgs_rgb, [0, 2, 3, 1])
  	


	return all_imgs_rgb, labels 
		
def save_numpy(img_list_file,  save_path):

	if not os.path.exists(save_path):
		os.makedirs(save_path)
	imgs_rgb,labels = read_img(img_list_file)
	save_imgs_path = os.path.join(save_path, 'imgs.npy')
	save_labels_path = os.path.join(save_path, 'labels.npy')
	np.save(save_imgs_path, imgs_rgb)
	np.save(save_labels_path, labels)


save_numpy(img_list_file, save_path)
