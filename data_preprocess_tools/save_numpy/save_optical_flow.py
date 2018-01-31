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


op_list_file = "/media/lci/storage/Video/candice/HMDB51_list/optical_flow_list1/train.list"
save_path = "/media/lci/storage/Video/saved_data/op/train"
stack_frame_num = 10

IMG_SIZE = 112



def read_optical_flow(op_list_file, stack_frame_num):


	with open(op_list_file) as f:
		lines_op = f.readlines()

	count=0
	
	# list for storing data and label for all the videos
	all_imgs_uv = []
	labels = []


	for i in range(len(lines_op)):
	#for i in range(10):
		
		label_op =  int(lines_op[i].split('\t')[1])


		op_path_suffix = lines_op[i].split('\t')[0]
		op_names = os.listdir(op_path_suffix)
		# storing data for each video img_uv.shape=[stack_frame_numx2, H, W]
		img_uv=[]
		start_idx = random.randint(0, len(op_names)//2-1-stack_frame_num)
	
		
		for i in range(stack_frame_num):
			
			path_img_u = os.path.join(op_path_suffix, 'flow_x_'+ str('%05d' % (start_idx+i+1)) + '.jpg')
			path_img_v = os.path.join(op_path_suffix, 'flow_y_'+ str('%05d' % (start_idx+i+1)) + '.jpg')
			
			img_u = cv2.imread(path_img_u, 0)
			img_v = cv2.imread(path_img_v, 0)

			img_u = cv2.resize(img_u, (IMG_SIZE, IMG_SIZE))
			img_v = cv2.resize(img_v, (IMG_SIZE, IMG_SIZE))
			
			img_uv.append(img_u)
			img_uv.append(img_v)
		
		all_imgs_uv.append(np.array(img_uv))
		labels.append(label_op)

	all_imgs_uv = np.array(all_imgs_uv)
	labels = np.array(labels)
	# change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
  	all_imgs_uv = np.transpose(all_imgs_uv, [0, 2, 3, 1])

  	
  	print(all_imgs_uv.shape)
  	print(labels)
	return np.array(all_imgs_uv), labels 
		
def save_numpy(op_list_file, stack_frame_num, save_path):

	if not os.path.exists(save_path):
		os.makedirs(save_path)
	imgs_uv,labels = read_optical_flow(op_list_file, stack_frame_num)
	save_imgs_path = os.path.join(save_path, 'imgs.npy')
	save_labels_path = os.path.join(save_path, 'labels.npy')
	np.save(save_imgs_path, imgs_uv)
	np.save(save_labels_path, labels)


save_numpy(op_list_file, stack_frame_num, save_path)
