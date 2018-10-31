import sys
from os.path import join
import cv2
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model


sys.path.append('../..')

from dnn_seg.net.utils.train_h import draw_autoenc_smaller,make_split_map


# TEST MODEL WITH CUSTOMLY DEFINED PATCHES====================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


draw_encoded_loc=r'D:\code_projects\dnn_seg\autoenc\encoded_maps_vis'
model_loc=r'D:\data_projects\neuron_fibers_data\models\weights_autoenc_dvov.00-0.2944.hdf5'


patchs=[]

img1 = cv2.imread(r'D:\data_projects\neuron_fibers_data\images\test_spread_interim\sp13909-img04-interim.tif', 0)
img2=cv2.imread(r'D:\data_projects\neuron_fibers_data\images\all_test_big_corrected\sp13909-img04-corrected.tif',0)
for img in [img1,img2]:
	patchs.append( img[834 - 22:834 + 23, 676 - 22:676 + 23])

	xcor, ycor = 1352, 922
	patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
	xcor, ycor = 1043, 871
	patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
	xcor, ycor = 1180, 1192
	patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
	xcor, ycor = 1458, 1134
	patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
	xcor, ycor = 1465, 934
	patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])

patchs=np.array(patchs)

model=load_model(model_loc,custom_objects={'tf':tf})
mod_name=model_loc[model_loc.index('weights_autoenc_'):]
draw_loc=join(draw_encoded_loc,mod_name)
if not os.path.exists(draw_loc): os.makedirs(draw_loc)




real_vals = make_split_map(patchs[:6],dopad=True)
real_vals2=make_split_map(patchs[6:],dopad=True,split_map=[1,2,3])
patches = np.zeros((2, 6, 48, 48, 3), dtype=np.uint8)
patches[0,:]=real_vals
patches[1,:]=real_vals2

draw_autoenc_smaller(model,draw_loc,patches)

#    draw_autoenc(model, img_w, img_h, draw_loc, val_lmdb)
sys.exit('Im am done with everything!!')
