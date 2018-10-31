#!/usr/bin/env python2
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
"""
Functions for creating temporary LMDBs
Used in test_views
"""
from __future__ import absolute_import

import argparse
from collections import defaultdict
import os
import sys
import glob
import time
import cv2
import shutil

import pickle
import skimage
import skimage.io


import lmdb
import numpy as np

from keras.models import load_model, Sequential,Model


sys.path.append("../../..")


from dnn_seg.data_prep.utils.patcher import image_iterator_new
from .train_h import pad_one_makesq, make_split_map

np.random.seed(8)
import random
random.seed(1)

import PIL.Image


def feedforward_net(model,image,batch_size,split_patch=False,do_dnnEnc=False):

    orig_imgsize=image.shape[0]
    if image.shape[0]<batch_size:
        image=np.vstack((image,
            np.zeros((batch_size-image.shape[0],)+image.shape[1:],dtype=image.dtype  ) ))

    if do_dnnEnc:
        pass
    elif  split_patch:
        image = split_patches(image)

    else:
        image = image.reshape((image.shape[0], image.shape[1], image.shape[2], 1))

    image_float=image.astype(np.float32)#skimage.img_as_float(image).astype(np.float32)*255

    # print image_float.shape

    predictions=model.predict(image_float,batch_size=batch_size,verbose=0)

    out_p=np.argmax(predictions,axis=1)

    return out_p[:orig_imgsize]

def clear_net(net):
    net.blobs['data'].set_data(np.zeros(net.blobs['data'].data.shape,dtype=np.float32) )

def iterate_batch(lis,bsize):
    howMuch=(len(lis)//bsize)+(1 if len(lis)%bsize!=0 else 0)
    
    for start,end in zip(range(0,howMuch*bsize,bsize),range(bsize,(howMuch+1)*bsize,bsize)):
        yield lis[start:end] 

def split_patches(x):
    x=x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
    xback, xax, xmil = np.zeros((3,) + x.shape, dtype=x.dtype)
    xback[x == 255] = 1
    xax[x == 128] = 1
    xmil[x == 0] = 1
    return np.concatenate((xmil, xax, xback), axis=-1)


def create_prediction(model,output_file,patcher_args,
    patch_size=45,do_expand=True,deploy_batch_size=1,subsample_mask=None,
                      do_dnnEnc=False,split_patch=False):


    if type(model)==str:
        model = load_model(model)
    elif isinstance(model,Sequential) or isinstance(model,Model):
        pass
    else:
        raise Exception('Model unknown')



     
    output = open(output_file, 'w')

    print ("Testing started - "+str(time.clock()))

    test_start = time.clock()


    image_counter = 0

    filenames, test_path, groundtruth_path, type_set, patches_count, save_path=patcher_args



    lenF=len(filenames)
    for find,filename in enumerate(filenames):
        print ('Running on img:{}'.format(filename[filename.rindex(os.path.sep)+1:]))
        sys.stdout.flush()
        patches=image_iterator_new(filename, test_path, groundtruth_path,
            type_set, patches_count, save_path,do_expand=do_expand,
            patch_size=patch_size,subsample_patch_mask=subsample_mask,
                do_special=None)


        print ('Len of patches is: ',len(patches))

        # for (image,c,coords) in patches[:10]:
        #     cv2.imshow('patch',image)
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        big_image=filename[filename.rindex(os.path.sep)+1:]
        patches=[pat for pat in patches if pat[0] is not None]


        ## set Smaller Size Here
        for patch_batch in iterate_batch(patches,deploy_batch_size):
            # if image_counter==5:
            #     print 'image_counter is',image.dtype
            
            single_time_start = time.clock()
            image_lis,class_img_lis,coords_lis=tuple(zip(*patch_batch))
            image_lis_n=np.array(list(image_lis))
            if do_dnnEnc:
                # !!!!! BECAUSE I GET ONE
                # image_lis=image_lis[:patch_size]
                #image_lis_n=image_lis_n.reshape(image_lis_n.shape + (1,))
                image_lis_n=make_split_map(image_lis_n)
                image_lis_n=pad_one_makesq(image_lis_n)
                image_lis_n[image_lis_n==0]=np.mean(image_lis_n)




            # image_lis_n=None

            out=feedforward_net(model,image_lis_n,deploy_batch_size,split_patch=split_patch,do_dnnEnc=do_dnnEnc)

            # print out[:5],type(out),out.shape
            # break

            for ind_pred,(image_xx,class_img,coords,out_img) in enumerate(zip(image_lis,class_img_lis,coords_lis,out)):
                output.write(big_image+'_{}_{}'.format(coords[0],coords[1])+
                    ";"+str(class_img)+';'+str(out_img)+";\n")

            image_counter+=len(patch_batch)
            single_time_end = time.clock()

            if ((image_counter//deploy_batch_size) % 700) == 0:
                print ("Classfying image #"+str(image_counter))
                print ("Image name is {}/ {} of {}".format(big_image,find,lenF))
                print ("Predicted class is #{}.".format(out[ind_pred]))
                print ("Real class is #{}.".format(str(class_img) ))
                print ("Test took ~ "+str(single_time_start-single_time_end)+"s")

        # clear_net(net)


    test_end = time.clock()

    print ("Test took ~ "+str(test_start-test_end)+"s\n"  ,'pictures count',image_counter  )

    



    return True


def color_images_in_folder(image_location, predictions_file, artificial_margin,img_size=512, color_1=(255, 124, 123),
                           color_2=(38, 255, 23), color_3=(255, 23, 240)
                           ,color_4=(25,255,37),suf_img='-corrected.lbl_visible_nonsoft.png'):
    with open(predictions_file) as f:
        predictions = f.readlines()

    names = []

    border_compensation = artificial_margin

    for prediction in predictions:
        filename, true_class,predicted_class, new_line = prediction.split(";")
        img_filename = filename.split(os.path.sep)[-1]
        
        true_name = img_filename[:img_filename.rindex('.')]
        if true_name not in names:
            names.append(true_name)


    if not os.path.exists(image_location):
        os.makedirs(image_location)

    for name in names:

        true_name_2 = os.path.join(image_location,name+suf_img) # image_location+'one_from_each/' + name+pred_name+'_one' #+ "-corrected.lbl_visible_nonsoft.png"


        img =np.zeros((img_size,img_size,3),dtype=int)

        for prediction in predictions:

            if name in str(prediction):
                filename, true_class,predicted_class, new_line = prediction.split(";")
                img_filename = filename.split(os.path.sep)[-1]
                pixel_x, pixel_y = img_filename.split('_')[-2:]


                pixel_val=img[int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)] 

                if int(predicted_class) == 0:
                    img[int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)] = color_1

                if int(predicted_class) == 1 :
                    img[int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)] = color_2

                if int(predicted_class) == 2:
                    img[int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)] = color_3



        print ('Done coloring image: ',name)
        cv2.imwrite(true_name_2, img)




def color_mistakes_in_folder(image_location,groundtruth_path, predictions_file, artificial_margin,img_size=512,same_sizegnd_pred=False,
                           color_used=(25,255,37),do_more_colors=False,suf_img='-corrected.lbl_visible_nonsoft.png'):
    # with open(predictions_file) as f:
    #     predictions = f.readlines()

    names = []

    border_compensation = artificial_margin

    with open(predictions_file) as predictions:
        for prediction in predictions:
            # print prediction
            # sys.exit('Exiting')
            filename, true_class,predicted_class, new_line= prediction.split(";")
            img_filename = filename.split(os.path.sep)[-1]
            true_name = img_filename[:img_filename.rindex('.')]
            if true_name not in names:
                names.append(true_name)



    if not os.path.exists(image_location):
        os.makedirs(image_location)

    for name in names:

        true_name_2 = os.path.join(image_location,name+suf_img) # image_location+'one_from_each/' + name+pred_name+'_one' #+ "-corrected.lbl_visible_nonsoft.png"

        print('Groundtruth ',os.path.join(groundtruth_path,name[:name.rindex('img')+5]+'-corrected.tif'))
        img =cv2.imread(os.path.join(groundtruth_path,name[:name.rindex('img')+5]+'-corrected.tif'))    #np.zeros((img_size,img_size,3),dtype=int)

        img[img==3]=255
        img[img==2]=128
        img[img==1]=0

        with open(predictions_file) as predictions:
            for prediction in predictions:

                if name in str(prediction):
                    filename, true_class,predicted_class, new_line = prediction.split(";")
                    img_filename=filename.split(os.path.sep)[-1]
                    pixel_x, pixel_y = img_filename.split('_')[-2:]

                    predicted_c=int(predicted_class)
                    true_c=int(true_class)


                    loc_draw=int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)
                    true_c=int(img[loc_draw][0]/127)


                    if not do_more_colors and predicted_c!=true_c:
                        img[loc_draw] = color_used

                    elif do_more_colors:
                        if true_c==0 and predicted_c!=true_c:
                            img[loc_draw]=color_used[0][predicted_c-1]

                        elif true_c==1 and predicted_c!=true_c:
                            img[loc_draw]=color_used[1][predicted_c//2]

                        elif true_c==2 and predicted_c!=true_c:
                            img[loc_draw]=color_used[2][predicted_c]

                    # if int(predicted_class) == 0 and pixel_val[0]!=0:
                    #     img[int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)] = color_4
                    #
                    # if int(predicted_class) == 1 and pixel_val[0]!=128:
                    #     img[int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)] = color_4
                    #
                    # if int(predicted_class) == 2 and pixel_val[0]!=255:
                    #     img[int(int(pixel_x) - border_compensation), int(int(pixel_y) - border_compensation)] = color_4



        print ('Done coloring image: ',name,true_name_2)
        cv2.imwrite(true_name_2, img)

