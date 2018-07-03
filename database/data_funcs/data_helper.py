import numpy as np
import cv2
import os

def create_chessmask(image_size,sparsity):
    step=int(image_size[0]*sparsity)

    mask_h=np.arange(image_size[0]*image_size[1]).reshape(image_size)
    bmask=np.zeros(image_size,dtype=np.bool)

    bmask[mask_h%step==0]=True

    return bmask


def draw_image(mat,do_random=False,spread_cor=False,custom_dir='.',img_suf=''):

    if do_random:
        suf=''.join(map(chr,np.random.randint(97,97+26,(4,))) )
    else:
        suf='one'

    suf+=img_suf
    if spread_cor:
        mat=mat.copy()
        mat-=1
        mat=mat*127

    if not os.path.exists(custom_dir):
        os.makedirs(custom_dir)
    cv2.imwrite(os.path.join(custom_dir,'temp_image_{}.png'.format(suf)),mat)