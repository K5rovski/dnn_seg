import numpy as np
import cv2
import os
import sys

sys.path.append('../../..')
from .patcher import image_iterator_new


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


def get_patches(lookup_test_path, groundtruth_path, patch_size, subsample_mask):
    filenames, test_path, type_set, patches_count, save_path = (
        [os.path.join(lookup_test_path, i) for i in os.listdir(lookup_test_path) if i.endswith('.tiff')],
        lookup_test_path, "all_test", 0, '#####')

    lenF = len(filenames)
    print(lenF, filenames[0])
    for find, filename in enumerate(filenames):
        print('Running on img:{}'.format(filename))
        sys.stdout.flush()
        patches = image_iterator_new(filename, test_path, groundtruth_path,
                                     type_set, patches_count, save_path, do_expand=True, patch_size=patch_size,
                                     subsample_patch_mask=subsample_mask)
        return patches

def transform_patch(x):
    xback, xax, xmil = np.zeros((3,) + x.shape, dtype=x.dtype)
    xback[(x == 255) | (x == 3)] = 1
    xax[(x == 128) | (x == 2)] = 1
    xmil[(x == 0) | (x == 1)] = 1
    x_patch = np.concatenate((xmil, xax, xback), axis=-1)
    new_x=np.pad(  x_patch, ((1,2),(1,2)),'constant',constant_values=0 )

    return new_x

