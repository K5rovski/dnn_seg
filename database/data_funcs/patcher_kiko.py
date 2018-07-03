import glob
import numpy as np
import os
import sys
import random
import time
import cv2
import itertools
import pickle

from data_funcs.data_helper import draw_image

random.seed(8)

def cut_kiko(expanded_image, groundtruth_image, pixel_x, pixel_y,
 border_bar, output_namebase, set_type, save_path,do_expand=False,do_patch=True):

    groundtruth_class = (str(groundtruth_image[pixel_x-border_bar, pixel_y-border_bar]))
    if not do_expand:
        groundtruth_class=str(groundtruth_image[pixel_x,pixel_y])


    # new_name = (output_namebase + "-px-" + str(pixel_x) + "_" + str(pixel_y) + ".png")
    # save_path+=output_namebase

    # print save_path+'/0/' + new_name

    if int(groundtruth_class) == 1:
        class_img='0'

    elif int(groundtruth_class) == 3:
        class_img='2'

    elif (int(groundtruth_class) == 2):
        class_img='1'
    else:
        return [None]*3

    if do_patch:

        patch = expanded_image[pixel_x - border_bar: pixel_x + border_bar+1, pixel_y - border_bar: pixel_y + border_bar+1]
    else:
        patch=None
    return patch,class_img,(pixel_x,pixel_y)



def cut_special(expanded_image, groundtruth_image, pixel_x, pixel_y,
 border_bar, output_namebase, set_type, save_path,do_expand=False,do_patch=True,not_mix_patches=False,**kwargs):

    perc_switch=kwargs.get('perc_switch',1)
    do_twopatch=kwargs.get('two_patch',False )
    do_fold_expand=kwargs.get('folded_expand',False)

    groundtruth_class = (str(groundtruth_image[pixel_x-border_bar, pixel_y-border_bar]))
     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if not do_expand :
        groundtruth_class=str(groundtruth_image[pixel_x,pixel_y])

    if do_twopatch:
        groundtruth_class = str(groundtruth_image[pixel_x, pixel_y])
    # new_name = (output_namebase + "-px-" + str(pixel_x) + "_" + str(pixel_y) + ".png")
    # save_path+=output_namebase

    # print save_path+'/0/' + new_name

    if int(groundtruth_class) == 1:
        class_img='0'

    elif (int(groundtruth_class) == 2):
        class_img='1'

    elif int(groundtruth_class) == 3:
        class_img='2'

    else:
        raise Exception('Bad groundtruth'+str(int(groundtruth_class) ))

    lookup_patch = expanded_image[pixel_x - border_bar: pixel_x + border_bar + 1,
            pixel_y - border_bar: pixel_y + border_bar + 1]

    gnd_patch=groundtruth_image[pixel_x - border_bar: pixel_x + border_bar + 1,
            pixel_y - border_bar: pixel_y + border_bar + 1]


    # if np.any( (lookup_patch==0) & (gnd_patch!=3) ) or \
    #      not np.array_equal(  ((gnd_patch==3) & (lookup_patch==0)),(lookup_patch==0)):
    #     raise Exception('These patches overlap...')
    if not not_mix_patches and do_patch:
        ret_patch=gnd_patch.copy()

        if  random.random()<perc_switch:
            ret_patch[(gnd_patch==3) & (lookup_patch==0)]=1
            # ret_patch[(gnd_patch == 3) & (lookup_patch == 128)] = 2


        # if random.random()<0.00005:
        #     draw_image(ret_patch,do_random=False,spread_cor=True,
        #                custom_dir='pics_mix40',img_suf='_{}_{}_ret'.format(pixel_x,pixel_y) )
        #     draw_image(gnd_patch, do_random=False,
        #                spread_cor=True, custom_dir='pics_mix40',img_suf='_{}_{}_gnd'.format(pixel_x,pixel_y))


    if do_twopatch and do_patch:

        if not_mix_patches:
            ret_patch=lookup_patch

        try:
            if gnd_patch.shape!=ret_patch.shape:
                np.array(ret_patch)-np.array(gnd_patch)
            ret_patch=np.concatenate((ret_patch,gnd_patch),axis=0)
        except:
            print(ret_patch.shape,gnd_patch.shape,pixel_y,pixel_x)
            raise
    if not do_patch:
        ret_patch=None

    if do_fold_expand:
        wsize=(border_bar*2)+1
        ret_patch=np.zeros((wsize*3,wsize*3),dtype=gnd_patch.dtype)
        for ind_in,(indI,indJ) in enumerate(itertools.product([-1, 0, 1], [-1, 0, 1])):
            singlP=expanded_image[pixel_x-border_bar+(wsize*indI):pixel_x+border_bar+(wsize*indI)+1,
                      pixel_y-border_bar+(wsize*indJ):pixel_y+border_bar+(wsize*indJ)+1]
            if singlP.shape[0]!=wsize or singlP.shape[1]!=wsize:
                singlP=np.zeros((wsize,wsize),dtype=gnd_patch.dtype)+255
            ret_patch[wsize+(wsize*indI):wsize+wsize+(wsize*indI),
                    wsize+(wsize*indJ):wsize+wsize+(wsize*indJ)]=singlP

    # if ret_patch is not None and ret_patch[0,0]==255:
    #     print('This is wrong')

    return ret_patch,class_img,(pixel_x,pixel_y)

def cut_special_stacktwo(expanded_image, groundtruth_image, pixel_x, pixel_y,
 border_bar, output_namebase, set_type, save_path,do_expand=False,do_patch=True,**kwargs):



    groundtruth_class = (str(groundtruth_image[pixel_x-border_bar, pixel_y-border_bar]))
    if not do_expand:
        groundtruth_class=str(groundtruth_image[pixel_x,pixel_y])


    # new_name = (output_namebase + "-px-" + str(pixel_x) + "_" + str(pixel_y) + ".png")
    # save_path+=output_namebase

    # print save_path+'/0/' + new_name

    if int(groundtruth_class) == 1:
        class_img='0'

    elif (int(groundtruth_class) == 2):
        class_img='1'

    elif int(groundtruth_class) == 3:
        class_img='2'

    else:
        raise Exception('Bad groundtruth')

    lookup_patch = expanded_image[pixel_x - border_bar: pixel_x + border_bar + 1,
            pixel_y - border_bar: pixel_y + border_bar + 1]

    gnd_patch=groundtruth_image[pixel_x - border_bar: pixel_x + border_bar + 1,
            pixel_y - border_bar: pixel_y + border_bar + 1]


    # if np.any( (lookup_patch==0) & (gnd_patch!=3) ) or \
    #      not np.array_equal(  ((gnd_patch==3) & (lookup_patch==0)),(lookup_patch==0)):
    #     raise Exception('These patches overlap...')

    if do_patch:
        ret_patch=np.vstack((gnd_patch,lookup_patch))
    else:
        ret_patch=None


    return ret_patch,class_img,(pixel_x,pixel_y)


### KIKOS CODE


def is_valid_pixel(img_size,offset_size,pix):
    if (pix-offset_size)>0 and pix<(img_size-offset_size):
        return True
    return False


def empty_patch():
    return None,None,(None,None)



def patch_image_expand_kiko(image_path, window_size, groundtruth_path,
    output_namebase, patches_per_image, set_type, save_path,PATCH_DIR,file_key,
    subsample_patch_mask=None,do_expand=True,do_print=False,do_patch=True,
                            do_special=None):

    start = time.clock()

    if do_print: print( "Patching Test: "+str(image_path))

    patcher_image = cv2.imread(image_path, 0)




    groundtruth_image = cv2.imread(groundtruth_path, 0)

    if do_print: print('Image type: ',patcher_image.dtype)

    WHITE = [255, 255, 255]

    input_height, input_width = patcher_image.shape
    border_bar = window_size // 2

    if do_expand:
        x = range(0 + border_bar, input_width + border_bar)
        y = range(0 + border_bar, input_height + border_bar)

        expanded_image = cv2.copyMakeBorder(patcher_image,
                                            border_bar,
                                            border_bar,
                                            border_bar,
                                            border_bar,
                                            cv2.BORDER_CONSTANT,
                                            value=WHITE)

        gnd_new_image = cv2.copyMakeBorder(groundtruth_image,
                                            border_bar,
                                            border_bar,
                                            border_bar,
                                            border_bar,
                                            cv2.BORDER_CONSTANT,
                                            value=[3,3,3])

    else:
        x = range(0 + border_bar, input_width - border_bar)
        y = range(0 + border_bar, input_height - border_bar)
        expanded_image=patcher_image
        gnd_new_image=groundtruth_image

        if do_print: print ('There will be {} patches, from {}-{}'\
            .format(len(x)*len(y) , 0+ border_bar, input_height - border_bar ))





    # cv2.imshow('image',expanded_image)
    # cv2.imshow('image g',groundtruth_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    pixels = []

    selected_pixels_gray=[]

    for i in range(0, len(x)):
        for j in range(0, len(y)):
            pixels.append((x[i], y[j]))

    if do_print: print("There should be: " + str(len(pixels)) + " patches.")

    if patches_per_image > 0:
        if do_print: print("We will be taking " + str(patches_per_image) + " patches.")
        selected_pixels = random.sample(pixels, patches_per_image)
    
    elif patches_per_image==-1 and PATCH_DIR is not None:
        selected_pixels=[]

        one_patch=os.path.join(PATCH_DIR,output_namebase+'.patches')
        # print '!!!!!!!!!!!!!!!!',one_patch

        with open(one_patch,'rb') as f:
            done_patches=set( i[2:] for i in pickle.load(f) )        


        for (pixel_x, pixel_y) in pixels:
            patch_name=(output_namebase + "-px-" + str(pixel_x) + "_" + str(pixel_y) + ".png")
            if patch_name not in done_patches:
                selected_pixels.append((pixel_x,pixel_y))

        if do_print: print( 'Patches done:{}, remaining patches:{},one patch:{}'.format(len(done_patches),
            len(selected_pixels),done_patches.pop()) )
    elif patches_per_image==-2 and PATCH_DIR is not None:
        img_mask_f=file_key+"_mask.tif"
        img_mask=cv2.imread(  os.path.join(PATCH_DIR,img_mask_f),0)
        selected_pixels=[]

        for ind,(pix_x,pix_y) in enumerate(zip(*np.where (img_mask==255)) ):
            if not do_expand and (not is_valid_pixel(input_height,border_bar,pix_x)\
             or not is_valid_pixel(input_height,border_bar,pix_y) ):
                continue

            selected_pixels.append((pix_x,pix_y))


        selected_pixels_gray=[]
        for ind,(pix_x,pix_y) in enumerate(zip(*np.where (img_mask==120)) ):
            if not do_expand and (not is_valid_pixel(input_height,border_bar,pix_x)\
             or not is_valid_pixel(input_height,border_bar,pix_y) ):
                continue

            selected_pixels_gray.append((pix_x,pix_y))


        selected_pixels.extend(selected_pixels_gray)


    else:
        if do_print: print ("We will be taking all patches.")
        selected_pixels = pixels


    if subsample_patch_mask is None:
        mask_choice=np.ones((len(selected_pixels), ),dtype=bool)
    else:
        mask_choice=subsample_patch_mask

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if subsample_patch_mask.shape[0]!=len(selected_pixels):
            raise Exception('Subsample mask not eqaul to patcher output')


    # Parallel(n_jobs=1)(delayed(cut)(expanded_image, groundtruth_image, pixel_x, pixel_y, border_bar, output_namebase, set_type, save_path) for (pixel_x, pixel_y) in selected_pixels)

    if not do_special is not None:
        patches=[]
        for ((pixel_x,pixel_y),do_pass) in zip(selected_pixels,mask_choice):
            if  do_pass:

                patches.append(cut_kiko(expanded_image,gnd_new_image,pixel_x,pixel_y,

                    border_bar,output_namebase,set_type,save_path,do_expand=do_expand,do_patch=do_patch ))

            else:
                patches.append(empty_patch())

    if do_special is not None:
        patches=[]
        selected_cellpixels_border = len(selected_pixels)-len(selected_pixels_gray)

        for indpix,((pixel_x, pixel_y),do_pass) in enumerate(zip(selected_pixels,mask_choice)):

            if do_pass and indpix<selected_cellpixels_border:

                patches.append(cut_special(expanded_image, gnd_new_image, pixel_x, pixel_y,
                                        border_bar, output_namebase, set_type, save_path, do_expand=do_expand,
                                        do_patch=do_patch,not_mix_patches=True,two_patch=do_special.get('two_patch',False),
                                           folded_expand=do_special.get('folded_expand', False)
                                           ))
            elif do_pass and indpix>=selected_cellpixels_border:
                patches.append(cut_special(expanded_image, gnd_new_image, pixel_x, pixel_y,
                                           border_bar, output_namebase, set_type, save_path, do_expand=do_expand,
                                           do_patch=do_patch,not_mix_patches=do_special.get('not_mix_patches',False),
                            perc_switch=do_special['perc_switch'],two_patch=do_special.get('two_patch',False),
                                           folded_expand=do_special.get('folded_expand',False)   ))
            else:
                patches.append(empty_patch())


    end = time.clock()

    if do_print: print ("Patching took: "+str(end-start)+"s.")

    return patches

def image_iterator_kiko(filename, test_path, groundtruth_path, type, patches,
        save_path,PATCH_DIR=None,do_expand=True,patch_size=45,
        do_print=False,do_patch=True,subsample_patch_mask=None,
        do_special=None):


    filename_small=filename[filename.rindex(os.path.sep)+1:filename.rindex('img')+5]
    # check here ####
    groundtruth_image_path = os.path.join(groundtruth_path,(filename_small + "-corrected.tif") )
    print('Starting patching on image: ',filename_small," ...\n")
    start_t=time.time()

    patches=patch_image_expand_kiko(filename,
                        patch_size,
                        groundtruth_image_path,
                       filename_small,
                        patches,
                        type,
                        save_path,PATCH_DIR,filename_small,
                            subsample_patch_mask=subsample_patch_mask,
                        do_expand=do_expand,do_print=do_print,do_patch=do_patch,
                        do_special=do_special)

    print('Finished patching: {:.2f} secs.'.format(time.time()-start_t))
    return patches

















# ================================================================
# ================================================================
# ================================================================
# ================================================================
# ================================================================
# ================================================================
# ================================================================
# ================================================================
# ================================================================

