
"""
Functions for creating  LMDBs

"""

import os
import sys
import time

import lmdb
import numpy as np
import skimage

from data_funcs.lmdb_helper import _get_image_transaction, _db_commit_sync, _db_just_put,_reduce_lmdbsize
from .patcher_kiko import image_iterator_kiko

np.random.seed(8)

import random
random.seed(1)


def make_segmented_patches(patch_passed_maps,passed_counts,shuffled_inds,batch_s,batch_e,    imgind):
    # size of array is size that the method produces
    # True values are the ones let go by rand

    img_patch_mask=patch_passed_maps[imgind]

    img_shuffle_s,img_shuffle_e=passed_counts[imgind],passed_counts[imgind+1]

    # Batch shuffled inds are the same size as the passed mask/ subset of the mask that passed
    # from these inds we batch subsample
    batch_shuffle_inds=np.where(  (shuffled_inds[img_shuffle_s:img_shuffle_e]>=batch_s) \
                                  & (shuffled_inds[img_shuffle_s:img_shuffle_e]<batch_e)   )[0]


    passing_inds=np.where(img_patch_mask)[0] # actually passing

    ret_batch_mask=np.zeros(img_patch_mask.shape,dtype=bool)

    # From the true/passed elements choose the batch shuffled inds
    ret_batch_mask[ passing_inds[batch_shuffle_inds]  ]=True

    return ret_batch_mask


def get_how_many_patches(patcher_args,do_expand,patch_size,rand_passed,smaller_size,mask_dir,do_special):
    ind,count_pass=0,0

    filenames, test_path, groundtruth_path, type_set, patches_count, save_path=patcher_args
    lenF=len(filenames)

    rand_passed_maps=[]
    passed_counts=[]
    for find,filename in enumerate(filenames):
        patches=image_iterator_kiko(filename, test_path, groundtruth_path,
                type_set, -2 if mask_dir is not None else patches_count , save_path,
        do_expand=do_expand,patch_size=patch_size,PATCH_DIR=mask_dir,do_patch=False,do_special=do_special)

        rand_passed_maps.append(np.zeros((len(patches),),dtype=np.bool) )
        passed_counts.append(count_pass)

        for pind,(image,class_img,coords) in enumerate(patches):
            if rand_passed[ind]<=smaller_size[int(class_img)]:
                count_pass+=1
                rand_passed_maps[-1][pind]=True

            ind+=1

    passed_counts.append(count_pass)
    # print 'Passed {} patches'.format(ind)
    return count_pass,passed_counts,rand_passed_maps


def create_lmdbs(db_folder,phase_done,patcher_args,image_list_folder,image_prepend,
 image_width=None, image_height=None,smaller_size=[1,1,1],mean_name=None,
    patch_size=45,random_key_prepend=12,do_expand=True,
    save_base=True,batch_size=10**5,mask_dir=None,base_GBsize=None,DO_VALIDATION=False,do_special=None):

    smaller_size=np.array(smaller_size).astype(float)
    print ('reduce three classes by: ',smaller_size)

    filenames, test_path, groundtruth_path, type_set, patches_count, save_path=patcher_args

    image_db = lmdb.open(os.path.join(db_folder, '%s_db' % phase_done),
            map_async=True,
            sync=False,
            max_dbs=0)



    image_sum =None
    image_count=0
    patch_distribuition={}
    ind=-1

    continous_ind=-1

    #!!!
    # Think whether patches count below should be 0
    patches=image_iterator_kiko(filenames[0], test_path, groundtruth_path,
            type_set, 0, save_path,do_expand=do_expand,patch_size=patch_size,do_patch=False)

    len_random_index=len(patches)*len(filenames)
    single_rand_len=len(patches)
    len_int = len(str(len_random_index))
    if len_int < 8: len_int = 8

    # rand passed has rand value for every possible patch,
    # so as to exclude those smaller than rev_smaller_size
    rand_passed=np.random.rand(len_random_index)

    len_passed,passed_counts,patch_passed_maps=get_how_many_patches(patcher_args,do_expand,patch_size,
                rand_passed,smaller_size,mask_dir,do_special)

    print ('!!!!!!! ACTUAL_SIZE_OF_DATASET: ',len_passed )

    if base_GBsize is None:
        if phase_done=='val': base_GBsize=30
        elif phase_done=='train': base_GBsize=100
        image_db.set_mapsize(int(base_GBsize * 1024 ** 3))
    else:
        image_db.set_mapsize(int(base_GBsize*1024**3 ))

    # randInt stores all the indexes of the actual patches
    randInt=np.arange(len_passed)
    np.random.shuffle(randInt)
        
    seq_key=0
    for batch_ind_s,batch_ind_e in zip(range(0,len_passed,batch_size),\
        range(batch_size,len_passed+batch_size,batch_size)):

        saved_patches,image_count, image_sum = run_single_batch(\
            batch_ind_s,batch_ind_e,continous_ind, do_expand,image_sum, image_count,  ind,
            patch_distribuition, patch_size, patcher_args,  randInt, rand_passed,
                      save_base, smaller_size,mask_dir,do_special,patch_passed_maps,passed_counts,DO_VALIDATION)

        random.shuffle(saved_patches)
        print ('shuffled now  commiting... saving {} patches'.format(len(saved_patches)))

        image_db_trans = _get_image_transaction(image_db)

        for val in saved_patches:

            _db_just_put(image_db_trans, str(seq_key).zfill(len_int).encode('utf-8'),val)
            seq_key+=1


        # start here
        status = _db_commit_sync(image_db, image_db_trans)
        print( '\n===\ncommited batch {} of {}\n'.format(batch_ind_e//batch_size,len_passed/batch_size))

    # close databases
    image_db.close()
    # label_db.close()

    print ('Ran through {} images another key: {}'.format(image_count,seq_key))
    print ('Image patch distribuitions are: %s'%patch_distribuition)
    # save mean
    mean_image = (image_sum / image_count).astype('uint8')




    # _save_mean(mean_image, os.path.join(db_folder, '%s_mean.png' % phase))


    # with open(os.path.join(db_folder,'img_size.info') ,'wb') as f:
    #     pickle.dump(patch_size,f)
    #
    #
    # with open(os.path.join(db_folder,'data_%s.info'%phase) ,'wb') as f:
    #     pickle.dump((image_count,patch_distribuition),f)


    return True


def run_single_batch(batch_ind_start,batch_ind_end,continous_ind, do_expand,image_sum, image_count,  ind,
                     patch_distribuition, patch_size, patcher_args,  randInt, rand_passed,
                      save_base, smaller_size,mask_dir,do_special,patch_passed_maps,passed_counts,DO_VALIDATION ):

    filenames, test_path, groundtruth_path, type_set, patches_count, save_path=patcher_args
    fLen=len(filenames)
    arr_lis=[]


    for find, filename in enumerate(filenames):
        print ('Running on img:{}'.format(filename[filename.rindex(os.path.sep) + 1:]))
        sys.stdout.flush()

        passed_batch_mask=make_segmented_patches(patch_passed_maps,passed_counts,
                                                 randInt,batch_ind_start,batch_ind_end,find)

        patches = image_iterator_kiko(filename, test_path, groundtruth_path,
                                          type_set, -2 if mask_dir is not None else patches_count,
                                      save_path, do_expand=do_expand,patch_size=patch_size,
                                      PATCH_DIR=mask_dir,do_print=True if find==0 and batch_ind_start==0 else False,
                                      subsample_patch_mask=passed_batch_mask,do_special=do_special)
        inside_print_ind=0

        start_image_time = time.time()

        ## set Smaller Size Here
        for (image, class_img, coords) in patches:

            if image is None:
                continue

            middle_val=image[image.shape[0]//2,image.shape[1]//2]

            patch_distribuition[class_img] = patch_distribuition.get(class_img, 0) + 1
            image_sum = image if image_sum is None else image_sum+image
            image = image.astype(np.uint8)
            class_arr=np.array(int(class_img)).astype(np.uint8)

            packed_s = class_arr.tostring() + image.tostring()

            valid_s='Image: {},interim_val: {},true_val: {}, coords: {}'\
                .format(filename[filename.rindex(os.path.sep) + 1:],
                        middle_val,class_img,coords).encode('utf-8')

            if DO_VALIDATION:
                arr_lis.append( valid_s)
            else:
                arr_lis.append(packed_s)

            image_count += 1
            inside_print_ind+=1


        # raw_input('wait')
        # print ind, 'IND IS'
        print (('Commiting {} patches /from image {},'+
                ' {} of {}  /time: {} sec. {} last index')\
               .format(inside_print_ind, filename[
                filename.rindex(
                os.path.sep) + 1:], \
                find, fLen,
                time.time() - start_image_time,
                randInt[ind]
                ) )
        sys.stdout.flush()

    return arr_lis,image_count, image_sum





def make_db_folder(db_folder):
    if os.path.exists(db_folder):
        print( 'Recreating DB_Folder')
        # sys.exit(1)
        os.system('rmdir /S /Q "{}"'.format(db_folder))
        os.makedirs(db_folder)

    else:
        os.makedirs(db_folder)


def reduce_files(db_folder,phase_done):
    lm_dir=os.path.join(db_folder,'{}_db'.format(phase_done))

    newsize=_reduce_lmdbsize(lm_dir)

    print('new {} base size: {:.2f} gb'.format(phase_done,newsize))
