import cv2
import lmdb
import numpy as np

import os
import glob
import math
import itertools
import matplotlib.pyplot as plt
from keras import backend as K

import numpy as np
from os.path import join


def make_intmodel(model,lay,lay_name):
    intmodel=K.function([model.layers[0].input, K.learning_phase()],
                                  [lay.output])
    return intmodel

def draw_someoutput(matrices,savefold,chosen_cmap='viridis',
    draw_suffix='some_mats',no_axis=False,doShow=False):

    one_dim=int(math.ceil(math.sqrt(matrices.shape[0])) )
    one_allm = matrices.shape[1]+ (int(matrices.shape[1]/8.0) or 1)
    big_mat=np.zeros((one_dim*one_allm,one_dim*one_allm))+matrices.max()

    one_m=matrices.shape[1]
    range_m=np.arange(one_dim)*one_allm

    for mat,(onei,twoi) in zip(matrices,itertools.product(range_m,range_m)):
        big_mat[onei:onei+one_m,twoi:twoi+one_m]=mat


    print(matrices.shape,big_mat.shape,matrices.min())
    f=plt.figure(figsize=(20,20))
    if no_axis: plt.axis('off')
    plt.imshow(big_mat,cmap=chosen_cmap)
    plt.savefig(join(savefold, '{}_.png'.format(draw_suffix)),dpi=300)
    if not doShow: plt.close(f)

def iterate_indef(lmdb_txn ,batch_size ,img_width ,img_height,do_continuous=False,
                  do_key=False,smallify_base=None,split_map=False,
                  two_patch_instance=False,
                  do_single_patch=False,
                  dnn_class_annot=False,
                  do_padSq=0):

    from keras.utils import np_utils

    arrayfull_x=np.zeros((batch_size, img_width* img_height),dtype= np.uint8)
    arrayfull_y=np.zeros((batch_size,1), dtype= np.uint8)
    stop_ind=np.array([0],dtype=np.uint8)



    if do_continuous:
        stop_limit=257
    else:
        stop_limit=1

    while stop_ind[0]<stop_limit:
        for ind,(key, val) in enumerate(lmdb_txn.cursor()):
            arrayfull_x[ind% batch_size]=np.fromstring(val[1:],dtype=np.uint8)
            arrayfull_y[ind% batch_size]=np.fromstring(val[0:1],dtype=np.uint8)

            if smallify_base is not None and ind>=smallify_base:
                return

            if ind%batch_size ==batch_size -1:
                x=arrayfull_x.reshape((-1,img_width, img_height,1 )).astype(np.float32)
                y=np_utils.to_categorical(arrayfull_y)
                if do_single_patch:
                    xN,yN=x[:, :(img_width // 2), :, :], x[:, (img_width // 2):, :, :]
                    yNA,yNM,yNB=np.zeros(yN.shape),np.zeros(yN.shape),np.zeros(yN.shape)
                    yNA[yN==2]=1
                    yNB[yN==3]=1
                    yNM[yN==1]=1

                    y_patch = pad_one_makesq(np.concatenate((yNM,yNA,yNB),axis=-1))
                    x_patch = pad_one_makesq(xN)
                    yield x_patch, y_patch
                elif  dnn_class_annot and not split_map:

                    x=pad_one_makesq(x) if do_padSq>0 else x
                    yield x,y
                elif split_map:
                    xback,xax,xmil=np.zeros((3,)+x.shape,dtype=x.dtype)

                    if np.sum(x==127)!=0:
                        pass
                        print('this is the sth')
                        raise

                    xback[(x==255)| (x==3)]=1
                    xax[(x==128) | (x==2)]=1
                    xmil[(x==0) | (x==1)]=1

                    if two_patch_instance:
                        xmilx,xmily=xmil[:,:(img_width//2),:,:],xmil[:,(img_width//2):,:,:]
                        xaxX, xaxY = xax[:,:(img_width // 2), :, :], xax[:,(img_width // 2):, :, :]
                        xbackx, xbacky = xback[:,:(img_width // 2), :, :], xback[:,(img_width // 2):, :, :]

                        y_patch=np.concatenate((xmily,xaxY,xbacky),axis=-1)
                        y_patch=pad_one_makesq(y_patch)
                        x_patch=pad_one_makesq(np.concatenate((xmilx,xaxX,xbackx),axis=-1))
                        yield x_patch,y_patch

                    elif dnn_class_annot:

                        x_multi=np.concatenate((xmil, xax, xback), axis=-1)

                        x_multi=pad_one_makesq(x_multi) if do_padSq>0 else x_multi
                        yield x_multi,y
                    else:
                        yield x,y
                elif do_key:
                    yield int(key.decode('utf-8')),(x,y)
                else:
                    yield x,y

        stop_ind[0]+=1



def iterate_unsup_indef(lmdb_txn ,batch_size ,img_width ,img_height,do_continuous=False,
                        do_key=False,divide_some=None,smallify_base=None,split_map=False,split_list=[1,2,3]):

    arrayfull_x=np.zeros((batch_size, img_width* img_height),dtype= np.uint8)
    stop_ind=np.array([0],dtype=np.uint8)

    divider=divide_some if divide_some is not None else 1.0



    if do_continuous:
        stop_limit=257
    else:
        stop_limit=1

    while stop_ind[0]<stop_limit:
        for ind,(key, val) in enumerate(lmdb_txn.cursor()):
            arrayfull_x[ind% batch_size]=np.fromstring(val[1:],dtype=np.uint8)

            if smallify_base is not None and ind >= smallify_base:
                return

            if ind%batch_size ==batch_size -1:
                x=arrayfull_x.reshape((-1,img_width, img_height,1 )).astype(float)
                x = x.astype(np.float32)


                if split_map:
                    xm=np.zeros(x.shape,dtype=x.dtype)
                    xa=np.zeros(x.shape,dtype=x.dtype)
                    xback=np.zeros(x.shape,dtype=x.dtype)
                    xa[x==split_list[1]]=1
                    xm[x==split_list[0]]=1
                    xback[x==split_list[2]]=1

                    xbig=np.concatenate((xm,xa,xback),axis=-1 )
                # x[x == split_list[2]] = 0
                # x[x == split_list[1]] = 0.5
                # x[x == split_list[0]] = 1
                #
                # x /= divider

                if do_key:
                    yield int(key.decode('utf-8')),x
                elif split_map:
                    yield xbig
                else:
                    yield x

        stop_ind[0]+=1


def save_relevant(savedir,add_str,files=None,str_to_save=None,just_return=False,descriptive_text=None):
    '''
    Save these py files, to easily track the experiment configuration
    
    '''
    if files is None:
        files='models.py;dnn_workspace.py'.split(';')

    if str_to_save is not None:
        pass
    elif str_to_save is None:
        str_to_save=''
        for filen in files:
            with open(filen, 'r') as f:
                one_s = ''.join(f.readlines())


            str_to_save+='Saving file: {}\n'.format(filen)+'\n===========\n'+one_s+'\n'

    if just_return:
        return str_to_save

    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    with open(os.path.join(savedir,add_str+'.txt'),'w') as f:
        f.write('Model for Prediction: {} is: \n------------\n'.format(add_str))
        if descriptive_text is not None: f.write('Description: {}\n'.format(descriptive_text))
        f.write(str_to_save)

def use_best_model(shrt_str):
    starting=r'../../data/models'
    models=[]
    len_prep=len('weights_quickbig')
    for model in os.listdir(starting):
        if not model.startswith('weights_quickbig_{}'.format(shrt_str) ):
            continue
        smodel=model[model.rindex('weights_quickbig')+len_prep+6:]
        accur=float(smodel[3:3+6])
        models.append((os.path.join(starting,model),accur))

    models.sort(key=lambda x:-x[1])
    print('Using model: ',models[0][0])
    return models[0][0]
def use_num_model(shrt_str,num_model=None):
    starting=r'../../data/models'
    models=[]
    len_prep=len('weights_quickbig')
    for model in os.listdir(starting):
        if not model.startswith('weights_quickbig_{}'.format(shrt_str) ):
            continue
        smodel=model[model.rindex('weights_quickbig')+len_prep+6:]
        accur = float(smodel[3:3 + 6])

        num_ep=int(smodel[:2])
        if num_ep is not None and num_ep==num_model:
            return os.path.join(starting,model)

        models.append((os.path.join(starting,model),accur))

    models.sort(key=lambda x:-x[1])
    print('Using model: ',models[0][0])
    return models[0][0]
def draw_onfirst_epoch(epoch,logs,quick_str=None,big_img_size=None,
                       model=None,str_to_save=None,split_map=False,do_test=True):
    if epoch!=0 or quick_str is None or model is None or str_to_save is None or big_img_size is None:
        return
    do_whole_pic = True
    draw_img_ext = '*.tiff'
    if do_test:
        quick_test(model, quick_str+'_ep0',big_img_size, do_all=do_whole_pic,
                   draw_img_ext=draw_img_ext,split_map=split_map)
    save_relevant('saved_quickmodels', quick_str,str_to_save=str_to_save)

def merge_graychannels(x):
    # black,grey,white=x[...,0],x[...,1],x[...,2]
    # ret_x=np.zeros(x.shape[:-1],dtype=int)
    # ret_x[black==1]=0
    # ret_x[grey==1]=128
    # ret_x[white==1]=255
    ret_x=np.argmax(x,axis=-1)

    return ret_x

def pad_one_makesq(x):
    # x_s=tf.shape(x)
    new_x=np.pad(  x, ((0,0),(1,2),(1,2),(0,0)),'constant',constant_values=0 )
    # new_x=tf.transpose(new_x,[0,3,1,2])
    return new_x

def draw_one(pic,savetxt,cmap_need='gray'):
    plt.figure(figsize=(20, 20))
    plt.imshow(pic, cmap=cmap_need)
    plt.savefig( savetxt + '.png', dpi=300)
    plt.close()

def make_split_map(patch,split_map=[0,128,255],dopad=False):
    patch=patch.reshape((-1,45,45,1))
    xback, xax, xmil = np.zeros((3,) + patch.shape, dtype=patch.dtype)

    xmil[patch==split_map[0]]=1
    xax[patch==split_map[1]]=1
    xback[patch==split_map[2]]=1
    x_patch=np.concatenate((xmil,xax,xback),axis=-1).reshape((-1,45,45,3))
    if dopad: x_patch=pad_one_makesq(x_patch)

    return x_patch


def draw_autoenc_smaller(model,draw_loc,imgs,suff='',showNum=-1):

    for lay in model.layers[0:]:
        pass
        lay_name = lay.get_config()['name']
        if lay_name.startswith('convolution2d_1'):
            encD = make_intmodel(model, lay, lay_name)
            print(lay_name, lay.get_output_shape_at(0))
            # break
        if lay_name.startswith('convolution2d_7'):
            decD=make_intmodel(model,lay,lay_name)

    enc_x = encD([  imgs[0,:] , 0])[0]
    rec_x=decD([  imgs[0,:] ,0])[0]
    rec_x=merge_graychannels(rec_x)


    # print(imgs.shape,imgs[0])
    for ii in range(6):
        aein= (merge_graychannels(imgs[0,ii,1:-2,1:-2]))*127
        aeout=(   merge_graychannels(imgs[1,ii,1:-2,1:-2]))*127
        aerec=rec_x[ii,1:-2,1:-2]*127
        
        aednnins=[]


        cv2.imwrite(os.path.join(draw_loc, '{}_ae_{}_input.tif'.format(suff,ii))
                    ,aein)
        cv2.imwrite(os.path.join(draw_loc, '{}_ae_{}_output.tif'.format(suff,ii))
                    , aeout)
        cv2.imwrite(os.path.join(draw_loc, '{}_ae_{}_autoenc_recreation.tif'.format(suff,ii))
                    , aerec)

        for jj in range(40):
            aednn=((enc_x[ii,:,:,jj]/np.max(enc_x[ii,:,:,jj]))*255).astype(np.uint8)

            #print(enc_x.shape)
            # cv2.imwrite(os.path.join(draw_loc, '{}_ae_{}_dnnInput_{}.tif'.format(suff,ii,jj))
            #             , aednnins.append(aednn) )
            aednnins.append(aednn)


        aednnins=np.array(aednnins)
        draw_someoutput(aednnins[np.random.choice(40,9,replace=False)]
            ,draw_loc,chosen_cmap='viridis',
            draw_suffix='{}_summary_autoenc_pretrain_layer_img-{}'\
                .format(suff,ii),doShow=True if ii==showNum else False)

    return aednnins



def draw_autoenc(model,img_w,img_h,draw_loc,val_loc,suff=''):

    lmdbval_env = lmdb.open(val_loc)
    lmdbval_txn = lmdbval_env.begin()
    gen_val=iterate_indef(lmdbval_txn, 500, img_w * 2, img_h, two_patch_instance=True,
                  do_continuous=True, split_map=True)

    base_i=np.zeros((80,2,img_w+3,img_h+3,3))

    for (x,y),indv in zip(gen_val,range(40)):
        base_i[(indv*2)+1]=(x[10],y[10])
        base_i[(indv*2)]=(x[400],y[400])

    lmdbval_env.close()
    for lay in model.layers[0:]:
        pass
        lay_name = lay.get_config()['name']
        if lay_name.startswith('convolution2d_1'):
            encD = make_intmodel(model, lay, lay_name)
            print(lay_name, lay.get_output_shape_at(0))
            # break
        if lay_name.startswith('convolution2d_7'):
            decD=make_intmodel(model,lay,lay_name)



    draw_someoutput(merge_graychannels(base_i[:,0,:,:,:])
                    ,draw_loc,draw_suffix='{}_enc_inputs'.format(suff),
                    chosen_cmap='gray')

    draw_someoutput(merge_graychannels(base_i[:, 1, :, :, :]),
                    draw_loc, draw_suffix='{}_enc_annotation'.format(suff),
    chosen_cmap = 'gray')

    enc_x = encD([base_i[:,0,:,:,:], 0])[0]
    rec_x=decD([ base_i[:,0,:,:,:] ,0])[0]
    pass
    draw_someoutput(merge_graychannels(rec_x),draw_loc,
        draw_suffix='{}_enc_recreations'.format(suff),chosen_cmap='gray')

    for indP,enc_xx in enumerate(enc_x):
        enc_matx=np.transpose(enc_xx,(2,0,1))
        draw_someoutput(enc_matx, draw_loc,
            draw_suffix='{}_rec_mars_{}'.format(suff,indP), chosen_cmap='gray')


def quick_test(model, addition_str, big_image_size, do_all=False, draw_img_ext='*.tiff', test_batch_size=180,
               just_draw_from_prediction=None, draw_better_mistakes=True, split_map=False,
               conf_dikt=None,two_patch_instance=False):
    from .prediction_funcs import create_prediction, \
        color_images_in_folder, color_mistakes_in_folder

    deploy_batch_size = test_batch_size
    image_width=45 if conf_dikt is None or 'patch_size' not in conf_dikt else conf_dikt['patch_size']

    groundtruth_path = conf_dikt['groundtruth']
    save_path='#########'


    lookup_test_path =conf_dikt['interim']

    if conf_dikt is not None:
        lookup_test_path=conf_dikt['img_path']
        from_draw=conf_dikt['from']
        to_draw=conf_dikt['to']
    else:
        from_draw=0
        to_draw=42742 # Big Number i.e. all

    

    output_prediction_file = join(conf_dikt['prediction_saveloc'],'test_'+addition_str+'.txt')

    colored_images = join(conf_dikt['DNNOutput_saveloc'],'DNNOutput_'+addition_str+'/')
    colors_colored = (255, 0, 0), (0, 0, 255), (0, 255, 0),
    colors_grey = (0, 0, 0), (128, 128, 128), (255, 255, 255)
    color_mistake = (25, 255, 37)
    color_mistakes=[
        [(102,51,0),(255,178,102)],
        [( 0, 0,102), (102, 102, 255)],

        [(0,76,153),(102,178,255)],
    ]

    test_image_paths = sorted([filename for ind, filename in enumerate(glob.glob(os.path.join(lookup_test_path, draw_img_ext))) \
                        ])   \
    [from_draw:to_draw]
    print(test_image_paths,from_draw,to_draw,lookup_test_path)

    imgmask = np.zeros((big_image_size,big_image_size), dtype=bool)

    if big_image_size==2048:
        imgmask[250:1000, 100:750] = True
        imgmask[600:1000, 1100:1700] = True

    if do_all:
        imgmask[:,:]=True

    # just_draw_from_prediction=r'D:\code_projects\Neuron_fibers_workspace\prediction_files\quick_test__nubt_runid6.wholeset_autoenc.txt'
    if just_draw_from_prediction is None:
        print('Drawing on these: ',len(test_image_paths),test_image_paths)

        create_prediction(model,output_prediction_file,(test_image_paths,lookup_test_path, groundtruth_path, "all_test", 0, save_path),
                          patch_size=image_width,do_expand=True,deploy_batch_size=deploy_batch_size
                          ,subsample_mask=imgmask.flatten(),split_patch=split_map,
        do_dnnEnc=two_patch_instance)
    else:
        output_prediction_file=just_draw_from_prediction
        if not os.path.exists(output_prediction_file):
            raise Exception('Can\'t draw from nonexistant prediction')

    color_images_in_folder( colored_images,output_prediction_file,int(image_width/2),
                            big_image_size,*colors_grey,suf_img='-DNNOutput.tif'
        )

    color_mistakes_in_folder(  colored_images,groundtruth_path,
                               output_prediction_file,int(image_width/2),big_image_size
                               ,color_used=color_mistakes if draw_better_mistakes else color_mistake,
                               do_more_colors=draw_better_mistakes,
                               suf_img='-colored_DNNOutput.tif',
                                same_sizegnd_pred=True if two_patch_instance else False,
        )




def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    import tensorflow as tf

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__=='__main__':
    train_lmdb = r'###############/train_db'
    lmdbtrain_env = lmdb.open(train_lmdb)
    lmdbtrain_txn = lmdbtrain_env.begin()

    for ind,(key,(x,y)) in zip(range(100),iterate_indef(lmdbtrain_txn,60,45,45,True,True)):
        print('For ind {}, last key is {}'.format(ind+1,(key+1)/60 ))

