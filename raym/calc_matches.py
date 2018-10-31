# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from functools import cmp_to_key
import pickle
import os
import cv2
from os.path import join


def euc(a,b):

    try:
        return np.sqrt(  ( (a[0]-b[0])**2) +(a[1]-b[1])**2)
    except:
        print(a.shape,b)
        raise


def avg_area_match(roi1,roi2,ser_roiA,ser2_roiA  ):
    A[A!=0]=0
    B[B!=0]=0

    # print(ser_roiA[roi1])

    cv2.fillPoly(A, [ser_roiA[roi1].astype(int) ], 1)
    cv2.fillPoly(B, [ser2_roiA[roi2].astype(int) ], 3)
    c=A+B

    first,second,unin=np.sum(A==1),np.sum(B==3),np.sum(c==4)

    return first,second,unin,((unin/first)+(unin/second))/2



def compr(a,b):
    a_inner=int(a[0][10:])
    b_inner=int(b[0][10:])

    aroi=int(a[0][:9].replace('-',''))
    broi=int(b[0][:9].replace('-',''))

    if a[0][:9]==b[0][:9]:
        return a_inner-b_inner

    return aroi-broi


def make_polygDikt(imgtype):
    polygs={}
    for imgname in imgnames:
        data=pd.read_csv(join(base_meas,'{}_{}.csv'.format(imgname,imgtype)),
                         sep=',',header=None,names =['id','x','y'])
        # print('!!!!!!!!!!!!!!!!!!!!!',data.shape)
        roiV=list(data.ix[:,:].values)


        roiV.sort(key= cmp_to_key(compr))
        roiN=np.array([row[-2:] for row in roiV]).astype(int)

        # maybe offset
        # roiN-=1

        rois=set([val[0][:9] for val in roiV  ])
        polygs.update({(imgname,roi):np.zeros((0,2),dtype=int) for roi in rois})

        for row in roiV:
            roik=str(row[0][:9])
            polygs[(imgname,roik)]=np.vstack((polygs[(imgname,roik)],(row[1],row[2]) ))

    return polygs



def makeMatch(dnn_imgD,cor_imgD,dnngP,corgP,dnn_rdikt,cor_rdikt):
    nonM=[]
    matchD={}

    for imgi,img in enumerate(imgnames):
        # if imgi==1:
        # break
        rois_dnni=dnn_imgD[img]
        rois_cori=cor_imgD[img]
        dnni=dnngP[rois_dnni]
        cori=corgP[rois_cori]

        dnniN=dnni.loc[:,['cX','cY']].values
        coriN=cori.loc[:,['cX','cY']].values

        print('working on: ',img,', img index: ',imgi,flush=True)
        for indr,(roiI,roiR) in enumerate(zip(cori.roi,coriN)):
            best10i=np.argsort(euc(dnniN.T,roiR) )[:20]
            # print([x for i,x in zip(range(10),dnn_rdikt.keys())],euc(dnniN.T,roiR).shape ,dnniN.shape)
            # sys.exit(')')

            best10=dnniN[best10i]
            best10R=dnni.iloc[best10i]

            # if indr%40==3: print('at roi: ',indr,cori.shape)
            done=False
            for indb,(bI,best) in enumerate(zip(best10R.roi,best10)):
                try:
                    first,second,inter,x=avg_area_match((img,roiI),(img,bI),cor_rdikt,dnn_rdikt)
                except:
                    print(best,best10R,pd.notnull(dnni.roi))
                    raise
                union=first+second+inter
                # print(first,second,inter,union)
                # if indr==10: sys.exit()


                if inter/first>areaThres and euc(best,roiR)<distThres:
                    matchD[frozenset((img,roiI,bI))]=[img,roiI,bI,first,second,inter,inter/first,euc(best,roiR),roiR,best]
                    done=True
                    break

            if not done:
                nonM.append((img,roiI))
            # nonMC.append((img,

    return nonM,matchD

def draw_rois(nonm,basec,fold):
    # nonM=np.array(nonm)
    imgD={}
    for img in set([i[0] for i in nonm]):
        imgD[img]=cv2.imread(os.path.join(basec,img+'-interim-colored_train_mistakes.tif'))

    for img,roiI,roiR in nonm:
        cv2.circle(imgD[img],tuple(map(int,(roiR))),10,(0,255,0),3)

    for imgn,img in imgD.items():
        cv2.imwrite(fold+'/img-mismatch-{}.tif'.format(imgn),img)

def get_listC(lis):
    all=[]
    # print(lis[0][0])
    for i in lis:
        if type(i)!=str and np.isnan(i):
            all.append(np.nan)
            continue
        str_farr=i.replace('[','').replace(']','').split()
        lis_farr=list(map(float,str_farr))
        all.append(np.array(lis_farr))
    return all



def getCBigger(lisS,val):
    ret=np.zeros((len(lisS),),dtype=int)
    for ind,lis in enumerate(lisS):
        ret[ind]=np.sum(lis>val)
    return ret

def getMielAxFac(mil,ax,fac):
    ret=np.zeros((len(mil),),dtype=int)
    for ind,(mlis,alis) in enumerate(zip(mil,ax)):
        ret[ind]=np.sum( (mlis/alis)>fac )
    return ret

# sys.exit('')

confD={}
for farg in sys.argv:
    if farg.startswith('--'):
        (arg,val) = farg.split("=")
        arg = arg[2:]
        confD[arg]=val


spcase1_add=confD['case_id_add1']
spcase2_add=confD['case_id_add2']

base_meas=join(confD['raymeas_save_dir'],confD['base_meas_roiDir'])  #'autoenc_mlset02_1ep/'



imgnames=list(set([img[img.rindex('sp'):img.rindex('img')+5] for img in os.listdir(confD['case1_picDir']) if img.endswith('.tif')]))
print("Working on these images: ",imgnames)


distThres=int(confD['base_dist_thres']) #80
areaThres=float(confD['base_area_thres']) #0.51

strictAThres=float(confD['strict_area_thres']) #0.6
strictDThres=int(confD['strict_dist_thres']) #20

meas_dir=confD['raymeas_save_dir']
add_str= confD['raymeas_add_id'] #'meta_info_mls02_gen'

caseType1=confD['case_id_add1']
caseType2=confD['case_id_add2']

dnn=pd.read_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType1,add_str)  ))  # meas/meta_cor_mls02_autoenc_ep1.csv
cor=pd.read_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType2,add_str)  ))
stat_str=confD['match_add_id']# 'AS_mls02_min_51p_autoenc_1ep'


imgs=int(confD['img_size'])

A,B=np.zeros((imgs,imgs),dtype=int)	,np.zeros((imgs,imgs),dtype=int)


dnnG=[float(i['Gratio'][1:-2]) if type(i['Gratio'])==str and i.Gratio[0]=='(' else np.nan for ind,i in dnn.iterrows() ]
corG=[float(i['Gratio'][1:-2]) if type(i['Gratio'])==str and i.Gratio[0]=='(' else np.nan for ind,i in cor.iterrows() ]

dnn['Gratio']=dnnG
cor['Gratio']=corG



# match roi polygons to rois
# if os.path.exists(join(meas_dir,'roi_{}.dikts'.format(stat_str))):
    # with open(join(meas_dir, 'roi_{}.dikts'.format(stat_str)),'rb') as f:
        # dnn_roiA,cor_roiA=pickle.load(f)
# else:
print('making polygs...',flush=True)
dnn_roiA=make_polygDikt(spcase1_add)
cor_roiA=make_polygDikt(spcase2_add)
with open(join(meas_dir,'roi_{}.dikts'.format(stat_str)),'wb') as f:
	pickle.dump((dnn_roiA,cor_roiA),f)





dnng=dnn.loc[pd.notnull(dnn['minorV'])]
corg=cor.loc[pd.notnull(cor['minorV'])]


print('Whole sizes: dnn:{} cor: {}, Measurable sizes: dnn: {},cor: {}'.format(dnn.shape[0],cor.shape[0],dnng.shape[0],corg.shape[0]))


dnn=cor=None

# from here operating on measurable images
dnn_imgs_dikt = {img: dnng.imgname == img for img in set(dnng.imgname)}
cor_imgs_dikt = {img: corg.imgname == img for img in set(corg.imgname)}



print('making matches...',flush=True)
nonM,matchD=makeMatch(dnn_imgs_dikt,cor_imgs_dikt,dnng,corg,dnn_roiA,cor_roiA)


print('non matches count than matches count: ',len(nonM),len(matchD),flush=True)



dataM=pd.DataFrame(data=list(matchD.values()),columns='img,roic,roid,areac,aread,intersect,area-fac,cent-dist,centCor,centDnn'.split(','))
dataM=dataM.assign(ismatch=[ 1 if row['area-fac']>strictAThres and row['cent-dist']<strictDThres else 0 for ind,row in dataM.iterrows() ])

dataM.to_csv(join(meas_dir,'match_info_{}.csv'.format(stat_str)))

# saving non-matches
corgInd=corg.set_index(['imgname','roi'])
nonMI=pd.MultiIndex.from_tuples(nonM,names='imgname,roi'.split(','))
corgInd.loc[nonMI,'cX cY minorV'.split()].to_csv(join(meas_dir,'cor_nonm_{}.csv'.format(stat_str)))





# sys.exit(' till now everything is dome')

