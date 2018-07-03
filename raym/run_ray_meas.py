import math
import sys
import os
import cv2
import time
import itertools
import matplotlib.cm as cm
from functools import cmp_to_key
import matplotlib.path as mpath
from numpy import array as arr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from sklearn import datasets, linear_model
from os.path import join






def compr(a,b):
    a_inner=int(a[0][10:])
    b_inner=int(b[0][10:])

    aroi=int(a[0][:9].replace('-',''))
    broi=int(b[0][:9].replace('-',''))

    if a[0][:9]==b[0][:9]:
        return a_inner-b_inner

    return aroi-broi


def euc(a,b):
    xdif=np.abs(a[0]-b[0])
    ydif=np.abs(a[1]-b[1])
#     print(xdif,ydif)
    return np.sqrt(xdif**2+ydif**2)

def ispix(img,x,y,size=2048):
    if x<0 or x>=size or y<0 or y>=size:
        return False
    return True


def get_angle_data(anglV,centV,pnts):

    #flip first
    anglV=math.pi-anglV

    newAng=((anglV+math.pi/2)%math.pi )-math.pi/2
    m1=math.tan(anglV-math.pi/2)
    m2=math.tan(newAng)

    b1=centV[1]-(m1*centV[0])
    b2s=pnts[:,1]-(m2*pnts[:,0])

    # print(anglV - (math.pi/2),anglV-math.pi/2,newAng)
    lenC=pnts.shape[0]
    intP=np.zeros((lenC,2))
    # line intersection. -> m1x+b1=m2x+b2

    intP[:,0]=(b2s-b1)/(m1-m2)

    intP[:,1]=(m2*intP[:,0])+b2s

    distOpp=euc(intP.T,pnts.T)
    distHip=euc(centV,pnts.T)


    minAngs=np.arcsin(distOpp/distHip)

    return m1,m2,b1,b2s,np.degrees(minAngs)

def travel(a,img,cent,polies,roi_border,roik,mielV,axonV,backV):



    xofs=a[0]-cent[0]
    yofs=a[1]-cent[1]

#     ystep=yofs/xofs if xofs!=0 else 1
#     xstep=1 if xofs!=0 else 0
#     if (cent[0]>a[0]):
#         xstep=-1
#     if cent[1]>a[1]:
#         ystep*=-1
    mofs=max(abs(xofs),abs(yofs))

    xstep=xofs/mofs
    ystep=yofs/mofs

#     if abs(ystep)>2:
#         xstep/=abs(ystep)
#         ystep=1

    curx,cury=a
    curix,curiy=int(a[0]),int(a[1])

    for iii in range(1):
        curx+=xstep
        cury+=ystep

    curix=int(round(curx))
    curiy=int(round(cury))

    while curix<imgw and curix>=0 and curiy<imgh and curiy>=0 and img[curiy,curix]==axonV:
        curix=int(round(curx))
        curiy=int(round(cury))
        curx+=xstep
        cury+=ystep

    curixT=int(round(curx))
    curiyT=int(round(cury))


    if  curixT<imgw and curixT>=0 and curiyT<imgh and curiyT>=0 and img[curiyT,curixT]==backV:

        isInside=True if roi_border[curiyT,curixT]==1 else False
        isInsideO=polies[roik].contains_point((curixT,curiyT))

        return -2,curixT,curiyT,isInside,isInsideO,euc(cent,a),-1

#     print('steps ',a,cent,ystep,xstep)
    while True:
        curix=int(round(curx))
        curiy=int(round(cury))
#         print(curix,curiy)

        if curix>=imgw or curiy>=imgh or curix<0 or curiy<0:
            return -1,curix,curiy,False,False,-1,-1

        if (img[curiy,curix]==backV or img[curiy,curix]==axonV ):
            sidex,sidey=abs(curx-a[0]),abs(cury-a[1])

            # isInside=np.any([polies[poli].contains_point((curix,curiy)) \
                             # for poli in polies if poli!=roik])
            isInside=True if roi_border[curiy,curix]==1 else False

            isInsideO=polies[roik].contains_point((curix,curiy))

#             print(cent,a)
#             sys.exit(-3)
            axonMdif=math.sqrt((sidex**2)+sidey**2)
#             if np.all(a==(648,329)):
#                 print('!!!!!!!!! here: ',curx,cury,axonMdif)

            res_sofar=axonMdif,curix,curiy,isInside,isInsideO,euc(cent,a)

            insideCl=-1
            for cl_size in range(cloud_size):
                curx+=xstep
                cury+=ystep
                curix=int(round(curx))
                curiy=int(round(cury))
                if ispix(img,curiy,curix) and img[curiy,curix]==mielV:
                    insideCl=cl_size
                    break

            return res_sofar+(insideCl,)



        curx+=xstep
        cury+=ystep

def calc_g(roiV,roiN,roiK,polies,roi_border,img,mielV,axonV,backV,minV,anglV,centV,imgofs=20):

	chosenRows=[indr for indr,val in enumerate(roiV) if  val[0].startswith(roiK)]
	lenC=len(chosenRows)




	resess=[travel((a,b),img,centV,polies,roi_border,roiK,mielV,axonV,backV) for a,b in \
				zip(roiN[chosenRows,0],roiN[chosenRows,1])]


	resess1=np.hstack((arr([r[1] for r in resess]).astype(int).reshape(-1,1),
					   arr([r[2] for r in resess]).astype(int).reshape(-1,1)))


	resess2=arr([r[0] for r in resess]).astype(float)

	resess3=arr([r[3] for r in resess]).astype(bool)
	resess4=arr([r[4] for r in resess]).astype(bool)
	resses5=arr([r[5] for r in resess]).astype(float)
	resses6=arr([r[6] for r in resess]).astype(int)

	m1,m2,b1,b2s,minDAngs=get_angle_data(anglV,centV,roiN[chosenRows])



	selectionInd=np.logical_not(resess3)  &np.logical_not(resess4) \
		 & (resses6==-1) & (resess2!=-1) & (minDAngs<minorAbsAngThres) & (resess2!=-2)

	#!!!!!!! axon is the radius
	axD=  resses5[ selectionInd ]# *2
	milD=resess2[selectionInd   ]


	axDFull=  resses5# *2
	milDFull=resess2

	resess1=resess1[selectionInd]

	if axD.shape[0]<1:
		if bool(int(confD['big_verbose'])): print('bad cell',roiK)
		return [None]*10
	# axT,milT=np.percentile(axD*2,10),np.percentile(milD,10)
	# return (axT,milT,axT/(axT+milT),axD/(axD+milD))

	rN=axD.shape[0]
	rNFull=axDFull.shape[0]

	isB=isH=isUnbalanced=False

	if np.sum(milDFull==-2)>miel_breakPerc*rNFull:
		isB=True
	if np.sum(milD>miel_highThres)>miel_highLong*rN:
		isH=True
	if roiK == '0132-0706':
		print('im here')
		pass
	if np.sum((milDFull/axDFull)>miel_axon_scale )>miel_axon_length*rNFull:

		isUnbalanced=True





	return  axD,axDFull,milD,milDFull,minV,isB,isH,isUnbalanced,np.sum( minDAngs<minorAbsAngThres),np.sum(selectionInd)

def run_onimg(imgname,imgsuf,based,doCorrected,mielV,axonV,backV,justSimple=None):
	if justSimple is not None:
		data = pd.read_csv(join(justSimple['imgfolder'], '{}_{}.csv'.format(justSimple['imgname'], justSimple['imgtype'])),
						   sep=',', header=None, names=['id', 'x', 'y'])
		data_t = pd.read_csv(join(justSimple['imgfolder'], '{}_{}_top.csv'.format(justSimple['imgname'], justSimple['imgtype']))
							 , sep=',', header=None, names=['id', 'cx', 'cy', 'minV', 'anglV'])
		img=justSimple['img']
	elif not doCorrected:
		img=cv2.imread(join(based_dnnout,'{}{}.tif'.format(imgname,imgsuf)),0)
		imgtype=caseType1
		baseMeas=baseDnnM
		# print(join(based_dnnout,'{}{}.tif'.format(imgname,imgsuf)))
	else:
		img=cv2.imread(join(based,'{}{}.tif'.format(imgname,imgsuf)),0)
		imgtype=caseType2
		baseMeas=baseDnnC

	if justSimple is None:
		data=pd.read_csv(join(baseMeas,'{}_{}.csv'.format(imgname,imgtype)),
			sep=',',header=None,names =['id','x','y'])
		data_t=pd.read_csv(join(baseMeas,'{}_{}_top.csv'.format(imgname,imgtype))
			,sep=',',header=None,names=['id','cx','cy','minV','anglV'])


	# ================================================================================================
	# ================================================================================================
	# ================================================================================================
	# ================================================================================================
	# ================================================================================================
	# ================================================================================================

	roiV=list(data.ix[:,:].values)


	roiV.sort(key= cmp_to_key(compr))
	roiN=np.array([row[-2:] for row in roiV]).astype(int)

	# maybe offset
	# roiN-=1

	rois=set([val[0][:9] for val in roiV  ])
	polygs={roi:np.zeros((0,2),dtype=int) for roi in rois}

	for row in roiV:
		roik=row[0][:9]
		polygs[roik]=np.vstack((polygs[roik],(row[1],row[2]) ))
	# print(list(polygs.values())[0])
	roiB=np.zeros((imgsize,imgsize),dtype=np.uint8)
	cv2.fillPoly(roiB, [p.astype(int) for p in polygs.values()], 1)

	for roi in rois:
		polygs[roi]=mpath.Path(polygs[roi])





	ges=np.zeros((0,))
	minors=np.zeros((0,))
	num_estimates=np.zeros((0,))

	bad_cells=np.zeros((0,2))
	bad_c_color=np.zeros((0,))

	metaInfo=[]
	c=0
	for indr,roi in enumerate(rois):
		# print(indr)

		chosenRowT=np.where(data_t.ix[:,'id']==roi)[0][0]

		centV=(data_t.ix[chosenRowT,'cx'],data_t.ix[chosenRowT,'cy'])
		minV=data_t.ix[chosenRowT,'minV']
		anglV=data_t.ix[chosenRowT,'anglV']*math.pi/180

		axD,axDF,milD,milDF,minorV,isB,isH,isUnbalanced,isAngle,numInstances=calc_g(roiV,roiN,roi,polygs,roiB,img,mielV,axonV,backV,minV,anglV,centV)

		# For different radial thinknesses
		#G_N=((axD)/((axD)+milD))

		if minorV is not None:
			# G_N=(minorV/2)/((minorV/2)+milD)
			G_N=((axD)/((axD)+milD))
		else:
			G_N=None
			isB,isH,isUnbalanced,isAngle=False,False,False,0
			axD,milD,minorV,mildF=np.nan,np.nan,np.nan,np.nan


		if GPlotByRoi and G_N is not None:
			G_N=(agg_func(G_N),)#(np.percentile(G_N,10),)
			M_N=(minorV,)
		elif G_N is not None:
			M_N=np.repeat(minorV,G_N.shape[0])


		if not np.any((isB,isH,isUnbalanced)) and G_N is not None:
			ges=np.concatenate((ges,G_N))
			minors=np.concatenate((minors,M_N))
			num_estimates=np.concatenate((num_estimates,(numInstances,)))
			c+=1
		elif G_N is not None:
			bad_c_color=np.concatenate((bad_c_color,('r' if isB else ('b' if isH else 'g') , ) ))
			bad_cells=np.vstack((bad_cells  ,np.array(centV).reshape(-1,2) ))

		if roi=='0132-0706':
			pass

		# print(roi,minorV,milD,milDF)
		metaInfo.append(  ( imgname,roi,isB,isH,isUnbalanced,isAngle,
				np.sum(milDF==-2),
				np.sum(milD==-2),
				np.sum(milD>miel_highThres),
				np.sum((milD/axD)>2 ),
				np.std(milD),
				milDF.shape[0] if milDF is not None else 0,
				milD.shape[0] if milDF is not None else 0,
				True if np.isnan(minorV) else False,
				agg_func(axD)
				,agg_func(milD)
				,G_N,
				'|'.join(map(str,milD)) if milDF is not None else None,
				'|'.join(map(str,axD)) if axDF is not None else None,
				'|'.join(map(str,milDF)) if milDF is not None else None,
				'|'.join(map(str,axDF)) if axDF is not None else None,
				minorV,
				centV[0],centV[1] )  )


	return metaInfo,ges,minors,num_estimates,c,bad_cells,bad_c_color

if __name__=="__main__":
	pass
	imgname = 'sp13750-img08'
	# based=r'D:\code_projects\dnn_seg\measuring\RayMeasuringPlugin\roi_ray_measuring\measuring_output_ml02rerun\ROI_Imagej_MeasurementDir'
	# imgsuf = 'DNN'
	# imgsize=2048
	# imgw = imgh = imgsize
	# cloud_size = 15
	# minorAbsAngThres = 45
	# GPlotByRoi=True
	# confD={}
	# confD['big_verbose']=1
	# #Cond parameters
	# miel_breakPerc =  0.15
	# miel_highThres = 80
	# miel_highLong =  0.3
	# miel_axon_scale =  6
	# miel_axon_length = 0.05
	# agg_func=np.mean
	# img=cv2.imread(r'D:\code_projects\dnn_seg\data\17_06_20_nubt_mlset02set_autoenc_1ep_justgray\sp13750-img08-interim-gray_predictions-g_nubt.00-0.8856.mlset02set_autoenc_1ep.tif',0)

	# print('Working on ... ', imgname, flush=True)
	# roiIndex, ges, minors, num_estimates, c, bad_cells, bad_c_color = run_onimg(
		# imgname, imgsuf, based, True, 0,128,255,justSimple={'imgname':imgname,'imgfolder':based,'imgtype':imgsuf,'img':img})
	# sys.exit('')

confD={}
for farg in sys.argv:
	if farg.startswith('--'):
		(arg,val) = farg.split("=")
		arg = arg[2:]
		confD[arg]=val


baseDnnM=join(confD['raymeas_save_dir'],
			  confD['base_meas_roiDir'])#'dnn_autoenc_mlset02_1ep/'

baseDnnC=baseDnnM#'cor_autoenc_mlset02_1ep/'


based=confD.get('case2_picDir',None)#r'../data/Corrected_mlset02_used'
based_dnnout=confD['case1_picDir'] #r'../data/17_06_20_nubt_mlset02set_autoenc_1ep'


imgsize=int(confD['img_size'])
imgw=imgh=imgsize




# Stop Params
cloud_size=int(confD['mielin_hole_size']) # 15
minorAbsAngThres=int(confD['roi_angle_offset']) # 45



# Cond parameters
miel_breakPerc=float(confD['mielin_break_length']) # 0.1
miel_highThres= int(confD['mielin_high_thres']) #80
miel_highLong= float(confD['mielin_high_length']) #0.3
miel_axon_scale=float(confD['mielin_axon_scale']) # 0.2
miel_axon_length=float(confD['mielin_axon_length']) # 0.5
# Meas Info
meas_dir=confD['raymeas_save_dir']
add_str= confD['raymeas_add_id'] #'meta_info_mls02_gen'

caseType1=confD['case_id_add1']
caseType2=confD.get('case_id_add2',None)

cols_meta=('imgname,roi,isB,isH,isUnbalanced,insideAngle'+
	',brokenMielC,brokenMielAC,highMielC,bigAxonC,stdMiel,beforeStopC,afterStopC,'+
	'goodStopC,axonV,mielV,Gratio,milList,axList,milFList,axFList,minorV,cX,cY').split(',')

# False for all measurements
GPlotByRoi=True

from functools import partial

agg_func=np.mean #partial(np.percentile,q=10)



# sys.exit("")

visrange1=int(confD['vis_range1'])
visrange2=int(confD['vis_range2'])

vr1=(1,2,3)
vr2=(1,2,3)
if visrange1==1:
	vr1=(0,128,255)
	
if visrange2==1:
	vr2=(0,128,255)

times=time.time()

gesFdnn=np.zeros((0,))
minorsFdnn=np.zeros((0,))
num_Festimates=np.zeros((0,))

roisdnn=[]
for imgn in os.listdir(based_dnnout):
	if  'sp' not in imgn or not imgn.endswith('.tif'):
		continue

	# print(os.listdir(based_dnnout))
	# continue
	imgname=imgn[imgn.rindex('sp'):imgn.rindex('img')+5]
	imgsuf=imgn[imgn.rindex('img')+5:imgn.rindex('.tif')]
	print('Working on ... ',imgname,flush=True)

	roiIndex,ges,minors,num_estimates,c,bad_cells,bad_c_color=run_onimg(
				imgname,imgsuf,based_dnnout,False,vr1[0],vr1[1],vr1[2])

	roisdnn.extend(roiIndex)
	gesFdnn=np.concatenate((gesFdnn,ges))
	minorsFdnn=np.concatenate((minorsFdnn,minors))
	num_Festimates=np.concatenate((num_Festimates,num_estimates))


if caseType2 is not None:
	gesFcor=np.zeros((0,))
	minorsFcor=np.zeros((0,))


	roiscor=[]
	# print(os.listdir(based))
	for imgn in os.listdir(based):
		# 'sp13938-img07' in imgn or
		if  not 'sp' in imgn or not imgn.endswith('.tif'):
			continue
		# print(imgn)
		imgname=imgn[imgn.rindex('sp'):imgn.rindex('img')+5]
		imgsuf=imgn[imgn.rindex('img')+5:imgn.rindex('.tif')]
		print('Working on ... ',imgname,flush=True)
		roiIndex,ges,minors,num_estimates,c,bad_cells,bad_c_color=run_onimg(
							imgname,imgsuf,based,True,vr2[0],vr2[1],vr2[2])

		roiscor.extend(roiIndex)
		gesFcor=np.concatenate((gesFcor,ges))
		minorsFcor=np.concatenate((minorsFcor,minors))



import pickle
import pandas as pd

# print('im here')

pd.DataFrame(data=roisdnn,columns=cols_meta).to_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType1,add_str)))


if caseType2 is not None: pd.DataFrame(data=roiscor,columns=cols_meta).to_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType2,add_str)))

