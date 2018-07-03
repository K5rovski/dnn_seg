

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn import datasets, linear_model
import time
import matplotlib as mp
import sys
from os.path import join


def fit_reg(xses,yses):
	meanX,meanY=np.mean(xses,axis=0),np.mean(yses,axis=0)
	
	# print(meanX.shape,meanY.shape)
	# print(((xses)))
	B1=np.sum((xses-meanX)*(yses-meanY))/np.sum( (xses-meanX)*(xses-meanX)  )
	B0=meanY-(B1*meanX)
	
	
	return B0,B1

	
def reg_predict(coefs,xses):
	return (coefs[1]*xses)+coefs[0]

def draw_matches(dnnI,cornm,match_tab,dissb_tab,dissu_tab,match_tabDDik):


    good_m=set([(row.img,row.roid)
                for ind,row in match_tab.iterrows() if row.ismatch])
    discutable_m=set([(row.img,row.roid)
                      for ind,row in match_tab.iterrows() if not row.ismatch])

    dissb_cond=set([(row.imgname,row.roi)
                    for ind,row in dissb_tab.iterrows()])

    dissu_cond=set([(row.imgname,row.roi)
                    for ind,row in dissu_tab.iterrows()])


    dissy_c,dissb_c,red_c,green_c,pink_c=0,0,0,0,0

    mapD=np.zeros(np.sum(np.logical_not(dnnI.goodStopC.values)),dtype=int)
    indM=0
    match_cor=[]
    for ind,row in dnnI.iterrows():


        # reverse condition
        if  row.goodStopC:
            pass

            if (row.imgname,row.roi) in good_m: print('bad roi is match')
            continue
        elif (row.imgname,row.roi) in dissb_cond:
            pass
            grat=float(row.Gratio.replace('(','').replace(',)',''))
            #plt.scatter(row.minorV,grat,facecolors='none', edgecolors='yellow',)#label='dnn measurements')
            dissy_c+=1
            mapD[indM]=mapVals[0]


        elif (row.imgname,row.roi) in dissu_cond:
            pass
            grat=float(row.Gratio.replace('(','').replace(',)',''))
            #plt.scatter(row.minorV,grat,facecolors='none', edgecolors='blue',)#label='dnn measurements')
            dissb_c+=1
            mapD[indM]=mapVals[1]

        elif (row.imgname,row.roi) in discutable_m:
            pass
            grat=float(row.Gratio.replace('(','').replace(',)',''))
            # plt.scatter(row.minorV,grat,facecolors='none', edgecolors='pink',)#label='dnn measurements')
            mapD[indM]=mapVals[2]
            pink_c+=1

        elif (row.imgname,row.roi) not in good_m:
            pass

            grat=float(row.Gratio.replace('(','').replace(',)',''))
            # plt.scatter(row.minorV,grat,facecolors='none', edgecolors='red',)#label='dnn measurements')
            red_c+=1
            mapD[indM]=mapVals[3]
        elif (row.imgname,row.roi) in good_m:
            pass
            mapD[indM]=mapVals[4]
            match_cor.append((row.imgname, match_tabDDik[(row.imgname,row.roi)],ind) )

            grat=float(row.Gratio.replace('(','').replace(',)',''))
            # plt.scatter(row.minorV,grat,facecolors='none', edgecolors='green',)#label='dnn measurements')
            green_c+=1
        else:
            raise


        if not row.goodStopC:
            indM+=1

    # print("yellow c: {}, blue c: {},red_c: {}, green_c: {}, pink_c: {}, indM: {}".format(dissy_c,dissb_c,red_c,green_c,pink_c,indM))

    return mapD,match_cor

confD={}
for farg in sys.argv:
    if farg.startswith('--'):
        (arg,val) = farg.split("=")
        arg = arg[2:]
        confD[arg]=val

meas_dir=confD['raymeas_save_dir']
meas_add_str= confD['raymeas_add_id'] #'meta_info_mls02_gen'

match_stat_str=confD.get('match_add_id',None)# 'AS_mls02_min_51p_autoenc_1ep'
postconds_id_str=confD.get('postconds_id',None)

caseType1=confD['case_id_add1']
caseType2=confD.get('case_id_add2',None)

dotwoCases=True if caseType2 is not None else False

mapRoiValT=int(confD.get('map_roi_thres',1))
case2doMatch=int(confD.get('match_rois_sp2',0))


title_text=confD.get('title_text',None)
total_str=confD.get('total_str','Total')
afterc_str=confD.get('afterc_str','after conds.')


print(case2doMatch,confD.get('match_rois_sp2'))

# !!!!!!!!!!!!!!!!!!!!!!
# if caseType2 is  None:
    # raise Exception('casetype2 not possible for now !!')

if caseType2 is None:
    # caseType2=confD['case_id_add1']
    dnnI=pd.read_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType1,meas_add_str)))
    corI=None
	
    # TwoRun Files
    matchF,cor_nm=[pd.DataFrame()]*2
    diss_tab,dissb_tab,dissu_tab=[set([])]*3

else:

    corI=pd.read_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType2,meas_add_str)))
    dnnI=pd.read_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType1,meas_add_str)))

    # Match Files
    matchF=pd.read_csv(join(meas_dir,'match_info_{}.csv'.format(match_stat_str)))
    cor_nm=pd.read_csv(join(meas_dir,'cor_nonm_{}.csv'.format(match_stat_str)))


    # Roi Conds Files
    diss_tab=pd.read_csv(join(meas_dir,'discarded_dnn_{}.csv'.format(postconds_id_str)))
    dissb_tab=pd.read_csv(join(meas_dir,'discardedb_dnn_{}.csv'.format(postconds_id_str)))
    dissu_tab=pd.read_csv(join(meas_dir,'discardedu_dnn_{}.csv'.format(postconds_id_str)))


mapVals=(1,2,3,4,5, 0)

if caseType2 is not None: cor=np.array([(float(g.replace('(','').replace(',)','')),m ) for g,m in  zip(corI.Gratio.values,corI.minorV.values) if isinstance(g,str)])
dnn=np.array([(float(g.replace('(','').replace(',)','')),m ) for g,m in  zip(dnnI.Gratio.values,dnnI.minorV.values) if isinstance(g,str) ])




areaF=float(confD['strict_area_thres']) #0.6
centD=int(confD['strict_dist_thres']) #20

match_tab=matchF
if caseType2 is not None:
    match_tab.ismatch=((matchF['area-fac']>areaF) & (matchF['cent-dist']<centD))
    # print('all: , matches: ',match_tab.shape[0],np.sum(match_tab.ismatch),dnn.shape[0],dissb_tab.shape[0],dissb_tab.shape[0])
    # match_tab=match_tab.loc[match_tab.ismatch]





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# import collections
# print([(i,c) for i,c in collections.Counter([(row.img,row.roid) for ind,row in match_tab.iterrows()]  ).items() if c>1 ])


match_tabDDik={(row.img,row.roid):row.roic for ind,row in match_tab.iterrows()}


matchC_set=set([(row.img,row.roic) for ind,row in match_tab.iterrows()])
if caseType2 is not None: allC_set=set([(row.imgname,row.roi) for ind,row in corI.iterrows()])




# regrtdnn=fit_reg(dnn[:,1].reshape(-1, 1), dnn[:,0].reshape(-1, 1))

if caseType2 is not None: 
	regrtcor=fit_reg(cor[:,1].reshape(-1, 1), cor[:,0].reshape(-1, 1))
	matchCM=np.array([True if (row.imgname,row.roi) in matchC_set else False for ind,row in corI.iterrows() if not row.goodStopC ])
	corsave=cor
	if case2doMatch: cor=cor[matchCM]
	print(case2doMatch,cor.shape,np.sum(matchCM))


# regrtcorG = linear_model.LinearRegression()
# regrtcorG.fit(cor[matchCM,1].reshape(-1, 1), cor[matchCM,0].reshape(-1, 1))

dnnregL=np.linspace(np.min(dnn[:,1]),np.max(dnn[:,1]),400)
if caseType2 is not None: corregL=np.linspace(np.min(cor[:,1]),np.max(cor[:,1]),400)






mp.rcParams.update({'font.size' :25,'lines.linewidth':3,'axes.labelsize':35})
mp.rc('xtick', labelsize=35)
mp.rc('ytick', labelsize=35)

plt.ylabel('Avg-G-ROI')
plt.xlabel('Minor (pixels)')

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




if caseType2 is not None:
	mapD,match_cor=draw_matches(dnnI,cor_nm,match_tab,dissb_tab,dissu_tab,match_tabDDik)
	#Maybe bad print!!!!! print('Dnn roi without post. conds== ',np.sum(mapD>mapVals[1]))
	# plt.scatter(dnn[mapD==4,1],dnn[mapD==4,0],facecolors='none', edgecolors='green',label='DNN Roi - Match')
	
	plt.scatter(cor[:,1],cor[:,0],facecolors='red', edgecolors='none',label='ROIs No_{0} = {1}'.format(caseType2,cor.shape[0]))
	plt.scatter(dnn[mapD>mapVals[mapRoiValT],1],dnn[mapD>mapVals[mapRoiValT],0],
	facecolors='none', edgecolors='green',label='ROIs No_{0} = {1}'.format(caseType1, np.sum(mapD>mapVals[mapRoiValT]) ))#afterc_str if dotwoCases else total_str ))
else:
	mapD=np.ones((dnn.shape[0],))*100
	plt.scatter(dnn[mapD>mapVals[mapRoiValT],1],dnn[mapD>mapVals[mapRoiValT],0],facecolors='none', edgecolors='green',label='{} Rois'.format(caseType1))


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


regrtdnnG=fit_reg(dnn[mapD>mapVals[mapRoiValT],1].reshape(-1, 1), dnn[mapD>mapVals[mapRoiValT],0].reshape(-1, 1))









# if caseType2 is not None: print('corrected coun: ',np.sum(matchCM),np.sum(mapD>mapVals[mapRoiValT]))



first_label='Regression {} {}'.format(caseType2,total_str)
second_label='Regression {} {}'.format(caseType1, afterc_str if dotwoCases else total_str )
# third_label='Reg. CORR Match'



plt.xlim([0,120])
plt.ylim([0,1])
# plt.xscale('log')

if caseType2 is not None: plt.plot(corregL,reg_predict(regrtcor,corregL.reshape(-1,1)) ,c='red',label=first_label)
plt.plot(dnnregL,reg_predict(regrtdnnG,dnnregL.reshape(-1,1)) ,c='green',label=second_label)


if caseType2 is not None:
	meanCor=np.mean(cor[:,0])
	plt.plot([0,120],np.repeat(meanCor,2),color='red',ls='--',
	label='Mean {0} {1}'.format(caseType2,total_str))
	
	plt.text(0,meanCor+0.02,'{0:.2f}'.format(meanCor),color='red',size=20)
meanDnn=np.mean(dnn[mapD>mapVals[mapRoiValT],0])
plt.plot([0,120],np.repeat(np.mean(dnn[mapD>mapVals[mapRoiValT],0]),2)
		,color='green',ls='--',
		label='Mean {0} {1}'.format(caseType1, afterc_str if dotwoCases else total_str ))

plt.text(0,meanDnn-0.05,'{0:.2f}'.format(meanDnn),color='green',size=20)
		



if caseType2 is not None:
	plt.text(5,0.95,'|Slope_{1} - Slope_{0}| / Slope_{0} = {2:.2f}%'.format(caseType2,caseType1,100*abs(regrtcor[1]-regrtdnnG[1])/regrtcor[1] ), \
			 fontdict={'size':25}, bbox=dict(facecolor='white', edgecolor='none'))

	plt.text(5,0.87,'|Mean_{1} - Mean_{0}| / Mean_{0} = {2:.2f}%'.format(caseType2,caseType1,100*abs(meanCor-meanDnn)/meanCor ), \
			 fontdict={'size':25}, bbox=dict(facecolor='white', edgecolor='none'))
	# plt.text(22,0.04,'Rois: {}={},{}={}'.format(caseType1,np.sum(mapD>mapVals[mapRoiValT]),caseType2,cor.shape[0]), \
			 # fontdict={'size':35})


		
		
			
reg_info=''
if caseType2 is not None: reg_info='\n{}: {} * x + {}'.format(first_label,regrtcor[1],regrtcor[0])
reg_info+='\n{}: {} * x + {}'.format(second_label,regrtdnnG[1],regrtdnnG[0])
# reg_info+='\n{}: {} * x + {}'.format(third_label,regrtcorG.coef_,regrtcorG.intercept_)

print(reg_info,flush=True)

twocase_str='Comparison between {} and {} cases,\n conds applied on {} case'.format(caseType1,caseType2,caseType1)
onecase_str='All ROIs in {} case'.format(caseType1)

title_text=title_text if title_text else (twocase_str if dotwoCases else onecase_str)
title_text='Total {0} ROIs= {2},  Measured {0} ROIs= {3},  No.{1} ROIs={4}'.format(caseType1,caseType2,np.sum(mapD>0),np.sum(mapD>mapVals[mapRoiValT]),cor.shape[0])
title_text=''
plt.title('{}'.format(title_text))   #G minor plot,\n
# plt.title('G minor plot'+reg_info)
plt.legend(loc='lower right')

plt.figure()
# plt.subplot(211)
plt.title(confD['hist_title'])
plt.xlabel('Minor size (px)')
plt.ylabel('Frequency')

plt.hist(dnn[mapD>mapVals[mapRoiValT],1],range=[0,100],bins=200,alpha=0.8,ec='green',lw=3,fc='none',label='{} hist, {} ROIs.'.format(caseType1,np.sum(mapD>mapVals[mapRoiValT]) ))

plt.hist(cor[:,1],range=[0,100],bins=200,ec='red',alpha=0.5,fc='none',lw=5,ls='solid',label='{} hist, {} ROIs.'.format(caseType2,cor.shape[0]))

plt.legend()

plt.figure()
binC=50
width=8

match_cor=np.array(match_cor)
print(match_cor)
good_corD={setc:ind for ind,setc in enumerate([(row.img,row.roid)
                for ind,row in match_tab.iterrows() if row.ismatch])
				}
good_corC={setc:ind for ind,setc in enumerate([(row.img,row.roic)
                for ind,row in match_tab.iterrows() if row.ismatch])
				}
				
mapCvals=sorted([(ind,good_corC[(row.imgname,row.roi)]) for ind,row in corI.iterrows() \
	if (row.imgname,row.roi) in good_corC],key=lambda x:x[1])
mapDvals=sorted([(ind,good_corD[(row.imgname,row.roi)]) for ind,row in dnnI.iterrows() \
	if (row.imgname,row.roi) in good_corD],key=lambda x:x[1])

dnnMM=dnnI.loc[np.logical_not(dnnI.goodStopC),'milList,axList'.split(',')]
# dnnMM=dnnMM.loc[:,'milList,axList'.split(',')]

# corMM=corI.iloc[mapCvals]
# corMM=corMM.loc[:,'milList,axList'.split(',')]


print('!!!!!!!!!!!!!!!!!!!!!!!!!')

results_roi=np.array([ (
2* (np.average(list(map(float,dr[0].split('|')) ))  +np.average(list(map(float,dr[1].split('|')) )) ) ,
   np.average(list(map(float,dr[0].split('|'))) ) ) for ind,dr in dnnMM.iterrows()])



ofs=(np.linspace(0,50,51)*  ((np.max(results_roi[:,0])-np.min(results_roi[:,0]))/50)) +np.min(results_roi[:,0])
print(ofs,((np.max(results_roi[:,0])-np.min(results_roi[:,0]))/50),np.max(results_roi[:,0]))
perBinMiel=[ dnn[(results_roi[:,0]>of1) & (results_roi[:,0]<=of2) ,1] for of1,of2 in zip(ofs,ofs[1:])]

print(sum([len(d) for d in perBinMiel]))

rects2 = plt.bar(ofs[:-1], [np.mean(bm) for bm in perBinMiel], width, color='y', yerr=[np.std(bm) for bm in perBinMiel])
# plt.xticks(np.arange(0,25,0.5),['{:.2f}'.format(o) for o in ofs])


plt.show()
