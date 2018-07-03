import matplotlib.pyplot as plt
import numpy as np

from  numpy import array as arr
import itertools
import pandas as pd
import collections
import sys
from os.path import join

def draw_broken(dnnI,matches):
    cols=['g' if (row.imgname,row.roi) in matches else 'r' for i,row in dnnI.iterrows() ]
    plt.figure(figsize=(20,20))
    plt.scatter(dnnI.minorV,dnnI.brokenMielC/dnnI.beforeStopC,c=cols)
    plt.figure(figsize=(20,20))

    plt.subplot(211)
    passes=[True if (row.imgname,row.roi) in matches else False for i,row in dnnI.iterrows() ]

    dnnP=dnnI.loc[passes]

    plt.scatter(dnnI.loc[passes].minorV,dnnI.loc[passes].brokenMielC/dnnI.loc[passes].beforeStopC,c='g')

    plt.title('Matches ROI')
    plt.ylabel('Perc. of whole roi, which is broken')
    plt.xlabel('minor value')

    plt.subplot(212)

    npasses=[False if (row.imgname,row.roi) in matches else True for i,row in dnnI.iterrows() ]
    dnnNP=dnnI.loc[npasses]

    plt.scatter(dnnI.loc[npasses].minorV,dnnI.loc[npasses].brokenMielC/dnnI.loc[npasses].beforeStopC,c='r')


    plt.title('NonMatches ROI')
    plt.ylabel('Perc. of whole roi, which is broken')
    plt.xlabel('minor value')

    if doSearch:
        resB=[]
        for brProc in np.linspace(0.1,0.5,250):
            matC,nmatC=np.sum( (dnnP.brokenMielC/dnnP.beforeStopC) >brProc ), \
                       np.sum( (dnnNP.brokenMielC/dnnNP.beforeStopC) >brProc )


            resB.append(( matC/nmatC,matC/(nmatC+matC),brProc,nmatC,matC   ))


        resB.sort(key=lambda x: (0.8*x[0]) )# -(0.2*x[1]) )
        print('results top by match/nm ratio: \n','\n'.join(map(str,resB)  )   ,flush=True)


    print('Chosen params are brPerc: {}, matches caught: {}, nonmatches caught: {}'.format(
        brPerc,np.sum( (dnnP.brokenMielC/dnnP.beforeStopC) >brPerc ), \
          np.sum( (dnnNP.brokenMielC/dnnNP.beforeStopC) >brPerc )),flush=True)



    if SAVE_BEST:
        badFullLoc=( (dnnI.brokenMielC/dnnI.beforeStopC) >brPerc)
        dnnI.loc[badFullLoc,'imgname roi'.split()].to_csv(join(meas_dir,'discardedb_dnn_{}.csv'.format(exp_date)))


    # plt.show()

def sum_bigger(lis,fac):
    retL=np.zeros(len(lis))

    for ind,row in enumerate(lis):
        retL[ind]=np.sum(arr([ r>fac for r in row] ))


    return retL

def unb_bigger(lisM,lisA,fac):
    retL=np.zeros(len(lisM))

    for ind,(rowM,rowA) in enumerate(zip(lisM,lisA) ):
        retL[ind]=np.sum(arr([ (rM/rA)>fac for rM,rA in zip(rowM,rowA) ] ))


    return retL


def draw_high(dnnI,matches):
    cols=['g' if (row.imgname,row.roi) in matches else 'r' for i,row in dnnI.iterrows() ]


    passes=[True if (row.imgname,row.roi) in matches else False for i,row in dnnI.iterrows() ]
    dnnP=dnnI.loc[passes]

    npasses=[False if (row.imgname,row.roi) in matches else True for i,row in dnnI.iterrows() ]
    dnnNP=dnnI.loc[npasses]


    milLP=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnP.iterrows() ]

    milLNP=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnNP.iterrows() ]




    if doSearch:
        resB=[]
        for ind,(highV,highPerc) in enumerate(itertools.product(
                np.linspace(50,250,15,dtype=int),
                np.linspace(1,30,15,dtype=int) )):
            matC,nmatC=np.sum( sum_bigger(milLP,highV) >(highPerc) ), \
                       np.sum( sum_bigger(milLNP,highV) >(highPerc) ),

            if ind%30==2: print('in explore at: ',ind)
            resB.append(( matC/nmatC,matC/(nmatC+matC),highV,highPerc,nmatC,matC   ))

        resB.sort(key=lambda x: (0.8*x[0]) )# -(0.2*x[1]) )

        print('results top by match/nm ratio: \n','\n'.join(map(str,resB)  )   ,flush=True)

    # highPerc=0.5
    # highV=120
    # matC,nmatC=np.sum( sum_bigger(milLP,highV) >(highPerc*dnnP.afterStopC) ),  \
    # np.sum( sum_bigger(milLNP,highV) >(highPerc*dnnNP.afterStopC) ),

    # print('chosen is: ',nmatC,matC,highV,highPerc)

    # plt.show()


def draw_unbalanced(dnnI,matches):

    passes=[True if (row.imgname,row.roi) in matches else False for i,row in dnnI.iterrows() ]

    dnnP=dnnI.loc[passes]

    npasses=[False if (row.imgname,row.roi) in matches else True for i,row in dnnI.iterrows() ]
    dnnNP=dnnI.loc[npasses]



    milLP=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnP.iterrows() ]
    milLNP=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnNP.iterrows() ]
    milLFull=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnI.iterrows() ]


    axLP=[ list(map(float,row.axFList.split('|'))) for ind,row in dnnP.iterrows() ]
    axLNP=[ list(map(float,row.axFList.split('|'))) for ind,row in dnnNP.iterrows() ]
    axLFull=[ list(map(float,row.axFList.split('|'))) for ind,row in dnnI.iterrows() ]


    if doSearch:
        resB=[]
        for ind,(iunbV,iunbPerc) in enumerate(itertools.product(
                np.linspace(2,10,15,dtype=int),
                np.linspace(0.05,0.8,15) )):


            matC,nmatC=np.sum( unb_bigger(milLP,axLP,iunbV) >(iunbPerc*dnnP.afterStopC) ), \
                       np.sum( unb_bigger(milLNP,axLNP,iunbV) >(iunbPerc*dnnNP.afterStopC) ),

            if ind%30==3: print('in explore at: ',ind)

            resB.append(( matC/nmatC,matC/(nmatC+matC),iunbV,iunbPerc,nmatC,matC   ))

        resB.sort(key=lambda x: (0.8*x[0]) )# -(0.2*x[1]) )

        print('results top by match/nm ratio: \n','\n'.join(map(str,resB)  )  ,flush=True )




    matC,nmatC=np.sum( unb_bigger(milLP,axLP,unbV) >(unbPerc*dnnP.afterStopC) ), \
               np.sum( unb_bigger(milLNP,axLNP,unbV) >(unbPerc*dnnNP.afterStopC) ),

    print('Chosen params are miel/ax high thres. : {}, miel/ax length: {}, matches caught: {}, nonmatches caught: {}'\
          .format(unbV,unbPerc,matC,nmatC),flush=True)

    if SAVE_BEST:
        badFullLoc=(  unb_bigger(milLFull,axLFull,unbV) >(unbPerc*dnnI.afterStopC)  )
        dnnI.loc[badFullLoc,'imgname roi'.split()].to_csv(join(meas_dir,'discardedu_dnn_{}.csv'.format(exp_date)))



def combine_conds(dnnI,matches):
    passes=[True if (row.imgname,row.roi) in matches else False for i,row in dnnI.iterrows() ]

    dnnP=dnnI.loc[passes]

    npasses=[False if (row.imgname,row.roi) in matches else True for i,row in dnnI.iterrows() ]
    dnnNP=dnnI.loc[npasses]


    # print('pass size , non pass size',dnnI.shape[0],dnnP.shape[0],dnnNP.shape[0])

    # print(  matches- set([(i,j) for i,j in dnnP.loc[('imgname','roi' )].values  ]) )

    milLP=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnP.iterrows() ]
    milLNP=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnNP.iterrows() ]
    milLFull=[ list(map(float,row.milFList.split('|'))) for ind,row in dnnI.iterrows() ]


    axLP=[ list(map(float,row.axFList.split('|'))) for ind,row in dnnP.iterrows() ]
    axLNP=[ list(map(float,row.axFList.split('|'))) for ind,row in dnnNP.iterrows() ]
    axLFull=[ list(map(float,row.axFList.split('|'))) for ind,row in dnnI.iterrows() ]



    matC,nmatC=np.sum( (  unb_bigger(milLP,axLP,unbV) >(unbPerc*dnnP.afterStopC)  ) | \
                       ( (dnnP.brokenMielC/dnnP.beforeStopC) >brPerc)  ), \
 \
               np.sum( ( unb_bigger(milLNP,axLNP,unbV) >(unbPerc*dnnNP.afterStopC) )  | \
                       ( (dnnNP.brokenMielC/dnnNP.beforeStopC) >brPerc)		)


    if SAVE_BEST:
        badFullLoc=(  unb_bigger(milLFull,axLFull,unbV) >(unbPerc*dnnI.afterStopC)  ) | \
                   ( (dnnI.brokenMielC/dnnI.beforeStopC) >brPerc)
        dnnI.loc[badFullLoc,'imgname roi'.split()].to_csv(join(meas_dir,'discarded_dnn_{}.csv'.format(exp_date)))

    print('Chosen params are miel/ax high thres. : {}, miel/ax length: {}, broken-mil length: {}, matches caught: {}, nonmatches caught: {}'\
          .format(unbV,unbPerc,brPerc,matC,nmatC),flush=True)



confD={}
for farg in sys.argv:
    if farg.startswith('--'):
        (arg,val) = farg.split("=")
        arg = arg[2:]
        confD[arg]=val


basec=confD['case2_picDir'] #'../data/17_06_20_nubt_mlset02set_autoenc_1ep'

caseType1=confD['case_id_add1']
meas_dir=confD['raymeas_save_dir']
meas_id=confD['raymeas_add_id']


areaF=float(confD['strict_area_thres']) #0.6
centD=int(confD['strict_dist_thres']) #20



unbPerc=float(confD['mielin_axon_length'])# 0.4
unbV=float(confD['mielin_axon_scale']) # 0.2

brPerc=float(confD['mielin_break_length']) # 0.1

doSearch=bool(int(confD.get('dosearch_post_params',0)))
# print(doSearch,'hhhhhhhhhhhhhhh')

do_mielax=bool(int(confD['do_miel_ax'])) # 1 or 0
do_broken_mil=bool(int(confD['do_broken_miel'])) # 1 or 0
match_stat_str=confD['match_add_id']
exp_date=confD['postconds_id']



nonm=pd.read_csv(join(meas_dir,'cor_nonm_{}.csv'.format(confD['match_add_id'])))
dnnI=pd.read_csv(join(meas_dir,'meta_{}_{}.csv'.format(caseType1,meas_id)))
dnnMS=pd.read_csv(join(meas_dir,'match_info_{}.csv'.format(match_stat_str)))


doPlot=False
SAVE_BEST=True

#!!  only good matches
dnnM=dnnMS.loc[(dnnMS['area-fac']>areaF) & (dnnMS['cent-dist']<centD)]

print('over 51 perc area and <60pix cent dist matches'+
      ': {} \ndnn matches after stricter cutting area({}) and cent({}): {}'.format(dnnMS.shape[0],
                                                                                   areaF,centD,dnnM.shape[0]),flush=True)


# print(dnnI.goodStopC.values)
# print(dnnI.iloc[dnnI.afterStopC.values!=0].shape,dnnI.shape)


dnnI=dnnI.iloc[dnnI.afterStopC.values!=0]


# print(dnnI.shape)

matches=set([(row.img,row.roid) \
             for ind,row in dnnM.iterrows()   ])


# print( [item for item, count in collections.Counter(matches).items() if count > 1] )

# print('matches c is: ',len(matches))

if do_broken_mil: draw_broken(dnnI,matches)

# draw_high(dnnI,matches)

if do_mielax: draw_unbalanced(dnnI,matches)

combine_conds(dnnI,matches)

# print('-----------------------------------------------')

if doPlot: plt.show()


