import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import detecQuake

import sacTool
import sys
sys.path.append("..")
from seism import StationList
import fcn as trainPS
from imp import reload
from obspy import UTCDateTime
import tool
from locate import locator
import names
detecQuake.maxA=1e15#这个是否有道理

#2019-06-17
workDir='/HOME/jiangyr/detecQuake/'# workDir: the dir to save the results
staLstFileL=['../stations/XU_sel.sta','../stations/SCYN_withComp_ac',]#station list file
bSec=UTCDateTime(2019,6,15).timestamp#begain date
eSec=UTCDateTime(2019,6,30).timestamp# end date
laL=[20,34]#area: [min latitude, max latitude]
loL=[93,110]#area: [min longitude, max longitude]
laN=35 #subareas in latitude
loN=35 #subareas in longitude
maxD=40#max ts-tp
f=[0.5,20]

if not os.path.exists(workDir+'output/'):
    os.makedirs(workDir+'output/')
if not os.path.exists(workDir+'/phaseDir/'):
    os.makedirs(workDir+'/phaseDir/')
#####no need to change########
taupM=tool.quickTaupModel(modelFile='include/iaspTaupMat')
modelL = [trainPS.genModel0('norm','p')[0],trainPS.genModel0('norm','s')[0]]
modelL[0].load_weights('model/norm_p_2000000_400000')
modelL[1].load_weights('model/norm_s_2000000_400000')
staInfos=StationList(staLstFileL[0])+StationList(staLstFileL[1])
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeLs=list()
#############################
v_i='SCYN_20201112_CN7'
p_i='SCYN_20201112_CN7'
for date in range(int(bSec),int(eSec), 86400):
    dayNum=int(date/86400)
    dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML,\
     modelL, date, mode='norm',f=f,maxD=maxD)
    quakeLs.append(detecQuake.associateSta(staL, aMat, \
        staTimeML, timeR=30, maxDTime=2, N=1,locator=\
        locator(staInfos),maxD=maxD))
    '''
    save:
    result's in  workDir+'phaseDir/phaseLstVp_i'
    result's waveform in  workDir+'output/outputVv_i/'
    result's plot picture in  workDir+'output/outputVv_i/'
    '''
    tool.saveQuakeLWaveform(staL, quakeLs[-1], \
        matDir=workDir+'output/outputV%s/'%v_i,\
            index0=-1500,index1=1500)
    detecQuake.plotResS(staL,quakeLs[-1],outDir\
        =workDir+'output/outputV%s/'%v_i)
    tool.saveQuakeLs(quakeLs, workDir+'phaseDir/phaseLstV%s'%p_i)
    staL=[]# clear data  to save memory
'''
R=[35,45,96,105]
staInfos=sacTool.readStaInfos('../staLstNMV2')
staInfos=tool.getStaInArea(staInfos,'staLstNMV2Select',R)
staInfos=sacTool.readStaInfos('staLstNMV2Select')
quakeLs=tool.readQuakeLs('../NM/phaseLstNMALLReloc_removeBadSta',staInfos)
detecQuake.plotQuakeDis(quakeLs[:1],R=R,staInfos=staInfos,topo=None,minCover=0)
plt.savefig('quakeLsDis.png',dpi=500)
'''


