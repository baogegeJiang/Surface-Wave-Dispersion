import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import detecQuake
import trainPSV3 as trainPS
import sacTool
import sys
sys.path.append("..")
from seism import StationList
from imp import reload
from obspy import UTCDateTime
import tool
from locate import locator
import names
detecQuake.maxA=1e4


workDir='/home/jiangyr/accuratePickerV3/testNew/'# workDir: the dir to save the results
staLstFile='stations/SCYN_withComp_ac'#station list file
bSec=UTCDateTime(2014,1,1).timestamp#begain date
eSec=UTCDateTime(2014,2,1).timestamp# end date
laL=[23,33]#area: [min latitude, max latitude]
loL=[96,107]#area: [min longitude, max longitude]
laN=35 #subareas in latitude
loN=35 #subareas in longitude
nameFunction=names.CEAName # the function you give in (2)  to get the file path
maxD=35#max ts-tp
f=[2,15]

#####no need to change########
taupM=tool.quickTaupModel(modelFile='include/iaspTaupMat')
modelL = [trainPS.loadModel('model/modelP_320000_0-2-15-with','norm','p'),\
trainPS.loadModel('model/modelS_320000_0-2-15-with','norm','s')]
staInfos=StationList(staLstFile)
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeLs=list()
#############################
v_i='SCYN_20200902V1'
p_i='SCYN_20200902V1'
for date in range(int(bSec),int(eSec), 86400):
    dayNum=int(date/86400)
    dayDir=workDir+('output/outputV%s/'%v_i)+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    staL = detecQuake.getStaL(staInfos, aMat, staTimeML,\
     modelL, date, getFileName=nameFunction,\
     mode='norm',f=f,maxD=maxD)
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


