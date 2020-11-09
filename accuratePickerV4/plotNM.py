import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import detecQuake
import trainPS
import sacTool
from imp import reload
from obspy import UTCDateTime
import numpy as np
import tool
from glob import glob
from locate import locator
import names
import mapTool as mt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['font.size']=6
staLstFile='stations/staLst_NM_New'
workDir='/home/jiangyr/accuratePickerV3/NM/'
R=[35,45,96,105]#[34,42,104,112]
staInfos=sacTool.readStaInfos(staLstFile)
laL=[35,45]
loL=[96,105]
laN=35
loN=35
taupM=tool.quickTaupModel(modelFile='include/iaspTaupMat')
modelL = [trainPS.loadModel('model/modelP_320000_100-2-15'),trainPS.loadModel('model/modelS_320000_100-2-15')]
aMat=sacTool.areaMat(laL,loL,laN,loN)
staTimeML= detecQuake.getStaTimeL(staInfos, aMat, taupM=taupM)
quakeLs=list()
bSec=UTCDateTime(2014,10,1).timestamp+0*86400*230
eSec=UTCDateTime(2015,1,1).timestamp+0*86400*231

quakeLs=tool.readQuakeLsByP('../NM/phaseLsNM20190901V?',staInfos)
detecQuake.plotQuakeDis(quakeLs,output='NM/quakeStaDis.pdf',R=[36,44,96,105],staInfos=staInfos,markersize=0.4,alpha=0.4,minCover=0.0,minSta=3,cmd='.b')
timeL0,mlL0,laL0,loL0,laL00,loL00=tool.getCatalog()
m=detecQuake.plotQuakeDis(quakeLs,output='NM/quakeStaDis.pdf',R=[36,44,96,105],staInfos=staInfos,\
    markersize=0.8,alpha=0.6,minCover=0.2,minSta=3,cmd='.b',topo='Ordos.grd',laL0=laL0,loL0=loL0,isBox=True)
timeL,laL,loL=tool.dayTimeDis(quakeLs,staInfos,mlL0,isBox=True)
tool.plotTestOutput()
tool.plotTestOutput(fileName='../resDataS_320000_100-2-15',phase='s',outDir='fig/testFigS/')
for i in range(50):
    quakeLs[10][i].plotByMat(staInfos,matDir='../NM/output20190901/',\
            outDir='fig/matWave/')
tool.plotWaveformByMat(quakeLs[0][25],staInfos)
tool.plotWaveformByMat(quakeLs[1][8],staInfos)

timeL0,mlL0,laL0,loL0,laL00,loL00,mlL00=tool.getCatalog()
timeL,laL,loL=tool.dayTimeDis(quakeLs,staInfos,mlL0,minCover=0.0,isBox=True,minSta=2)
timeL,laL,loL=tool.dayTimeDis(quakeLs,staInfos,mlL00,minCover=-1,isBox=False,minSta=3)
count,laLN,loLN,mlLN,timeLN=tool.compareTime(timeL,timeL0,laL,loL,laL00,loL00,mlL00)
timeL,laL,loL=tool.dayTimeDis(quakeLs,staInfos,mlL00,minCover=-1,isBox=True,minSta=3)



quakeL=[]
loc=locator(staInfos)
for qL in quakeLs :
    for q in qL:
        q,res=loc.locate(q,maxDT=50)
        if res <2 and res >0:
            print(q)
            quakeL.append(q)
tool.saveQuakeLs([quakeL],'NM/phaseLsNM20190901All')
quakeLs=tool.readQuakeLs('../NM/phaseLsNM20190901All',staInfos)
quakeL=tool.readQuakeLs('../NM/phaseLsNM20190901AllV2',staInfos)[0]
m=detecQuake.plotQuakeDis([quakeL],output='fig/quakeStaDis.pdf',R=[36,44,96,105],staInfos=staInfos,\
    markersize=0.8,alpha=0.6,minCover=0.2,minSta=3,cmd='.b',topo='Ordos.grd',laL0=laL0,loL0=loL0,isBox=True,\
    laLN=laLN,loLN=loLN)

tool.getStaInArea(staInfos,'stations/staNMSouth',[38,40,98,101])
staInfosR=sacTool.readStaInfos('stations/staNMSouth')
staCon=tool.getDateCon('include/fileLst')
sDate=UTCDateTime(2013,1,1).timestamp
for i in range(len(staInfosR)):
    station=staInfosR[i]['sta']
    if station in staCon:
        plt.plot(iL,iL*0+i,'.r',markersize=0.5)
        iL=np.where(staCon[station]==0)[0]
    else:
        print(station)

for i in range(len(laLN)):
    plt.plot((timeLN[i]-sDate)/86400,-1,'.b')
    print(UTCDateTime(timeLN[i]))

plt.show()
quakeLsTrain=tool.readQuakeLs('phaseLst_Train',staInfos)

dTimeL=[]
for qL in quakeLsTrain:
    for q in qL:
        for r in q:
            dTimeL.append(r[2]-r[1])
plt.hist(np.array(dTimeL),np.arange(100))
plt.show()
modelL = [\
    trainPS.loadModel('../modelP_320000_0-2-15'),\
    trainPS.loadModel('../modelS_320000_0-2-15')]
wk='201412/D20141201000108_20/'
fileL=[[wk+'NAGARA.E.SAC'],\
[wk+'NAGARA.N.SAC'],\
[wk+'NAGARA.U.SAC']]

wk='../example/'
fileL=[[wk+'XX.JMG.2008190000000.BHE'],\
[wk+'XX.JMG.2008190000000.BHN'],\
[wk+'XX.JMG.2008190000000.BHZ']]
detecQuake.showExample(fileL,modelL,t=[-10,250])


modelL = [trainPS.loadModel('../modelP_320000_0-2-15-No'),\
    trainPS.loadModel('../modelP_320000_0-2-15'),\
    trainPS.loadModel('../modelS_320000_0-2-15-No'),\
    trainPS.loadModel('../modelS_320000_0-2-15')]
wk='../'
fileL=[[wk+'GS.HYS.20150818.BHE.SAC'],[wk+'GS.HYS.20150818.BHN.SAC'],\
            [wk+'GS.HYS.20150818.BHZ.SAC']]
detecQuake.showExampleV2(fileL,modelL,t=[-10,50],staName='HYS')

#201909180500_664.BHE.sac

modelL = [trainPS.loadModel('modelP_320000_0-2-15-No'),\
    trainPS.loadModel('modelP_320000_0-2-15'),\
    trainPS.loadModel('modelS_320000_0-2-15-No'),\
    trainPS.loadModel('modelS_320000_0-2-15')]
wk='./'
fileL=[[wk+'201909180500_664.BHE.sac'],[wk+'201909180500_664.BHN.sac'],\
            [wk+'201909180500_664.BHZ.sac']]
detecQuake.showExampleV2(fileL,modelL,t=[-10,50])

modelL = [trainPS.loadModel('modelP_320000_0-2-15-No'),\
    trainPS.loadModel('modelP_320000_0-2-15-with'),\
    trainPS.loadModel('modelS_320000_0-2-15-No'),\
    trainPS.loadModel('modelS_320000_0-2-15-with')]
wk='./badSac/'
fileL=[[wk+'NM.EKH.20150819.BHE.SAC'],[wk+'NM.EKH.20150819.BHN.SAC'],\
            [wk+'NM.EKH.20150819.BHZ.SAC']]
detecQuake.showExampleV2(fileL,modelL,t=[-100,1000])

modelL = [trainPS.loadModel('modelP_320000_0-2-15-No'),\
    trainPS.loadModel('modelP_320000_0-2-15-with'),\
    trainPS.loadModel('modelS_320000_0-2-15-No'),\
    trainPS.loadModel('modelS_320000_0-2-15-with')]
wk='./badSac/'
fileL=sacFileNameL=[[wk+'201909180500_664.BHE.sac'],[wk+'201909180500_664.BHN.sac'],\
            [wk+'201909180500_664.BHZ.sac']]
detecQuake.showExampleV2(fileL,modelL,t=[-100,1000])


modelL = [trainPS.loadModel('modelP_320000_0-2-15-No'),\
    trainPS.loadModel('modelP_320000_0-2-15-with'),\
    trainPS.loadModel('modelS_320000_0-2-15-No'),\
    trainPS.loadModel('modelS_320000_0-2-15-with')]
wk='./badSac/'
fileL=sacFileNameL=[[wk+'201909220500_692.BHE.sac'],[wk+'201909220500_692.BHN.sac'],\
            [wk+'201909220500_692.BHZ.sac']]
detecQuake.showExampleV2(fileL,modelL,t=[-100,1000],staName='692')


fileDir='../NM/output20190901/16071/'
quakeLNew=[]
for filename in glob(fileDir+'*mat'):
    quakeLNew.append(tool.mat2quake(filename))
    print(quakeLNew[-1])
tool.saveQuakeLs([quakeLNew],'../NM/phaseLsNM20190901V0')
quakeLNew=tool.readQuakeLs('../NM/phaseLsNM20190901V0',staInfos)[0]
for q in quakeLNew:
    q,res=loc.locate(q,maxDT=50)
    if res <2 and res >0:
        print(q)
        quakeL.append(q)

tool.saveQuakeLs([quakeL],'../NM/phaseLsNM20190901AllV2')