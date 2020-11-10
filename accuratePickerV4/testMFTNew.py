import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import detecQuake
import pyMFTCuda
import sacTool
from imp import reload
from obspy import UTCDateTime
import tool
from locate import locator
import names
from cudaFunc import nptype,convert



workDir='/home/jiangyr/accuratePickerV3/testNew/'# workDir: the dir to save the results
staLstFile='staLstNMV2Select'#station list file
quakeTmpFile='phaseLstNewTomoReloc'
bSec=UTCDateTime(2014,1,1).timestamp#begain date
eSec=UTCDateTime(2014,1,3).timestamp# end date
nameFunction=names.NMFileName # the function you give in (2)  to get the file path  

#####no need to change########
staInfos=sacTool.readStaInfos(staLstFile)
quakeTmpL=tool.readQuakeLs(quakeTmpFile,staInfos)[0]
loc=locator(staInfos)
waveformL,tmpNameL=tool.loadWaveformL(quakeTmpL,matDir='outputV1_with',\
    isCut=True,index0=-200,index1=300,f=[2,8],convert=convert,nptype=nptype,\
    resampleN=-1)
#############################
v_i='20_with_2_8'
p_i='20_with_2_8'
quakeCCLs=[]
for date in range(int(bSec),int(eSec), 86400):
    dayNum=int(date/86400)
    dayDir=workDir+('outputCCV%s/'%v_i)+str(dayNum)
    if os.path.exists(dayDir):
        print('done')
        continue
    date=UTCDateTime(float(date))
    print('pick on ',date)
    
    ####################################################################
    staL = detecQuake.getStaL(staInfos,[], [],\
     [], date, getFileName=names.NMFileNameBe,f=[1/2,20]\
     ,f_new=[2,8],isPre=False,delta0=0.02,resampleN=-1,\
     eTime=date+86400/2)

    quakeBe=pyMFTCuda.doMFTAll(staL,waveformL,date,locator=loc,\
        tmpNameL=tmpNameL,quakeRefL=quakeTmpL,deviceL=['cuda:0'],\
        minDelta=50*10,maxCC=0.4,maxDis=150,MINMUL=7,minMul=3,\
        mincc=0.4,winTime=0.4,delta=0.02,n=86400*25)
    tool.saveQuakeLWaveform(staL, quakeBe, \
        matDir=workDir+'outputCCV%s/'%v_i,\
            index0=-1500,index1=1500)
    detecQuake.plotResS(staL,quakeBe,outDir\
        =workDir+'outputCCV%s/'%v_i)
    ###################################################################

    staL = detecQuake.getStaL(staInfos,[], [],\
     [], date, getFileName=names.NMFileNameAf,f=[1/2,20]\
     ,f_new=[2,8],isPre=False,delta0=0.02,resampleN=-1,\
     bTime=date+86400/2)
    quakeAf=pyMFTCuda.doMFTAll(staL,waveformL,date,locator=loc,\
        tmpNameL=tmpNameL,quakeRefL=quakeTmpL,deviceL=['cuda:0'],\
        minDelta=50*10,maxCC=0.4,maxDis=150,MINMUL=7,minMul=3,\
        mincc=0.4,winTime=0.4,delta=0.02,n=86400*25)
    tool.saveQuakeLWaveform(staL, quakeAf, \
        matDir=workDir+'outputCCV%s/'%v_i,\
            index0=-1500,index1=1500)
    detecQuake.plotResS(staL,quakeAf,outDir\
        =workDir+'outputCCV%s/'%v_i)
    ##################################################################
    quakeCCLs.append(quakeBe+quakeAf)
    sta=[]
    '''
    save:
    result's in  workDir+'phaseLst'
    result's waveform in  workDir+'output/'
    result's plot picture in  workDir+'output/'
    '''
    tool.saveQuakeLs(quakeCCLs, workDir+'phaseLstCCV%s'%p_i)
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

