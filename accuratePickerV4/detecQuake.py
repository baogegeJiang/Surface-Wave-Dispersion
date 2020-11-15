import obspy
import numpy as np
from sacTool import getDataByFileName, staTimeMat
import os
from glob import glob
import matplotlib.pyplot as plt
import tool
import sacTool
from tool import Record, Quake, QuakeCC,getYmdHMSj
from multiprocessing import Process, Manager
import threading
import time
import math
from mathFunc import getDetec, prob2color
try:
    from mapTool import plotQuakeCCDis, plotQuakeDis
except:
    print('warning cannot load mapTool')
else:
    pass
from distaz import DistAz
from numba import jit
maxA=2e19
os.environ["MKL_NUM_THREADS"] = "10"
@jit
def isZeros(a):
    new = a.reshape([-1,10,a.shape[-1]])
    if (new.std(axis=(1))==0).sum()>5:
        return True
    return False


indexL0=range(275, 1500)
@jit
def predictLongData(model, x, N=2000, indexL=range(750, 1250)):
    if len(x) == 0:
        return np.zeros(0)
    N = x.shape[0]
    Y = np.zeros(N)
    perN = len(indexL)
    loopN = int(math.ceil(N/perN))
    perLoop = int(1000)
    inMat = np.zeros((perLoop, 2000, 1, 3))
    #print(len(x))
    zeroCount=0
    for loop0 in range(0, int(loopN), int(perLoop)):
        loop1 = min(loop0+perLoop, loopN)
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                inMat[loop-loop0, :, :, :] = processX(x[sIndex: sIndex+2000, :])\
                .reshape([2000, 1, 3])
        outMat = (model.predict(inMat)[:,:,:,:1]).reshape([-1, 2000])
        for loop in range(loop0, loop1):
            i = loop*perN
            if isZeros(inMat[loop-loop0, :, :, :]):
                zeroCount+=1
                continue
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                Y[indexL0[0]+sIndex: indexL0[-1]+1+sIndex] = \
                np.append(Y[indexL0[0]+sIndex: indexL0[-1]+1+sIndex],\
                    outMat[loop-loop0, indexL0].reshape([-1])\
                    ).reshape([2,-1]).max(axis=0)
    if zeroCount>0:
        print('zeros: %d'%zeroCount)
                
    return Y
'''
@jit
def processX(X, rmean=True, normlize=True, reshape=True):
    if reshape:
        X = X.reshape(-1, 2000, 1, 3)
    if rmean:
        X = X - X.mean(axis=(1, 2)).reshape([-1, 1, 1, 3])
    if normlize:
        X = X/(X.std(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]))
    return X
'''
@jit
def processX(X, rmean=True, normlize=False, reshape=True,isNoise=False,num=2000):
    if reshape:
        X = X.reshape(-1, num, 1, 3)
    #print(X.shape)
    if rmean:
        X-= X.mean(axis=1,keepdims=True)
    if normlize:
        X /=(X.std(axis=(1, 2, 3),keepdims=True))
    if isNoise:
        X+=(np.random.rand(X.shape[0],num,1,3)-0.5)*np.random.rand(X.shape[0],1,1,3)*X.max(axis=(1,2,3),keepdims=True)*0.15*(np.random.rand(X.shape[0],1,1,1)<0.1)
    return X



def originFileName(net, station, comp, YmdHMSJ, dirL=['data/']):
    #dir='tmpSacFile/'
    sacFileNames = list()
    Y = YmdHMSJ
    for dir in dirL:
        sacFileNamesStr = dir+net+'.'+station+'.'+Y['Y']+Y['j']+\
            '*'+comp
        for file in glob(sacFileNamesStr):
            sacFileNames.append(file)
    return sacFileNames

class sta(object):
    def __init__(self, station, day, modelL=None, staTimeM=None,\
     loc=None, comp=['BHE','BHN','BHZ'], getFileName=originFileName, \
     freq=[-1, -1], mode='mid', isClearData=False,\
     taupM=tool.quickTaupModel(),isPre=True,delta0=0.02,R=[-91,91,\
    -181,181],maxD=80,bTime=None,eTime=None):
        self.net = station['net']
        self.loc = station.loc()
        self.station = station['sta']
        self.sta = station
        self.day = day
        self.comp = comp
        self.taupM=taupM
        if loc[0]<R[0] or loc[0]>R[1] or loc[1]<R[2] or loc[1]>R[3]:
            self.data=sacTool.Data(np.zeros((0,3)))
            print('skip')
        else:
            self.data = getDataByFileName(station.getFileNames(self.day), freq=freq,delta0=delta0,\
                maxA=maxA,bTime=bTime,eTime=eTime)
        #print(len(sta.data.data))
        print(self.station,self.data.bTime,self.data.eTime,self.data.data.std())
        self.timeL = list()
        self.vL = list()
        self.mode = mode
        if isPre==True:
            indexLL = [range(275, 775), range(275, 775)]
            if mode=='norm':
                minValueL=[0.5,0.5]
            if mode=='high':
                minValueL=[0.4, 0.4]
            if mode=='mid':
                minValueL=[0.25, 0.25]
            if mode=='low':
                minValueL=[0.2, 0.2]
            if mode=='higher':
                minValueL=[0.6, 0.6]
            minDeltaL=[500, 750]
            for i in range(len(modelL)):
                tmpL = getDetec(predictLongData(modelL[i], self.data.data,\
                 indexL=indexLL[i]), minValue=minValueL[i], minDelta =\
                  minDeltaL[i])
                print('find',len(tmpL[0]))
                self.timeL.append(tmpL[0])
                self.vL.append(tmpL[1])
            self.pairD = self.getPSPair(maxD=maxD)
            self.isPick = np.zeros(len(self.pairD))
            self.orignM = self.convertPS2orignM(staTimeM)
            if isClearData:
                self.clearData()

    def __repr__(self):
        reprStr=self.net + ' '+self.station+\
        str(self.loc)
        return 'detec in station '+ reprStr


    def getSacFileNamesL(self, station):
        return station.getFileNames(self.day)

    def clearData(self):
        self.data.data = np.zeros((0, 3))

    def plotData(self):
        colorStr = ['.r', '.g']
        plt.plot(self.data.data[:,2]/self.data.data[:,2].max()\
            + np.array(0))
        for i in range(len(self.timeL)):
            plt.plot(self.timeL[i],self.vL[i], colorStr[i])
        plt.show()

    def calOrign(self, pTime, sTime):
        return self.taupM.get_orign_times(pTime, sTime, self.data.delta)

    def getPSPair(self, maxD=80):
        pairD = list()
        if len(self.timeL) == 0:
            return pairD
        if self.data.delta==0:
            return pairD
        maxN = maxD/self.data.delta
        pN=len(self.timeL[0])
        sN=len(self.timeL[1])
        j0=0
        for i in range(pN):
            pTime = self.timeL[0][i]
            if i < pN-1 and self.mode != 'low':
                pTimeNext = self.timeL[0][i+1]
            else:
                pTimeNext= self.timeL[0][i]+maxN
            pTimeNext = min(pTime+maxN, pTimeNext)
            isS = 0
            for j in range(j0, sN):
                if isS==0:
                    j0=j
                if self.timeL[1][j] > pTime and self.timeL[1][j] < pTimeNext:
                    sTime=self.timeL[1][j]
                    #print(pTime, sTime)
                    pairD.append([pTime*self.data.delta, sTime*self.data.delta\
                        , self.calOrign(pTime, sTime)*self.data.delta, \
                        (sTime-pTime)*self.data.delta, i, j])
                    isS=1
                if self.timeL[1][j] >= pTimeNext:
                    break
        return pairD

    def convertPS2orignM(self, staTimeM, maxDTime=2):
        laN = staTimeM.minTimeD.shape[0]
        loN = staTimeM.minTimeD.shape[1]
        orignM = [[list() for j in range(loN)] for i in range(laN)]
        if len(self.pairD)==0:
            return orignM
        bSec = self.data.bTime.timestamp
        timeL = np.zeros(len(self.pairD))
        for i in range(len(self.pairD)):
            timeL[i] = self.pairD[i][2]+bSec
        sortL = np.argsort(timeL)
        for i in sortL:
            for laIndex in range(laN):
                for loIndex in range(loN):
                    if self.pairD[i][3] >= staTimeM.minTimeD[laIndex][loIndex] - maxDTime \
                    and self.pairD[i][3] <= staTimeM.maxTimeD[laIndex][loIndex] + maxDTime:
                        pTime = self.pairD[i][0]+bSec
                        sTime = self.pairD[i][1]+bSec
                        timeTmp = [pTime, sTime, timeL[i], i]
                        orignM[laIndex][loIndex].append(timeTmp)
        return orignM
    def filt(self,f=[-1,-1],filtOrder=2):
        self.data.filt(f,filtOrder)
        return self
    def resample(self,resampleN):
        self.data.resample(resampleN)
        return self
        
def argMax2D(M):
    maxValue = np.max(M)
    maxIndex = np.where(M==maxValue)
    return maxIndex[0][0], maxIndex[1][0]


def associateSta(staL, aMat, staTimeML, timeR=30, minSta=3, maxDTime=3, N=1, \
    isClearData=False, locator=None, maxD=80):
    timeN = int(timeR)*2
    startTime = obspy.UTCDateTime(2100, 1, 1)
    endTime = obspy.UTCDateTime(1970, 1, 1)
    staN = len(staL)
    for staIndex in range(staN):
        if isClearData:
            staL[staIndex].clearData()
        staL[staIndex].isPick = staL[staIndex].isPick*0
    for staTmp in staL:
        if len(staTmp.data.data) == 0:
            continue
        startTime = min(startTime, staTmp.data.bTime)
        endTime = max(endTime, staTmp.data.eTime)
    startSec = int(startTime.timestamp-90)
    endSec = int(endTime.timestamp+30)
    if N==1:
        quakeL=[]
        __associateSta(quakeL, staL, \
            aMat, staTimeML, startSec, \
            endSec, timeR=timeR, minSta=minSta,\
             maxDTime=maxDTime,locator=locator,maxD=maxD)
        return quakeL
    for i in range(len(staL)):
        staL[i].clearData()
    manager=Manager()
    quakeLL=[manager.list() for i in range(N)]
    perN = int(int((endSec-startSec)/N+1)/timeN+1)*timeN
    processes=[]
    for i in range(N):
        process=Process(target=__associateSta, args=(quakeLL[i], \
            staL, aMat, staTimeML, startSec+i*perN, \
            startSec+(i+1)*perN+1))
        #process.setDaemon(True)
        process.start()
        processes.append(process)

    for process in processes:
        print(process)
        process.join()
    quakeL=list()

    for quakeLTmp in quakeLL:
        for quakeTmp in quakeLTmp:
            quakeL.append(quakeTmp)
    return quakeL
    

def __associateSta(quakeL, staL, aMat, staTimeML, startSec, endSec, \
    timeR=30, minSta=3, maxDTime=3, locator=None,maxD=80):
    print('start', startSec, endSec)
    laN = aMat.laN
    loN = aMat.loN
    staN = len(staL)
    timeN = int(timeR)*30
    stackM = np.zeros((timeN*3, laN, loN))
    tmpStackM=np.zeros((timeN*3+3*maxDTime, laN, loN))
    stackL = np.zeros(timeN*3)
    staMinTimeL=np.ones(staN)*0
    quakeCount=0
    dTimeL=np.arange(-maxDTime, maxDTime+1)
    for loop in range(2):
        staOrignMIndex = np.zeros((staN, laN, loN), dtype=int)
        staMinTimeL=np.ones(staN)*0
        count=0
        for sec0 in range(startSec, endSec, timeN):
            count=count+1
            if count%10==0:
                print('process:',(sec0-startSec)/(endSec-startSec)*100,'%  find:',len(quakeL))
            stackM[0:2*timeN, :, :] = stackM[timeN:, :, :]
            stackM[2*timeN:, :, :] = stackM[0:timeN, :, :]*0
            tmpStackM=tmpStackM*0
            st=sec0+2*timeN - maxDTime
            et=sec0+3*timeN + maxDTime
            for staIndex in range(staN):
                tmpStackM=tmpStackM*0
                for laIndex in range(laN):
                    for loIndex in range(loN):
                        if len(staL[staIndex].orignM[laIndex][loIndex])>0:
                            index0=staOrignMIndex[staIndex, laIndex, loIndex]
                            for index in range(index0, len(staL[staIndex].orignM[laIndex][loIndex])):
                                timeT = staL[staIndex].orignM[laIndex][loIndex][index][2]
                                pairIndex = staL[staIndex].orignM[laIndex][loIndex][index][3]
                                if timeT >et:
                                    staOrignMIndex[staIndex, laIndex, loIndex] = index
                                    break
                                if timeT > st and staL[staIndex].isPick[pairIndex]==0:
                                    pIndex = staL[staIndex].pairD[pairIndex][4]
                                    sIndex = staL[staIndex].pairD[pairIndex][5]
                                    pTime = staL[staIndex].timeL[0][pIndex]
                                    sTime = staL[staIndex].timeL[1][sIndex]
                                    staOrignMIndex[staIndex, laIndex, loIndex] = index
                                    if pTime * sTime ==0:
                                        continue
                                    tmpStackM[int(timeT-sec0)+dTimeL, laIndex, loIndex]=\
                                    tmpStackM[int(timeT-sec0)+dTimeL, laIndex, loIndex]*0+1
                                    '''
                                    for dt in range(-maxDTime, maxDTime+1):
                                        tmpStackM[int(timeT-sec0+dt), laIndex, loIndex]=1
                                    '''
                stackM[2*timeN: 3*timeN, :, :] += tmpStackM[2*timeN: 3*timeN, :, :]

            stackL = stackM.max(axis=(1,2))
            peakL, peakN = tool.getDetec(stackL, minValue=minSta, minDelta=timeR)

            for peak in peakL:
                if peak > timeN and peak <= 2*timeN:
                    time = peak + sec0
                    laIndex, loIndex = argMax2D(stackM[peak, :, :].reshape((laN, loN)))
                    quakeCount+=1
                    quake = Quake(loc=[aMat[laIndex][loIndex].midLa, aMat[laIndex][loIndex].midLo,10.0],\
                        time=time, randID=quakeCount)
                    for staIndex in range(staN):
                        isfind=0
                        if staTimeML[staIndex].minTimeS[laIndex,loIndex]\
                        -staTimeML[staIndex].minTimeP[laIndex,loIndex] > maxD:
                            continue
                        if len(staL[staIndex].orignM[laIndex][loIndex]) != 0:
                            for index in range(staOrignMIndex[staIndex, laIndex, loIndex], -1, -1):
                                if int(abs(staL[staIndex].orignM[laIndex][loIndex][index][2]-time))<=maxDTime:
                                    if staL[staIndex].isPick[staL[staIndex].\
                                            orignM[laIndex][loIndex][index][3]]==0:
                                        pairDIndex = staL[staIndex].orignM[laIndex][loIndex][index][3]
                                        pIndex = staL[staIndex].pairD[pairDIndex][4]
                                        sIndex = staL[staIndex].pairD[pairDIndex][5]
                                        if staL[staIndex].timeL[0][pIndex] > 0 and \
                                                staL[staIndex].timeL[1][sIndex] > 0:
                                            quake.append(Record(staIndex, \
                                                staL[staIndex].orignM[laIndex][loIndex][index][0], \
                                                staL[staIndex].orignM[laIndex][loIndex][index][1],\
                                                staL[staIndex].vL[0][pIndex],\
                                                staL[staIndex].vL[1][sIndex]))
                                            isfind=1
                                            staL[staIndex].timeL[0][pIndex] = 0
                                            staL[staIndex].timeL[1][sIndex] = 0
                                            staL[staIndex].isPick[pairDIndex] = 1
                                            break
                                if staL[staIndex].orignM[laIndex][loIndex][index][2] < time - maxDTime:
                                    break
                            if isfind==0:
                                pTime=0
                                sTime=0
                                pProb=-1
                                sProb=-1
                                pTimeL=staL[staIndex].timeL[0]*staL[staIndex].data.delta\
                                +staL[staIndex].data.bTime.timestamp
                                sTimeL=staL[staIndex].timeL[1]*staL[staIndex].data.delta\
                                +staL[staIndex].data.bTime.timestamp
                                pTimeMin=time+staTimeML[staIndex].minTimeP[laIndex,loIndex]-maxDTime
                                pTimeMax=time+staTimeML[staIndex].maxTimeP[laIndex,loIndex]+maxDTime
                                sTimeMin=time+staTimeML[staIndex].minTimeS[laIndex,loIndex]-maxDTime
                                sTimeMax=time+staTimeML[staIndex].maxTimeS[laIndex,loIndex]+maxDTime
                                validP=np.where((pTimeL/1e5-pTimeMin/1e5)*(pTimeL/1e5-pTimeMax/1e5)<=0)
                                if len(validP)>0:
                                    if len(validP[0])>0:
                                        pTime=pTimeL[validP[0]][0]
                                        pIndex=validP[0][0]
                                        pProb = staL[staIndex].vL[0][pIndex]
                                if pTime < 1:
                                    continue
                                validS=np.where((sTimeL-sTimeMin)*(sTimeL-sTimeMax) < 0)
                                if len(validS)>0:
                                    if len(validS[0])>0:
                                        sTime=sTimeL[validS[0]][0]
                                        sIndex=validS[0][0]
                                        sProb = staL[staIndex].vL[1][sIndex]
                                if pTime > 1:
                                    if sTime <1  and staL[staIndex].vL[0][pIndex]<0.3:
                                        continue
                                    staL[staIndex].timeL[0][pIndex]=0
                                    if sTime >1:
                                        staL[staIndex].timeL[1][sIndex]=0
                                    quake.append(Record(staIndex, pTime, sTime, pProb, sProb))
                    if locator != None and len(quake)>=3:
                        try:
                            quake,res=locator.locate(quake,maxDT=25)
                            print(quake.time,quake.loc,res)
                        except:
                            print('wrong in locate')
                        else:
                            pass
                    quakeL.append(quake)
    return quakeL

def getStaTimeL(staInfos, aMat,taupM=tool.quickTaupModel()):
    #manager=Manager()
    #staTimeML=manager.list()
    staTimeML=list()
    for staInfo in staInfos:
        loc=staInfo.loc()[:2]
        staTimeML.append(staTimeMat(loc, aMat, taupM=taupM))
    return staTimeML

def getSta(staL,i, staInfo, date, modelL, staTimeM, loc, \
        freq,getFileName,taupM, mode,isPre=True,R=[-90,90,\
    -180,180],comp=['BHE','BHN','BHZ'],maxD=80,delta0=0.02,\
    bTime=None,eTime=None):
    staL[i] = sta(staInfo, date, modelL, staTimeM, loc, \
            freq=freq, getFileName=getFileName, taupM=taupM, \
            mode=mode,isPre=isPre,R=R,comp=comp,maxD=maxD,\
            delta0=delta0,bTime=bTime,eTime=eTime)


def getStaL(staInfos, aMat=[], staTimeML=[], modelL=[],\
    date=obspy.UTCDateTime(0), getFileName=originFileName,\
    taupM=tool.quickTaupModel(), mode='mid', N=5,\
    isPre=True,f=[2, 15],R=[-90,90,\
    -180,180],maxD=80,f_new=[-1,-1],delta0=0.02,resampleN=-1\
    ,bTime=None,eTime=None):
    staL=[None for i in range(len(staInfos))]
    threads = list()
    for i in range(len(staInfos)):
        staInfo=staInfos[i]
        nt = staInfo['net']
        st = staInfo['sta']
        loc = [staInfo['la'],staInfo['lo']]
        comp=staInfo['comp']
        if len(staTimeML)>0:
            staTimeM=staTimeML[i]
        else:
            staTimeM=None
        print('process on sta: ',i)
        getSta(staL, i, staInfo, date, modelL, staTimeM, loc, \
            f, getFileName, taupM, mode,isPre=isPre,R=R,\
            comp=comp,maxD=maxD,delta0=delta0,bTime=bTime,\
            eTime=eTime)
        staL[i].filt(f_new)
        staL[i].resample(resampleN)
    return staL
    for i in range(len(threads)):
        print('process on sta: ',i)
        thread = threads[i]
        while threading.activeCount()>N:
            time.sleep(0.1)
        thread.start()

    for i in range(len(threads)):
        threads[i].join()
        print('sta: ',i,' completed')
         
    return staL
'''

'''
def showExample(filenameL,modelL,delta=0.02,t=[]):
    data=getDataByFileName(filenameL,freq=[2,15])
    data=data.data[:2000*50]
    
    #i0=int(750/delta)
    #i1=int(870/delta)
    #plt.specgram(np.sign(data[i0:i1,1])*(np.abs(data[i0:i1,1])**0.5),NFFT=200,Fs=50,noverlap=190)
    data/=data.max()/2
    #plt.colorbar()
    #plt.show()
    plt.close()
    plt.figure(figsize=[4,4])
    yL=[predictLongData(modelL[i],data) for i in range(2)]
    timeL=np.arange(data.shape[0])*delta-720
    #print(data.shape,timeL.shape)
    for i in range(3):
        plt.plot(timeL,np.sign(data[:,i])*(np.abs(data[:,i]))+i,'k',linewidth=0.3)
    for i in range(2):
        plt.plot(timeL,yL[i]-i-1.5,'k',linewidth=0.5)
    if len(t)>0:
        plt.xlim(t)
    plt.yticks(np.arange(-2,3),['S','P','E','N','Z'])
    plt.ylim([-2.7,3])
    plt.xlabel('t/s')
    plt.savefig('fig/complexCondition.eps')
    plt.savefig('fig/complexCondition.tiff',dpi=300)
    plt.close()
    

def showExampleV2(filenameL,modelL,delta=0.02,t=[],staName='sta'):
    data=getDataByFileName(filenameL,freq=[2,15])
    data=data.data[:3500*50]
    
    #i0=int(750/delta)
    #i1=int(870/delta)
    #plt.specgram(np.sign(data[i0:i1,1])*(np.abs(data[i0:i1,1])**0.5),NFFT=200,Fs=50,noverlap=190)
    data/=data.max()/2
    #plt.colorbar()
    #plt.show()
    plt.close()
    plt.figure(figsize=[4,4])
    yL=[predictLongData(model,data) for model in modelL]
    timeL=np.arange(data.shape[0])*delta-720
    #print(data.shape,timeL.shape)
    for i in range(3):
        plt.plot(timeL,np.sign(data[:,i])*(np.abs(data[:,i]))+i,'k',linewidth=0.3)
    for i in range(len(modelL)):
        plt.plot(timeL,yL[i]-i-1.5,'k',linewidth=0.5)
        #plt.plot(timeL,yL[i]*0+0.5-i-1.5,'--k',linewidth=0.5)
    if len(t)>0:
        plt.xlim(t)
    plt.yticks(np.arange(-4,3),['S1','S0','P1','P0','E','N','Z'])
    plt.ylim([-4.7,3])
    plt.xlabel('t/s')
    plt.savefig('fig/complexConditionV2_%s.eps'%staName)
    plt.savefig('fig/complexConditionV2_%s.tiff'%staName,dpi=300)
    plt.close()

def plotRes(staL, quake, filename=None):
    colorStr='br'
    for record in quake:
        color=0
        pTime=record[1]
        sTime=record[2]
        staIndex=record[0]
        if staIndex>100:
            color=1
        print(staIndex,pTime, sTime)
        st=quake.time-10
        et=sTime+10
        if sTime==0:
            et=pTime+30
        pD=(pTime-quake.time)%1000
        if pTime ==0:
            pD = ((sTime-quake.time)/1.73)%1000
        if staL[staIndex].data.data.size<100:
            continue
        print(st, et, staL[staIndex].data.delta)
        timeL=np.arange(st, et, staL[staIndex].data.delta)
        #data = staL[staIndex].data.getDataByTimeL(timeL)
        data=staL[staIndex].data.getDataByTimeLQuick(timeL)
        if timeL.shape[0] != data.shape[0]:
            print('not same length for plot')
            continue
        if timeL.size<1:
            print("no timeL for plot")
            continue
        indexL=np.arange(data.shape[0])
        if pTime>0:
            index0=max(int((pTime-5-st)/staL[staIndex].data.delta),0)
            index1=int((pTime+5-st)/staL[staIndex].data.delta)
            indexL=np.arange(index0,index1)
        #if record.pProb()>1 or record.pProb()<0:
        #    plt.plot(timeL, data[:, 2]/data[indexL,2].max()+pD,colorStr[color],linewidth=0.3)
        #else:
        if True:
            color = prob2color(record.pProb())
            plt.plot(timeL, data[:, 2]/data[indexL,2].max()+pD,color=color,linewidth=0.3)
        plt.text(timeL[0],pD+0.5,'%s %.2f %.2f'%(staL[staIndex].station,record.pProb(),\
            record.sProb()))
        if pTime>0:
            plt.plot([pTime, pTime], [pD+2, pD-2], 'g')
            if isinstance(quake,QuakeCC):
                plt.text(pTime+1,pD+0.5,'%.2f'%record.getPCC())
        if sTime >0:
            plt.plot([sTime, sTime], [pD+2, pD-2], 'k')
            if isinstance(quake,QuakeCC):
                plt.text(sTime+1,pD+0.5,'%.2f'%record.getSCC())
    if isinstance(quake,QuakeCC):
        plt.title('%s %.3f %.3f %.3f cc:%.3f' % (obspy.UTCDateTime(quake.time).\
            ctime(), quake.loc[0], quake.loc[1],quake.loc[2],quake.cc))
    else:
        plt.title('%s %.3f %.3f %.3f' % (obspy.UTCDateTime(quake.time).\
            ctime(), quake.loc[0], quake.loc[1],quake.loc[2]))
    if filename==None:
        plt.show()
    if filename!=None:
        dayDir=os.path.dirname(filename)
        if not os.path.exists(dayDir):
            os.mkdir(dayDir)
        plt.savefig(filename,dpi=300)
        plt.close()

def plotResS(staL,quakeL, outDir='output/'):
    for quake in quakeL:
        filename=outDir+'/'+quake.filename[0:-3]+'png'
        #filename=outDir+'/'+str(quake.time)+'.jpg'
        try:
            plotRes(staL,quake,filename=filename)
        except:
            pass
        else:
            pass

def getStaLByQuake(staInfos, aMat, staTimeML, modelL,quake,\
    getFileName=originFileName,taupM=tool.quickTaupModel(), \
    mode='mid', N=5,isPre=False,bTime=-100,delta0=0.02):
    staL=[None for i in range(len(staInfos))]
    threads = list()
    for i in range(len(staInfos)):
        staInfo=staInfos[i]
        nt = staInfo['net']
        st = staInfo['sta']
        loc = [staInfo['la'],staInfo['lo']]
        print('process on sta: ',i)
        dis=DistAz(quake.loc[0],quake.loc[1],staInfos[i]['la'],\
            staInfos[i]['lo']).getDelta()
        date=obspy.UTCDateTime(quake.time+taupM.get_travel_times(quake.loc[2],dis)[0].time+bTime)
        getSta(staL, i, nt, st, date, modelL, staTimeML[i], loc, \
            [0.01, 15], getFileName, taupM, mode,isPre=isPre,delta0=delta0)
    return staL
