import numpy as np
import h5py
import scipy.io as sio
from scipy import interpolate as interp
from obspy import UTCDateTime, taup
import obspy
from multiprocessing import Process, Manager
from mathFunc import matTime2UTC,rad2deg, getDetec
import re
from glob import glob
from distaz import DistAz
import matplotlib.pyplot as plt
from scipy import signal
from openpyxl import Workbook
from handleLog import getLocByLogsP
import os

plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['font.size']=7
def processX(X, rmean=True, normlize=True, reshape=True):
    if reshape:
        X = X.reshape(-1, 2000, 1, 3)
    if rmean:
        X = X - X.mean(axis=(1, 2)).reshape([-1, 1, 1, 3])
    if normlize:
        X = X/(X.std(axis=(1, 2, 3)).reshape([-1, 1, 1, 1]))
    return X

def getYmdHMSj(date):
    YmdHMSj = {}
    YmdHMSj.update({'Y': date.strftime('%Y')})
    YmdHMSj.update({'m': date.strftime('%m')})
    YmdHMSj.update({'d': date.strftime('%d')})
    YmdHMSj.update({'H': date.strftime('%H')})
    YmdHMSj.update({'M': date.strftime('%M')})
    YmdHMSj.update({'S': date.strftime('%S')})
    YmdHMSj.update({'j': date.strftime('%j')})
    YmdHMSj.update({'J': date.strftime('%j')})
    return YmdHMSj


class Record(list):
    '''
    the basic class 'Record' used for single station's P and S record
    it's based on python's default list
    [staIndex pTime sTime]
    we can setByLine
    we provide a way to copy
    '''
    def __init__(self,staIndex=-1, pTime=-1, sTime=-1, pProb=100, sProb=100):
        super(Record, self).__init__()
        self.append(staIndex)
        self.append(pTime)
        self.append(sTime)
        self.append(pProb)
        self.append(sProb)
    def __repr__(self):
        return self.summary()
    def summary(self):
        Summary='%d'%self[0]
        for i in range(1,len(self)):
            Summary=Summary+" %f "%self[i]
        Summary=Summary+'\n'
        return Summary
    def copy(self):
        return Record(self[0],self[1],self[2])
    def getStaIndex(self):
        return self[0]
    def staIndex(self):
        return self.getStaIndex()
    def pTime(self):
        return self[1]
    def sTime(self):
        return self[2]
    def pProb(self):
        return self[-2]
    def sProb(self):
        return self[-1]
    def setByLine(self,line):
        if isinstance(line,str):
            line=line.split()
        self[0]=int(line[0])
        for i in range(1,len(line)):
            self[i]=float(line[i])
        return self
    def set(self,line,mode='byLine'):
        return self.setByLine(line)
    def clear(self):
            self[1]=.0
            self[2]=.0
    def selectByReq(self,req={},waveform=None,oTime=None):
        '''
        maxDT oTime
        minSNR tLBe tlAf
        minP(modelL, predictLongData)
        '''
        if 'minRatio' in req and oTime!=None:
            if self.pTime()>0 and self.sTime()>0:
                if (self.sTime()-oTime)/(self.pTime()-oTime)<req['minRatio']:
                    self.clear()
        if 'maxDT' in req and 'oTime' in req:
            if self.pTime()>0 and \
                self.pTime()-req['oTime']>req['maxDT']:
                self.clear()
        if isinstance(waveform,dict):
            if 'minSNR' in req :
                if self.calSNR(req,waveform)< req['minSNR']:
                    self.clear()
            if 'minP' in req or 'minS' in req :
                p, s = self.calPS(req, waveform)
                if 'minP' in req:
                    if p < req['minP']:
                        self[1] = 0.
                if 'minS' in req:
                    if s < req['minS']:
                        self[2] = 0.
        return self
    def calSNR(self,req={},waveform=None):
        '''
        tLBe tlAf
        '''
        if not isinstance(waveform,dict):
            return -1
        staIndexNew=np.where(waveform['staIndexL'][0]==self.getStaIndex())[0]
        i0=np.where(waveform['indexL'][0]==0)[0]
        delta=waveform['deltaL'][0][staIndexNew]
        if delta==0:
            return -1
        iN=waveform['indexL'][0].size
        tLBe=[-15,-5]
        tLAf=[0,10]
        if 'tLBe' in req:
            tlBe=tLBe
        if 'tLAf' in req:
            tLAf=tLAf
        isBe=int(max(0,i0+int(tLBe[0]/delta)))
        ieBe=int(i0+int(tLBe[1]/delta))
        isAf=int(i0+int(tLAf[0]/delta))
        ieAf=int(min(iN-1,i0+int(tLAf[1]/delta)))
        aBe=np.abs(waveform['pWaveform'][staIndexNew,isBe:ieBe]).max()
        aAf=np.abs(waveform['pWaveform'][staIndexNew,isAf:ieAf]).max()
        return aAf/(aBe+1e-9)
    def calPS(self,req={},waveform=None):
        '''
        tLBe tlAf
        '''
        if not isinstance(waveform,dict):
            return -1
        staIndexNew=np.where(waveform['staIndexL'][0]==self.getStaIndex())[0][0]
        i0=np.where(waveform['indexL'][0]==0)[0][0]
        delta=waveform['deltaL'][0][staIndexNew]
        if delta==0:
            return -1
        p = -1
        s = -1
        if self.pProb()<=1:
            p = self.pProb()
            s = self.sProb()
            return p, s
        if 'modelL' in req and 'predictLongData' in req:
            modelL = req['modelL']
            if self.pTime()>100 and 'minP' in req:
                predictLongData = req['predictLongData']
                y = predictLongData(modelL[0], waveform['pWaveform'][staIndexNew])
                #print(y.shape)
                p = y[int(i0-5):int(i0+5)].max()
            if self.sTime()>100 and 'minS' in req:
                predictLongData = req['predictLongData']
                y = predictLongData(modelL[1], waveform['sWaveform'][staIndexNew])
                s = y[int(i0-5):int(i0+5)].max()
        #print(p,s)
        return p, s

#MLI  1976/01/01 01:29:39.6 -28.61 -177.64  59.0 6.2 0.0 KERMADEC ISLANDS REGION 
class Quake(list):
    '''
    a basic class 'Quake' used for quake's information and station records
    it's baed on list class
    the elements in list are composed by Records
    the basic information: loc, time, ml, filename, randID
    we use randID to distinguish time close quakes
    you can set the filename (file path) storing the waveform by yourself in .mat form
    otherwise, we would generate the file like dayNum/time_randID.mat

    we provide way for set by line/WLX/Mat and from IRIS/NDK
    '''
    def __init__(self, loc=[-999, -999,10], time=-1, randID=None, filename=None, ml=None):
        super(Quake, self).__init__()
        self.loc = [0,0,0]
        self.loc[:len(loc)]=loc
        self.time = time
        self.ml=ml
        if randID != None:
            self.randID=randID
        else:
            self .randID=int(10000*np.random.rand(1))
        if filename != None:
            self.filename = filename
        else:
            self.filename = self.getFilename()
    def __repr__(self):
        return self.summary()
    def summary(self,count=0,inShort=False):
        ml=-9
        if self.ml!=None:
            ml=self.ml
        if inShort:
            return '%.2f %.4f %.4f %.1f %.1f '%(self.time, self.loc[0]\
                , self.loc[1], self.loc[2], self.ml)
        Summary='quake: %f %f %f num: %d index: %d randID: %d filename: \
            %s %f %f\n' % (self.loc[0], self.loc[1],\
            self.time, self.num(),count, self.randID, \
            self.filename,ml,self.loc[2])
        return Summary

    def getFilename(self):
        dayDir = str(int(self.time/86400))+'/'
        return dayDir+str(int(self.time))+'_'+str(self.randID)+'.mat'

    def append(self, addOne):
        if isinstance(addOne, Record):
            super(Quake, self).append(addOne)
        else:
            raise ValueError('should pass in a Record class')

    def copy(self):
        quake=Quake(loc=self.loc,time=self.time,randID=self.randID,filename=self.filename\
            ,ml=self.ml)
        for record in self:
            quake.append(record.copy())
        return quake

    def set(self,line,mode='byLine'):
        if mode=='byLine':
            return self.setByLine(line)
        elif mode=='fromNDK':
            return self.setFromNDK(line)
        elif mode=='fromIris':
            return self.setFromIris(line)
        elif mode=='byWLX':
            return self.setByWLX(line)
        elif mode=='byMat':
            return self.setByMat(line)
    def setByLine(self,line):
        if len(line) >= 4:
            if len(line)>12:
                self.loc = [float(line[1]), float(line[2]),float(line[-1])]
            else:
                self.loc = [float(line[1]), float(line[2]),0.]
            if len(line)>=14:
                self.ml=float(line[-2])
            self.time = float(line[3])
            self.randID=int(line[9])
            self.filename=line[11]
        return self
    def setFromNDK(self,line):
        sec=float(line[22:26])
        self.time=UTCDateTime(line[5:22]+"0.00").timestamp+sec
        self.loc=[float(line[27:33]),float(line[34:41]),float(line[42:47])]
        m1=float(line[48:55].split()[0])
        m2=float(line[48:55].split()[-1])
        self.ml=max(m1,m2)
        self.filename = self.getFilename()
        return self
    def setFromIris(self,line):
        line=line.split('|')
        self.time=UTCDateTime(line[1]).timestamp
        self.loc[0]=float(line[2])
        self.loc[1]=float(line[3])
        self.loc[2]=float(line[4])
        self.ml=float(line[10])
        self.filename = self.getFilename()
        return self

    def setByWLX(self,line,staInfos=None):
        self.time=UTCDateTime(line[:22]).timestamp
        self.loc=[float(line[23:29]),float(line[30:37]),float(line[38:41])]
        self.ml=float(line[44:])
        self.filename=self.getFilename()
        return self

    def setByMat(self,q):
        pTimeL=q[0].reshape(-1)
        sTimeL=q[1].reshape(-1)
        PS=q[2].reshape(-1)
        self.randID=1
        self.time=matTime2UTC(PS[0])
        self.loc=PS[1:4]
        self.ml=PS[4]
        self.filename=self.getFilename()
        for i in range(len(pTimeL)):
            pTime=0
            sTime=0
            if pTimeL[i]!=0:
                pTime=matTime2UTC(pTimeL[i])
                if sTimeL[i]!=0:
                    sTime=matTime2UTC(sTimeL[i])
                self.append(Record(i,pTime,sTime))
        return self

    def getReloc(self,line):
        self.time=self.tomoTime(line)
        self.loc[0]=float(line[1])
        self.loc[1]=float(line[2])
        self.loc[2]=float(line[3])
        return self

    def tomoTime(self,line):
        m=int(line[14])
        sec=float(line[15])
        return UTCDateTime(int(line[10]),int(line[11]),int(line[12])\
            ,int(line[13]),m+int(sec/60),sec%60).timestamp

    def calCover(self,staInfos,maxDT=None):
        '''
        calculate the radiation coverage
        '''
        coverL=np.zeros(360)
        for record in self:
            if record.pTime()==0 and record.sTime()==0:
                continue
            if maxDT!=None:
                if record.pTime()-self.time>maxDT:
                    continue
            staIndex= int(record[0])
            la=staInfos[staIndex]['la']
            lo=staInfos[staIndex]['lo']
            dep=staInfos[staIndex]['dep']/1e3
            delta,dk,Az=self.calDelta(la,lo,dep)
            R=int(60/(1+dk/200)+60)
            N=((int(Az)+np.arange(-R,R))%360).astype(np.int64)
            coverL[N]=coverL[N]+1
        L=((np.arange(360)+180)%360).astype(np.int64)
        coverL=np.sign(coverL)*np.sign(coverL[L])*(coverL+coverL[L])
        coverRate=np.sign(coverL).sum()/360
        return coverRate

    def getPTimeL(self,staInfos):
        timePL=np.zeros(len(staInfos))
        for record in self:
            if record.pTime()!=0:
                timePL[record.getStaIndex()]=record.pTime()
        return timePL
        
    def getSTimeL(self,staInfos):
        timeSL=np.zeros(len(staInfos))
        for record in self:
            if record.sTime()!=0:
                timeSL[record.getStaIndex()]=record.sTime()
        return timeSL

    def findTmpIndex(self,staIndex):
        count=0
        for record in self:
            if int(record.getStaIndex())==int(staIndex):
                return count
            count+=1
        return -999

    def setRandIDByMl(self):
        self.randID=int(np.floor(np.abs(10+self.ml)*100));
        namePre=self.filename.split('_')[0]
        self.filename=namePre+'_'+str(self.randID)+'.mat'

    def outputWLX(self):
        Y=getYmdHMSj(UTCDateTime(self.time))
        tmpStr=Y['Y']+'/'+Y['m']+'/'+Y['d']+' '+\
        Y['H']+':'+Y['M']+':'+'%05.2f'%(self.time%60)+\
        ' '+'%6.3f %7.3f %3.1f M %3.1f'%(self.loc[0],\
            self.loc[1],self.loc[2],self.ml)
        return tmpStr

    def num(self,maxDT=None): 
        count=0
        for record in self:
            if record.pTime()==0 and record.sTime()==0:
                continue
            if maxDT!=None:
                if record.pTime()-self.time>maxDT:
                    continue
            count+=1
        return count

    def calDelta(self,la,lo,dep=0):
        D=DistAz(la,lo,self.loc[0],self.loc[1])
        delta=D.getDelta()
        dk=D.degreesToKilometers(delta)
        dk=np.linalg.norm(np.array([dk,self.loc[2]+dep]))
        Az=D.getAz()
        return delta,dk,Az

    def calDeltaDt(self,staInfos):
        deltaP=[]
        deltaS=[]
        dtP=[]
        dtS=[]
        dkP=[]
        dkS=[]
        for record in self:
            staIndex=record.staIndex()
            la=staInfos[staIndex]['la']
            lo=staInfos[staIndex]['lo']
            dep=staInfos[staIndex]['dep']/1e3
            delta,dk,Az=self.calDelta(la,lo,dep)
            if record.pTime()>0:
                deltaP.append(delta)
                dkP.append(dk)
                dtP.append(record.pTime()-self.time)
            if record.sTime()>0:
                deltaS.append(delta)
                dtS.append(record.sTime()-self.time)
                dkS.append(dk)
        return deltaP,dkP,dtP,deltaS,dkS,dtS

    def calTimePS(self):
        PS=[]
        for record in self:
            if record.pTime()>0 and record.sTime()>0:
                PS.append([record.pTime()-self.time,\
                    record.sTime()-self.time])
        return PS

    def selectByReq(self,req={},waveform=None,isPrint=False):
        '''
        bTime eTime ((R inR) minDis maxDis) minMl maxDep \
        minCover outDir (maxErr locator --maxDTLoc) minSta
        
        Record:
            maxDT (oTime)
            minSNR (tLBe tlAf)
        
        '''
        maxDTLoc=None
        if 'maxDTLoc' in req:
            maxDTLoc=req['maxDTLoc']
        if 'bTime' in req:
            if self.time<req['bTime']:
                if isPrint:
                    print('before bTime')
                return False
        if 'eTime' in req:
            if self.time > req['eTime']:
                if isPrint:
                    print('after bTime')
                return False
        if 'R' in req:
            R=req['R']
            if 'inR' in req and (self.loc[0]<R[0] or \
                self.loc[0]>R[1] or self.loc[1]<R[2] or self.loc[1]>R[3]):
                if req['inR']:
                    if isPrint:
                        print('not in R')
                    return False
            if 'minDis' in req or 'maxDis' in req:
                midLa=(R[0]+R[1])/2
                midLo=(R[2]+R[3])/2
                delta=self.calDelta(midLa,midLo)
                if 'minDis' in req:
                    if delta<req['minDis'] :
                        if isPrint:
                            print('less than minDis')
                        return False  
                if 'maxDis' in req:
                    if delta>req['maxDis'] :
                        if isPrint:
                            print('more than maxDis')
                        return False
        if 'minMl' in req:
            if self.ml<req['minMl']:
                if isPrint:
                    print('less than minML')
                return False
        if 'maxDep' in req:
            if self.loc[2]>req['maxDep']:
                if isPrint:
                    print('deeper than maxDeep')
                return False
        if 'minSta' in req:
            if self.num(maxDTLoc)< req['minSta']:
                if isPrint:
                    print('not enough records')
                return False
        if 'minCover' in req and 'staInfos' in req:
            if self.calCover(req['staInfos'],maxDTLoc)<req['minCover']:
                if isPrint:
                    print('not enough ray coverage')
                return False
        for record in self:
            req['oTime']=self.time
            for record in self:
                record.selectByReq(req,waveform,self.time)
        if 'minCover' in req and 'staInfos' in req:
            if self.calCover(req['staInfos'],maxDTLoc)<req['minCover']:
                if isPrint:
                    print('not enough ray coverage')
                return False
        if 'outDir' in req :
            if not os.path.exists(req['outDir']+self.filename):
                if isPrint:
                    print('no file')
                return False
        if 'maxErr' in req and 'locator' in req:
            if 'maxDTLoc' in req:
                maxDT=req['maxDTLoc']
            else:
                maxDT=30
            self,res=req['locator'].locate(self,maxDT=maxDT,\
                isDel=True,maxErr=req['maxErr'])
            if 'maxStd' in req:
                if res>req['maxStd']:
                    if isPrint:
                        if res == 999:
                            print('not enough record to locate')
                        else:
                            print('too much erro')
                    return False
        if 'minSta' in req:
            if self.num(maxDTLoc) < req['minSta']:
                if isPrint:
                    print('not enough records')
                return False
        return True

    def loadWaveform(self,matDir='output',\
        isCut=False,index0=-250,index1=250,\
        f=[-1,-1],filtOrder=2):
        fileName = matDir+'/'+self.filename
        waveform=sio.loadmat(fileName)
        if f[0]>0:
            if len(waveform['deltaL'])==0:
                print('wrong delta')
                return None
            f0=0.5/waveform['deltaL'].max()
            b, a = signal.butter(filtOrder, [f[0]/f0,f[1]/f0], 'bandpass')
            waveform['pWaveform']=signal.filtfilt(b,a,waveform['pWaveform'],axis=1)
            waveform['sWaveform']=signal.filtfilt(b,a,waveform['sWaveform'],axis=1)
        if isCut:
            i0=np.where(waveform['indexL'][0]==index0)[0][0]
            i1=np.where(waveform['indexL'][0]==index1)[0][0]
            waveform['pWaveform']=waveform['pWaveform'][:,i0:i1,:].\
            astype(np.float32)
            waveform['sWaveform']=waveform['sWaveform'][:,i0:i1,:].\
            astype(np.float32)
            waveform['indexL']=[waveform['indexL'][0][i0:i1]]
        return waveform

    def isP(self,staIndex):
        for record in self:
            if record.getStaIndex()==staIndex and record.pTime()>0:
                return True
        return False

    def isS(self,staIndex):
        for record in self:
            if record.getStaIndex()==staIndex and record.sTime()>0:
                return True
        return False

    def saveWaveform(self,staL, quakeIndex, matDir='output/'\
    ,index0=-500,index1=500,dtype=np.float32):
        indexL = np.arange(index0, index1)
        iNum=indexL.size
        fileName = matDir+'/'+self.filename
        loc=self.loc
        dayDir=os.path.dirname(fileName)
        if not os.path.exists(dayDir):
            os.mkdir(dayDir)
        pWaveform = np.zeros((len(self), iNum, 3),dtype=dtype)
        sWaveform = np.zeros((len(self), iNum, 3),dtype=dtype)
        staIndexL = np.zeros(len(self))
        pTimeL = np.zeros(len(self))
        sTimeL = np.zeros(len(self))
        deltaL = np.zeros(len(self))
        ml=0
        sACount=0
        for i in range(len(self)):
            record = self[i]
            staIndex = record.getStaIndex()
            pTime = record.pTime()
            sTime = record.sTime()
            staIndexL[i] = staIndex
            pTimeL[i] = pTime
            sTimeL[i] = sTime
            if pTime != 0:
                try:
                    pWaveform[i, :, :] = staL[staIndex].data.getDataByTimeLQuick\
                    (pTime + indexL*staL[staIndex].data.delta)
                except:
                    print("wrong p wave")
                else:
                    pass
                deltaL[i] = staL[staIndex].data.delta
            if sTime != 0:
                try:
                    sWaveform[i, :, :] = staL[staIndex].data.getDataByTimeLQuick\
                    (sTime + indexL*staL[staIndex].data.delta)
                except:
                    print("wrong s wave")
                else:
                    deltaL[i]= staL[staIndex].data.delta
                    dk=DistAz(staL[staIndex].loc[0],staL[staIndex].loc[1],loc[0],loc[1]).getDelta()*111.19
                    sA=getSA(sWaveform[i, :, :])*staL[staIndex].data.delta
                    ml=ml+np.log10(sA)+1.1*np.log10(dk)+0.00189*dk-2.09-0.23
                    sACount+=1
        if sACount==0:
            ml=-999
        else:
            ml/=sACount
        sio.savemat(fileName, {'time': self.time, 'loc': self.loc, \
            'staIndexL': staIndexL, 'pTimeL': pTimeL, 'pWaveform': \
            pWaveform, 'sTimeL': sTimeL, 'sWaveform': sWaveform, \
            'deltaL': deltaL, 'indexL': indexL,'ml':ml})
        return ml

    def calML(self, staInfos, matDir='output/',minSACount=3,waveform=None):
        if not isinstance(waveform,dict):
            waveform=self.loadWaveform(matDir=matDir)
        ml=0
        loc=self.loc
        sACount=0
        if len(waveform['staIndexL'])<=0:
            ml=-999
            print('wrong ml')
            return ml
        for i in range(len(waveform['staIndexL'][0])):
            if waveform['sTimeL'][0][i]!=0:
                staIndex=int(waveform['staIndexL'][0][i])
                dk=DistAz(staInfos[staIndex]['la'],\
                    staInfos[staIndex]['lo'],loc[0],loc[1]).getDelta()*111.19
                if dk<30 or dk>300:
                    continue
                sA=getSA(waveform['sWaveform'][i, :, :])*waveform['deltaL'][0][i]
                ml=ml+np.log10(sA)+1.1*np.log10(dk)+0.00189*dk-2.09-0.23
                sACount+=1
        if sACount<minSACount:
            ml=-999
        else:
            ml/=sACount
        self.ml=ml
        print(sACount,ml)
        return ml

    def plotByMat(self,staInfos,matDir='NM/output20190901/',mul=0.15,\
        compP=2,compS=1,outDir='NM/output20190901V2/',waveform=None,\
        figFileName=None,isAl=True,isAd=False,vp=6,vs=6/1.73,isK=False,
        az0=0,daz0=-9):
        #loadWaveformByQuake(quake,matDir='output',isCut=False,index0=-250,index1=250,f=[-1,-1]):
        
        if isAl:
            al=0
        else:
            al=1
        if isAd:
            ad=1
        else:
            ad=0
        if not isinstance(waveform,dict):
            waveform=self.loadWaveform(matDir=matDir)
        pWaveform=waveform['pWaveform'].reshape((-1,\
            waveform['pWaveform'].shape[1],1,3))
        sWaveform=waveform['sWaveform'].reshape((-1,\
            waveform['sWaveform'].shape[1],1,3))
        pWaveform/=pWaveform.max(axis=1,keepdims=True)/mul
        sWaveform/=sWaveform.max(axis=1,keepdims=True)/mul
        staIndexL=waveform['staIndexL'][0].astype(np.int64)
        shift=1
        eqLa=self.loc[0]
        eqLo=self.loc[1]
        maxDis=0
        minDis=100
        pTimeL=self.getPTimeL(staInfos)
        sTimeL=self.getPTimeL(staInfos)
        #ml=self.calML(staInfos,waveform=waveform)
        print(self.summary(inShort=True))
        plt.figure(figsize=[6,4])
        for i in range(len(staIndexL)):
            staInfo=staInfos[staIndexL[i]]
            timeL=waveform['indexL'][0]*waveform['deltaL'][0][i]
            Dis=DistAz(eqLa,eqLo,staInfo['la'],staInfo['lo'])
            dis=Dis.getDelta()
            az=Dis.getAz()
            if daz0>0:
                daz=az-az0
                if np.abs(daz%180)>daz0:
                    continue
                shift=np.sign(np.cos(daz/180*np.pi))
            if isK:
                dz=(staInfo['dep']/1e3+self.loc[2])/111.19
                dis=(dz**2+dis**2)**0.5
            maxDis=max(dis,maxDis)
            minDis=min(dis,minDis)

            if pTimeL[staIndexL[i]]!=0 and waveform['pTimeL'][0][i]!=0:
                plt.subplot(2,1,1)
                adTime=-dis*111.19/vp*ad
                plt.plot(timeL+ al*(waveform['pTimeL'][0][i]-self.time)+adTime\
                    ,pWaveform[i,:,0,compP]\
                    +dis*shift,'k',linewidth=0.3)
                oTime=(timeL[0]+timeL[-1])/2
                #plt.plot(np.array([oTime,oTime])+ waveform['pTimeL'][0][i]-quake.time,np.array([dis-0.5,dis+0.5]),'-.r',linewidth=0.5)
            if sTimeL[staIndexL[i]]!=0 and waveform['sTimeL'][0][i]!=0:
                plt.subplot(2,1,2)
                adTime=-dis*111.19/vs*ad
                plt.plot(timeL+ al*(waveform['sTimeL'][0][i]-self.time)+adTime,\
                    sWaveform[i,:,0,compS]\
                    +dis*shift,'k',linewidth=0.3)
                oTime=(timeL[0]+timeL[-1])/2
                #plt.plot(np.array([oTime,oTime])+ waveform['sTimeL'][0][i]-quake.time,np.array([dis-0.5,dis+0.5]),'-.r',linewidth=0.5)
        plt.subplot(2,1,1)
        #plt.title(self.summary(inShort=True))
        for i in range(2):
            plt.subplot(2,1,i+1)
            if isAl:
                plt.xlim([timeL[0],timeL[-1]])
                h0,=plt.plot(np.array([oTime,oTime])*0,\
                    np.array([minDis-2.5,maxDis+2.5]),\
                    '-.k',linewidth=1)
                if i==0:
                    plt.legend((h0,),['p'])
                else:
                    plt.legend((h0,),['s'])
                plt.ylim(minDis-0.5,maxDis+0.8)
            if i==1:
                plt.xlabel('t/s')
            plt.ylabel('D/Rad')
        if not isinstance(figFileName,str):
            dirName=os.path.dirname('%s/%s.eps'%(\
                outDir,self.filename[:-4]))
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            plt.savefig('%s/%s.eps'%(outDir,self.filename[:-4]))
            plt.savefig('%s/%s.png'%(outDir,self.filename[:-4]),\
                dpi=300)
        else:
            dirName=os.path.dirname(figFileName)
            if not os.path.exists(dirName):
                os.makedirs(dirName)
            plt.savefig(figFileName,dpi=300)
        plt.close()

    def __gt__(self,self1):
        return self.time>self1.time

    def __ge__(self,self1):
        return self.time>=self1.time

    def __eq__(self,self1):
        return self.time==self1.time

    def __lt__(self,self1):
        return self.time<self1.time

    def __le__(self,self1):
        return self.time<=self1.time

class RecordCC(Record):
    def __init__(self,staIndex=-1, pTime=-1, sTime=-1, pCC=-1, sCC=-1, pM=-1, pS=-1, sM=-1, sS=-1):
        self.append(staIndex)
        self.append(pTime)
        self.append(sTime)
        self.append(pCC)
        self.append(sCC)
        self.append(pM)
        self.append(pS)
        self.append(sM)
        self.append(sS)

    def getPCC(self):
        return self[3]

    def getSCC(self):
        return self[4]

    def getPM(self):
        return self[5]

    def getPS(self):
        return self[6]

    def getSM(self):
        return self[7]

    def getSS(self):
        return self[8]

    def getPMul(self):
        return (self.getPCC()-self.getPM())/self.getPS()

    def getSMul(self):
        return (self.getSCC()-self.getSM())/self.getSS()



class QuakeCC(Quake):
    '''
    expand the basic class Quake for storing the quake result 
    of MFT and WMFT
    the basic information include more: cc,M,S,tmpName
    '''
    def __init__(self, cc=-9, M=-9, S=-9, loc=[-999, -999,10],  time=-1, randID=None, \
        filename=None,tmpName=None,ml=None):
        super(Quake, self).__init__()
        self.cc=cc
        self.M=M
        self.S=S
        self.loc = [0,0,0]
        self.loc[:len(loc)]=loc
        self.time = time
        self.tmpName=tmpName
        self.ml=ml
        if randID != None:
            self.randID=randID
        else:
            self .randID=int(10000*np.random.rand(1))+10000*2
        if filename != None:
            self.filename = filename
        else:
            self.filename = self.getFilename()

    def append(self, addOne):
        if isinstance(addOne, RecordCC):
            super(Quake, self).append(addOne)
        else:
            raise ValueError('should pass in a RecordCC class')

    def getMul(self, isNum=False):
        if not isNum:
            return (self.cc-self.M)/self.S
        else:
            return (self.cc-self.M)/self.S*((len(self)+10)**0.5)

    def calCover(self,staInfos,minCC=0.5):
        '''
        calculate the radiation coverage which have higher CC than minCC
        '''
        coverL=np.zeros(360)
        for record in self:
            if (record.pTime()>0 or record.sTime())>0 \
            and(record.getPCC()>minCC or record.getSCC()>minCC): 
                staIndex= int(record[0])
                Az=DistAz(staInfos[staIndex]['la'],staInfos[staIndex]['lo'],\
                    self.loc[0],self.loc[1]).getAz()
                dk=DistAz(staInfos[staIndex]['la'],staInfos[staIndex]['lo'],\
                    self.loc[0],self.loc[1]).getDelta()*111.19
                R=int(45/(1+dk/50)+45)
                N=((int(Az)+np.arange(-R,R))%360).astype(np.int64)
                coverL[N]=coverL[N]+1
        L=((np.arange(360)+180)%360).astype(np.int64)
        coverL=np.sign(coverL)*np.sign(coverL[L])*(coverL+coverL[L])
        coverRate=np.sign(coverL).sum()/360
        return coverRate

    def summary(self,count=0):
        ml=-9
        if self.ml!=None:
            ml=self.ml
        Summary='quake: %f %f %f num: %d index: %d randID: %d filename: \
            %s %s %f %f %f %f %f\n' % (self.loc[0], self.loc[1],\
            self.time, len(self),count, self.randID, \
            self.filename,str(self.tmpName),self.cc,self.M,self.S,ml,self.loc[2])
        return Summary

    def setByLine(self,line):
        if len(line) >= 4:
            self.loc = [float(line[1]), float(line[2]),float(line[-1])]
            if len(line)>=18:
                self.ml=float(line[-2])
                self.S=float(line[-3])
                self.M=float(line[-4])
                self.cc=float(line[-5])
                self.tmpName=line[-6]
            self.time = float(line[3])
            self.randID=int(line[9])
            self.filename=line[11]
        return self

    def setByMat(self,q):
        '''
        0('tmpIndex', 'O'), ('name', 'O'), ('CC', 'O'), 
        3('mean', 'O'), ('std', 'O'), ('mul', 'O'), 
        6('pCC', 'O'), ('sCC', 'O'), ('pM', 'O'), 
        9('sM', 'O'), ('pS', 'O'), ('sS', 'O'), 
        12('PS', 'O'), ('pTime', 'O'), ('sTime', 'O'),
        15('pD', 'O'), ('sD', 'O'), ('tmpTime', 'O'), 
        18('oTime', 'O'), ('eTime', 'O')])
        '''
        self.cc=q[2][0,0]
        self.S=q[4][0,0]
        self.M=q[3][0,0]
        self.tmpName=str(q[1][0])
        pTimeL=q[13].reshape(-1)
        sTimeL=q[14].reshape(-1)
        pCC=q[6].reshape(-1)
        sCC=q[7].reshape(-1)
        pM=q[8].reshape(-1)
        sM=q[9].reshape(-1)
        pS=q[10].reshape(-1)
        sS=q[11].reshape(-1)
        PS=q[12].reshape(-1)
        self.time=matTime2UTC(PS[0])
        self.loc=PS[1:4]
        self.ml=PS[4]
        self.filename=self.getFilename()
        for i in range(len(pTimeL)):
            pTime=0
            sTime=0
            if pTimeL[i]!=0:
                pTime=matTime2UTC(pTimeL[i])
                if sTimeL[i]!=0:
                    sTime=matTime2UTC(sTimeL[i])
                self.append(RecordCC(i,pTime,sTime,pCC[i],sCC[i],pM[i],pS[i],sM[i],sS[i]))
        return self

    def setByWLX(self,line,tmpNameL=None):
        lines=line.split()
        self.time=UTCDateTime(lines[1]+' '+lines[2]).timestamp
        self.loc=[float(lines[3]),float(lines[4]),float(lines[5])]
        self.ml=float(lines[6])
        self.cc=float(lines[7])
        tmpStr=lines[10]
        tmpTime=UTCDateTime(int(tmpStr[0:4]),int(tmpStr[4:6]),int(tmpStr[6:8]),int(tmpStr[8:10]),\
                int(tmpStr[10:12]),int(tmpStr[12:14])).timestamp
        tmpName=str(int(tmpTime))
        if tmpNameL!=None:       
            for tmpN in tmpNameL:
                if tmpName == tmpN.split('/')[-1].split('_')[0]:
                    tmpName=tmpN
        self.tmpName=tmpName
        self.filename=self.getFilename()
        return self

def plotDeltaDt(quakeL,staInfos,figName1='delta-dt.png',\
    figName2='dk-dt.png',figName3='P-S.png'):
    quakeL=quakeL
    deltaP=[]
    deltaS=[]
    dtP=[]
    dtS=[]
    dkP=[]
    dkS=[]
    PS=[]
    for quake in quakeL:
        deltaPTmp,dkPTmp,dtPTmp,deltaSTmp,dkSTmp,dtSTmp\
        =quake.calDeltaDt(staInfos)
        PS+=quake.calTimePS()
        deltaP += deltaPTmp
        deltaS += deltaSTmp
        dkP += dkPTmp
        dkS += dkSTmp
        dtP += dtPTmp
        dtS += dtSTmp
    deltaP = np.array(deltaP)
    deltaL = np.array(deltaS)
    dtP = np.array(dtP)
    dtS = np.array(dtS)
    plt.plot(deltaP,dtP,'.b',markersize=0.05)
    plt.plot(deltaS,dtS,'.r',markersize=0.05)
    plt.xlabel('delta/rad')
    plt.ylabel('t/s')
    plt.savefig(figName1,dpi=400)
    plt.close()
    plt.plot(dkP,dtP,'.b',markersize=0.05)
    plt.plot(dkS,dtS,'.r',markersize=0.05)
    plt.xlabel('dk/rad')
    plt.ylabel('t/s')
    plt.savefig(figName2,dpi=400)
    plt.close()
    PS=np.array(PS)
    timeP=PS[:,0]
    timeS=PS[:,1]
    plt.plot(timeP,timeS,'.k',markersize=0.05)
    plt.xlabel('pTime/s')
    plt.ylabel('sTime/s')
    plt.savefig(figName3,dpi=400)
    plt.close()

def mat2quake(filename):
    mat=sio.loadmat(filename)
    time=mat['time'][0][0]
    loc=mat['loc'][0]
    ml=mat['ml'][0]
    randID=int(filename.split('.')[-2].split('_')[-1])
    quake=Quake(loc,time,randID=randID,ml=ml)
    for i in range(mat['staIndexL'][0].size):
        staIndex=mat['staIndexL'][0][i]
        pTime=mat['pTimeL'][0][i]
        sTime=mat['sTimeL'][0][i]
        quake.append(Record(staIndex,pTime,sTime))
    return quake

def removeBadSta(quakeLs,badStaLst=[]):
    for quakeL in quakeLs:
        for quake in quakeL:
            for record in quake:
                if record.getStaIndex() in badStaLst:
                    record.clear()
                    print("setOne")

def getQuakeLD(quakeL):
    D={}
    for quake in quakeL:
        D[quake.filename]=quake 
    return D

def divideRange(L, N):
    dL = (L[1]-L[0])/N
    subL = np.arange(0, N+1)*dL+L[0]
    dR = {'minL': subL[0:-1], 'maxL': subL[1:], 'midL': \
    (subL[0:-1]+subL[1:])/2}
    return dR

def findLatterOne(t0, tL):
    indexL=np.where(tL>t0)
    if indexL.shape[0]>0:
        return indexL[0], tL(indexL[0])
    return -1, -1

class arrival(object):
    def __init__(self, time):
        self.time = time

class quickTaupModel:
    '''
    pre-calculated taup model for quick usage
    '''
    def __init__(self, modelFile='include/iaspTaupMat'):
        matload = sio.loadmat(modelFile)
        self.interpP = interp.interp2d(matload['dep'].reshape([-1]),\
            matload['deg'].reshape([-1]), matload['taupMatP'])
        self.interpS = interp.interp2d(matload['dep'].reshape([-1]),\
            matload['deg'].reshape([-1]), matload['taupMatS'])
        pTime0 = matload['taupMatP'][:,0].reshape([-1])
        sTime0 = matload['taupMatS'][:,0].reshape([-1])
        dTime = sTime0-pTime0
        dL = np.argsort(dTime)
        self.interpO = interp.interp1d(dTime[dL], pTime0[dL], \
            fill_value='extrapolate')

    def get_travel_times(self,dep, deg, phase_list='p'):
        if phase_list[0]=='p':
            a = arrival(self.interpP(dep, deg)[0])
        else:
            a = arrival(self.interpS(dep, deg)[0])
        return [a]

    def get_orign_times(self, pIndex, sIndex, delta):
        return pIndex-self.interpO((sIndex-pIndex)*delta)/delta

def getQuakeInfoL(quakeL,loc0=np.array([37.8,140,0])):
    PS=np.zeros((len(quakeL),5))
    for i in range(len(quakeL)):
        PS[i,0]=quakeL[i].time
        PS[i,1:4]=quakeL[i].loc-loc0
        PS[i,4]=quakeL[i].ml
    return PS

def saveQuakeLs(quakeLs, filename,mod='o'):
    with open(filename, 'w') as f:
        if mod=='o':
            count = 0
            for quakeL in quakeLs:
                f.write('day\n')
                for quake in quakeL:
                    f.write(quake.summary(count))
                    count +=1
                    for record in quake:
                        if record.pTime()+record.sTime()>0:
                            f.write(record.summary())
        if mod=='ML':
            for quake in quakeLs:
                f.write(quake.outputWLX()+'\n')

def getStaNameIndex(staInfos):
    staNameIndex=dict()
    for i in range(len(staInfos)):
        staNameIndex.update({staInfos[i]['sta']: i})
    return staNameIndex

def readQuakeLs(filenames, staInfos, mode='byQuake', \
    N=200, dH=0, isQuakeCC=False,key=None,minMul=8,\
    tmpNameL=None):
    '''
    read quakeLst in different form to differen form
    '''
    def getQuakeLFromDay(day):
        quakeL=list()
        for q in day:
            if not isQuakeCC:
                quake=Quake().set(q,'byMat')
            else:
                quake=QuakeCC().set(q,'byMat')
            if quake.time>0:
                quakeL.append(quake)
        return quakeL

    if mode=='byMatDay':
        dayMat=sio.loadmat(filenames)
        if key ==None:
            key=dayMat.keys()[-1]
        dayMat=dayMat[key][-1]
        quakeLs=list()
        for day in dayMat:
            day=day[-1][-1]
            quakeLs.append(getQuakeLFromDay(day))
        return quakeLs

    if mode=='byMat':
        dayMat=sio.loadmat(filenames)
        if key == None:
            key=dayMat.keys()[-1]
        day=dayMat[key][-1]
        return getQuakeLFromDay(day)
        
    if mode=='byWLX':
        quakeL=[]
        with open(filenames) as f:
            if not isQuakeCC:
                for line in f.readlines():
                    quakeL.append(Quake().set(line,'byWLX'))
            else:
                for line in f.readlines()[1:]:
                    quakeL.append(QuakeCC().set(line,'byWLX'))
        return quakeL

    with open(filenames) as f:
        lines = f.readlines()
        quakeLs = list()
        if mode == 'byQuake':
            for line in lines:
                line = line.split()
                if line[0] == 'day':
                    quakeL = list()
                    quakeLs.append(quakeL)
                    continue
                if line[0].split()[0] == 'quake:':
                    if isQuakeCC:
                        quake=QuakeCC()
                    else:
                        quake = Quake()
                    quake.set(line,'byLine')
                    quakeL.append(quake)
                    continue
                if isQuakeCC:
                    record=RecordCC()
                else:
                    record=Record()
                quake.append(record.set(line,'byLine'))
            return quakeLs
        if mode == 'bySta':
            staArrivalP = [[] for i in range(N)]
            staArrivalS = [[] for i in range(N)]
            maxStaCount = 0
            for line in lines:
                line = line.split()
                if line[0] == 'day':
                    continue
                if line[0].split()[0] == 'quake:':
                    continue
                staIndex = int(line[0])
                maxStaCount = max(staIndex+1, maxStaCount)
                timeP = float(line[1])
                timeS = float(line[2])
                if timeP > 1:
                    staArrivalP[staIndex].append(timeP)
                if timeS > 1:
                    staArrivalS[staIndex].append(timeS)
            return staArrivalP[0:maxStaCount], staArrivalS[0:maxStaCount]

        if mode == 'SC':
            staNameIndex = getStaNameIndex(staInfos)
            staArrivalP = [[] for i in range(len(staInfos))]
            staArrivalS = [[] for i in range(len(staInfos))]
            for line in lines:
                if line[0] == '2':
                    continue
                lineCell = line.split(',')
                time=UTCDateTime(lineCell[1]).timestamp+dH*3600
                staIndex = staNameIndex[lineCell[0].strip()]
                if lineCell[2][0] == 'P':
                    staArrivalP[staIndex].append(time)
                else:
                    staArrivalS[staIndex].append(time)
            return staArrivalP, staArrivalS

        if mode == 'NDK':
            quakeLs=[]
            time0=0
            for i in range(0,len(lines),5):
                line=lines[i]
                quake=Quake()
                quake=quake.set(line,'fromNDK')
                time1=int(quake.time/86400)
                print(time1)
                if time1>time0:
                    quakeLs.append([])
                    time0=time1
                quakeLs[-1].append(quake)
            return quakeLs

        if mode=='IRIS':
            quakeLs=[]
            time0=0
            for i in range(0,len(lines)):
                line=lines[i]
                quake=Quake()
                quake=quake.set(line,'fromIris')
                time1=int(quake.time/86400)
                print(time1)
                if time1>time0:
                    quakeLs.append([])
                    time0=time1
                quakeLs[-1].append(quake)
            return quakeLs

def readQuakeLsByP(filenamesP, staInfos=None, mode='byQuake',  N=200, dH=0,key=None\
    ,isQuakeCC=False):
    quakeLs=[]
    fileL=glob(filenamesP)
    fileL.sort()
    for file in fileL:
        quakeLs=quakeLs+readQuakeLs(file, staInfos, mode=mode,  N=N, dH=dH,\
            key=key,isQuakeCC=isQuakeCC)
    return quakeLs

def compareTime(timeL, timeL0, minD=2):
    timeL = np.sort(np.array(timeL))
    timeL0 = np.sort(np.array(timeL0))
    dTime = list()
    count0 = 0
    count = 0
    N = timeL.shape[0]
    N0 = timeL0.shape[0]
    i = 0
    for time0 in timeL0:
        if i == N-1:
            break
        if time0 < timeL[0]:
            continue
        if time0 > timeL[-1]:
            break
        count0 += 1
        for i in range(i, N):
            if abs(timeL[i]-time0)<minD:
                dTime.append(timeL[i]-time0)
                count += 1
            if i == N-1:
                break
            if timeL[i] > time0:
                break
    return dTime, count0, count

def getSA(data):
    data=data-data.mean()
    return data.cumsum(axis=0).max()

'''
select earthquakes for different uses
'''
def selectQuakeByDis(quakeLs,R,staInfos,minDis=0,maxDis=20,outDir='output/'\
        ,bTime=UTCDateTime(1970,1,1).timestamp,\
        eTime=UTCDateTime(2100,1,1).timestamp,minMl=5):
    req={'bTime':bTime,'eTime':eTime,'minMl':minMl,'minDis':minDis,\
        'maxDis': maxDis, 'R':R , 'inR':True, 'staInfos':staInfos}
    quakeLNew=[]
    for quakeL in quakeLs:
        for quake in quakeL:
            if quake.selectByReq(req=req):
                quakeLNew.append(quake)
    return quakeLNew

def selectQuake(quakeLs,R,staInfos,minSta=10,laN=30,loN=30,maxCount=25,minCover=0.8,\
    maxDep=60,isF=True,outDir='output/'):
    quakeL=[]
    quakeNumL=[]
    req={'R':R,'inR':True,'minSta':minSta,'staInfos':staInfos,'minCover':minCover\
        ,'maxDep':maxDep,'outDir':outDir}
    laL=np.arange(R[0],R[1],(R[1]-R[0])/laN)
    loL=np.arange(R[2],R[3],(R[3]-R[2])/loN)
    aM=np.zeros((laN+1,loN+1))
    for quakeLTmp in quakeLs:
        for quake in quakeLTmp:
            if quake.selectByReq(req):
                quakeL.append(quake)
                quakeNumL.append(quake.num())
    L=np.argsort(-np.array(quakeNumL))
    quakeLNew=[]
    for i in L:
        quake=quakeL[i]
        laIndex=np.argmin(np.abs(quake.loc[0]-laL))
        loIndex=np.argmin(np.abs(quake.loc[1]-loL))
        if aM[laIndex][loIndex]>=maxCount:
            continue
        aM[laIndex][loIndex]+=1
        quakeLNew.append(quake)
    return quakeLNew

def selectRecord(quake,maxDT=35):
    quake.selectByReq(req={'maxDT':maxDT})
    return quake

def resampleWaveform(waveform,n):
    #b,a=signal.bessel(2,1/n*0.8)
    waveform['deltaL']=waveform['deltaL']*n
    waveform['indexL']=[waveform['indexL'][0][0:-1:n]/n]
    N=waveform['indexL'][0].size
    '''
    waveform['pWaveform']=signal.resample(signal.filtfilt(\
        b,a,waveform['pWaveform'][:,0:N*n,:],axis=1),N,axis=1)
    waveform['sWaveform']=signal.resample(signal.filtfilt(\
        b,a,waveform['sWaveform'][:,0:N*n,:],axis=1),N,axis=1)
    return waveform
    '''
    waveform['pWaveform']=signal.resample(waveform['pWaveform'][:,0:N*n,:],N,axis=1)
    waveform['sWaveform']=signal.resample(waveform['sWaveform'][:,0:N*n,:],N,axis=1)

def resampleWaveformL(waveformL,n):
    for waveform in waveformL:
        waveform=resampleWaveform(waveform,n)
    return waveformL

def getMLFromWaveformL(quakeL, staInfos, matDir='output/',isQuick=False):
    count=0
    for quake in quakeL:
        count+=1
        print(count)
        if quake.ml!=None and isQuick:
            if quake.ml >-2 and quake.ml<3:
                continue
        quake.ml=quake.getML(staInfos,matDir=matDir)

def getMLFromWaveformLs(quakeLs, staInfos, matDir='output/'):
    for quakeL in quakeLs:
        for quake in quakeL:
            quake.ml=quake.getML(quake,staInfos,matDir=matDir)
            
def saveQuakeLWaveform(staL, quakeL, matDir='output/',\
    index0=-500,index1=500,dtype=np.float32):
    if not os.path.exists(matDir):
        os.mkdir(matDir)
    for i in range(len(quakeL)):
         quakeL[i].ml=quakeL[i].saveWaveform(staL, i,\
          matDir=matDir,index0=index0,index1=index1,dtype=dtype)

def loadWaveformL(quakeL,matDir='output',isCut=False,index0=-250,\
    index1=250,f=[-1,-1],filtOrder=2,nptype=np.float32,convert=1,
    resampleN=-1):
    waveformL=[]
    tmpNameL=[]
    for quake in quakeL:
        tmpNameL.append(quake.filename)
        waveformL.append(quake.loadWaveform(\
            matDir=matDir,isCut=isCut,index0=index0,\
            index1=index1,f=f,filtOrder=filtOrder))
    if resampleN>0:
        resampleWaveformL(waveformL,resampleN)
    for waveform in waveformL:
        waveform['pWaveform']=(waveform['pWaveform']*convert).astype(nptype)
        waveform['sWaveform']=(waveform['sWaveform']*convert).astype(nptype)
    return waveformL, tmpNameL

loadWaveformLByQuakeL=loadWaveformL

def genTaupTimeM(model='iasp91', N=6, matFile='include/iaspTaupMat',\
    depN=200, degN=4000):
    managers = Manager()
    resL = [managers.list() for i in range(N)]
    taupDict = {'dep': None,'deg': None, 'taupMatP': None, 'taupMatS':None}
    taupDict['deg'] = np.power(np.arange(degN)/degN,2)*180
    taupDict['dep'] = np.concatenate([np.arange(depN/2),\
        np.arange(depN/2)*10+depN/2])
    taupDict['taupMatP']= np.zeros((degN, depN))
    taupDict['taupMatS']= np.zeros((degN, depN))
    taupM = taup.TauPyModel(model=model)
    processL=list()
    for i in range(N):
        processTmp = Process(target=_genTaupTimeM, \
            args=(taupDict, i, N, taupM, resL[i]))
        processTmp.start()
        processL.append(processTmp)

    for processTmp in processL:
        processTmp.join()

    for index in range(N):
        i=0
        for depIndex in range(index, depN, N):
            for degIndex in range(degN):
                taupDict['taupMatP'][degIndex, depIndex]=resL[index][i][0]
                taupDict['taupMatS'][degIndex, depIndex]=resL[index][i][1]
                i += 1
    sio.savemat(matFile, taupDict)
    return taupDict

def _genTaupTimeM(taupDict, index, N, taupM, resL):
    depN = len(taupDict['dep'][:])
    degN = len(taupDict['deg'][:])
    for depIndex in range(index, depN, N):
        print(depIndex)
        dep = taupDict['dep'][depIndex]
        for degIndex in range(degN):
            deg = taupDict['deg'][degIndex]
            if degIndex==0:
                print(depIndex, degIndex, getEarliest(taupM.get_travel_times\
                    (dep, deg, ['p', 'P', 'PP', 'pP'])))
            resL.append([getEarliest(taupM.get_travel_times(dep, deg, \
                ['p', 'P', 'PP', 'pP'])), getEarliest(taupM.get_travel_times\
            (dep, deg, ['s', 'S', 'SS', 'sS']))])

def getEarliest(arrivals):
        time=10000000
        if len(arrivals)==0:
            print('no phase')
            return 0
        for arrival in arrivals:
            time = min(time, arrival.time)
        return time

'''
this part is designed for getting sta info (loc and file path)
'''
def getStaAndFileLst(dirL,filelstName,staFileName):
    def writeMFileInfo(f,mFile,dayDir,staName):
        comp=['BH','BH','BH']
        fileIndex=mFile[0:6]+'_'+comp[int(mFile[-3])-1]
        f.write("%s %s %s\n"%(staName,fileIndex,dayDir))
    staLst={}
    with open(filelstName,'a') as f:
        for Dir in dirL:
            for tmpDir in glob(Dir+'/[A-Z]*'):
                try:
                    if not os.path.isdir(tmpDir):
                        continue
                    for staName in os.listdir(tmpDir):
                        staDir=tmpDir+'/'+staName+'/'
                        if not os.path.isdir(staDir):
                            continue
                        staLogsP=staDir+'*log'
                        try:
                            la,lo,laD,loD,z,zD = getLocByLogsP(staLogsP)
                        except:
                            print('wrong')
                            continue
                        else:
                            pass
                        print(tmpDir,staName,la,lo,laD,loD,z,zD)
                        if la !=999 and lo!=999:
                            if staName in staLst:
                                if laD+loD < staLst[staName][2]+staLst[staName][3]:
                                    staLst[staName]=[la,lo,laD,loD,tmpDir,z,zD]
                            else:
                                staLst[staName]=[la,lo,laD,loD,tmpDir,z,zD]
                        continue
                        for dayDir in glob(staDir+'R*'):
                            for hourDir in glob(dayDir+'/'+'00'):
                                for mFile in glob(hourDir+'/*1.m'):
                                    mFile=mFile.split('/')[-1]
                                    writeMFileInfo(f,mFile,dayDir,staName)
                except:
                    print("***********erro*********")
                else:
                    pass
    with open(staFileName,'w+') as staFile :
        for staName in staLst:
            staFile.write("hima %s BH %f %f %f %f %f %f\n"%(staName,staLst[staName][1], \
                staLst[staName][0],staLst[staName][3],staLst[staName][2],\
                staLst[staName][5],staLst[staName][6]))

def getThem():
    fileL=['/media/jiangyr/XIMA_I/XIMA_I/','/media/jiangyr/XIMA_II/','/media/jiangyr/XIMA_III/XIMA_III/',\
            '/media/jiangyr/XIMA_IV/XIMA_IV/','/media/jiangyr/XIMA_V/XIMA_V/']
    getStaAndFileLst(fileL,'fileLst','staLst')

def checkFile(filename):
    sta={}
    with open(filename) as f:
        for line in f.readlines():
            staIndex=line.split(' ')[1].split('/')[-1]
            if staIndex in sta:
                print(line)
                print(sta[staIndex])
            else:
                sta[staIndex]=line

def loadFileLst(staInfos,filename):
    staDict={}
    for staTmp in staInfos:
        staDict.update({staTmp['sta']:{}})
    with open(filename) as f:
        for line in f.readlines():
            infos=line.split()
            if infos[0] in staDict:
                staDict[infos[0]].update({infos[1]:infos[2]})
    return staDict

def getStaInArea(staInfos,fileNew,R):
    with open(fileNew,'w+') as f:
        for staInfo in staInfos:
            if staInfo['la']>=R[0] and \
                    staInfo['la']<=R[1] and \
                    staInfo['lo']>=R[2] and \
                    staInfo['lo']<=R[3]:
                f.write("%s %s %s %f %f 0 0 %f\n"%(staInfo['net'],staInfo['sta'],\
                    staInfo['comp'][0][0:2],staInfo['lo'],staInfo['la'],staInfo['dep']))

def getStaInfoFromSac(sacDir, staInfoFile='staLstSac',staInfos=[],\
    dataDir='dataDis/',R=[33,44,106,116]):
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    staLst={}
    for staInfo in staInfos:
        staLst[staInfo['sta']]=[staInfo['la'],staInfo['lo']]
    with open(staInfoFile,'a') as f:
        for sacDirTmp in glob(sacDir+'/20*00/'):
            for sacFile in glob(sacDirTmp+'/*.BHE'):
                fileName=sacFile.split('/')[-1]
                net,station=fileName.split('.')[0:2]
                print(net,station)
                if station not in staLst:
                    sac=obspy.read(sacFile)[0].stats.sac
                    print("%s %s %s %f %f 0 0 %f\n"%('hima',station,\
                        'BH',sac['stlo'],sac['stla'],sac['stel']))
                    f.write("%s %s %s %f %f 0 0 %f\n"%('hima',station,\
                        'BH',sac['stlo'],sac['stla'],sac['stel']))
                    staLst[station]=[sac['stla'],sac['stlo']]
                plt.plot(staLst[station][1],staLst[station][0],'.r')
            figName=dataDir+sacDirTmp.split('/')[-2]+'.png'
            plt.xlim(R[2:])
            plt.ylim(R[:2])
            plt.savefig(figName)
            plt.close()

def toTmpNameD(tmpNameL):
    tmpNameD={}
    for i in range(len(tmpNameL)):
        tmpNameD[tmpNameL[i]]=i
    return tmpNameD

def synQuake(staInfos,loc,indexL=[],N=20,modelFile='include/iaspTaupMat',\
    oTime=0,isS=True,ml=-9):
    quake=Quake(time=oTime)
    quake.loc=loc
    quake.ml=ml
    if isinstance(modelFile,str):
        timeM=quickTaupModel(modelFile)
    else:
        timeM=modelFile
    if len(indexL)==0:
        indexL=np.floor(np.random.rand(N)*len(staInfos)).astype(np.int64)
    for index in indexL:
        staLa=staInfos[index]['la']
        staLo=staInfos[index]['lo']
        dep=staInfos[index]['dep']/1000+loc[2]
        delta=DistAz(quake.loc[0],quake.loc[1],\
                    staLa,staLo).delta
        timeP=timeM.get_travel_times(dep,delta,'p')[0].time+oTime
        timeS=0
        if isS:
            timeS=timeM.get_travel_times(dep,delta,'s')[0].time+oTime
        quake.append(Record(index,timeP,timeS))
    return quake

def synQuakeV2(quake,staInfos,indexL=[],N=20,modelFile='include/iaspTaupMat',\
   isS=True):
    quake.clear()
    loc=quake.loc
    oTime=quake.time
    if isinstance(modelFile,str):
        timeM=quickTaupModel(modelFile)
    else:
        timeM=modelFile
    if len(indexL)==0:
        indexL=np.floor(np.random.rand(N)*len(staInfos)).astype(np.int64)
    for index in indexL:
        staLa=staInfos[index]['la']
        staLo=staInfos[index]['lo']
        dep=staInfos[index]['dep']/1000+loc[2]
        delta=DistAz(quake.loc[0],quake.loc[1],\
                    staLa,staLo).delta
        timeP=timeM.get_travel_times(dep,delta,'p')[0].time+oTime
        timeS=0
        if isS:
            timeS=timeM.get_travel_times(dep,delta,'s')[0].time+oTime
        if not isinstance(quake,QuakeCC):
            quake.append(Record(index,timeP,timeS))
        else:
            quake.append(RecordCC(index,timeP,timeS))
    return quake

def analysis(quakeLs,staInfos,outDir='fig/',minSta=6,maxDep=80,\
    bTime=UTCDateTime(2014,1,1).timestamp,eTime=UTCDateTime(2017,10,1).timestamp):
    dayNum=int((eTime-bTime)/86400)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    mlL=[]
    timeL=[]
    #staNum=np.zeros(len(staInfos))
    staDayNum=np.zeros((len(staInfos),dayNum))
    for quakeL in quakeLs:
        for quake in quakeL:
            if len(quake)<minSta:
                continue
            if quake.loc[2]>maxDep:
                continue
            if quake.ml==None:
                continue
            if quake.ml<-5:
                continue
            mlL.append(quake.ml)
            timeL.append(quake.time)
            dayIndex=int((quake.time-bTime)/86400)
            for record in quake:
                staDayNum[record.getStaIndex(),dayIndex]+=1
    #plt.subplot(2,2,1)
    plt.hist(np.array(mlL),bins=40)
    plt.title('ml dis')
    plt.savefig(outDir+'mlDis.png')
    plt.xlabel('ml')
    plt.ylabel('count')
    plt.close()

    #plt.subplot(2,2,2)
    #plt.hist(staDayNum.sum(axis=1),bins=20)
    plt.bar(np.arange(len(staInfos)),staDayNum.sum(axis=1))
    plt.xlabel('sta Index')
    plt.ylabel('record count')
    plt.title('station record num')
    plt.savefig(outDir+'staRecordNum.png')
    plt.close()
    
    #plt.subplot(2,2,3)
    i=0
    for staInfo in staInfos:
        plt.plot(staInfo['lo'],staInfo['la'],'^b', markersize=np.log(1+np.sign(staDayNum[i,:]).sum()))
        i+=1
    plt.title('station dis with date num')
    plt.savefig(outDir+'staDis.png')
    plt.close()

    #plt.subplot(2,2,4)
    i=0
    plt.pcolor(np.sign(staDayNum).transpose())
    plt.xlabel('sta Index')
    plt.ylabel('date from '+UTCDateTime(bTime).strftime('%Y%m%d'))
    plt.title('sta date')
    plt.savefig(outDir+'staDate.png')
    plt.close()

    pTimeL=[]
    sTimeL=[]
    depL=[]
    for quakeL in quakeLs:
        for quake in quakeL:
            for record in quake:
                pTime=record.pTime()
                sTime=record.sTime()
                if sTime > 0:
                    pTimeL.append(pTime-quake.time)
                    sTimeL.append(sTime-quake.time)
                    depL.append(quake.loc[2])
    pTimeL=np.array(pTimeL)
    sTimeL=np.array(sTimeL)
    depL=np.array(depL)
    #plt.plot(pTimeL,sTimeL,'.',markersize=0.02,markerfacecolor='blue',markeredgecolor='blue',alpha=0.8)
    plt.scatter(pTimeL,sTimeL,0.01,depL,alpha=0.5,marker=',')
    plt.title('pTime-sTime')
    plt.xlabel('pTime')
    plt.ylabel('sTime')
    plt.savefig(outDir+'pTime-sTime.png',dpi=300)
    plt.close()


def calSpec(x,delta):
    spec=np.fft.fft(x)
    N=x.size
    fL=np.arange(N)/N*1/delta
    return spec,fL

def plotSpec(waveform,isNorm=True,plotS=False,alpha=0.1):
    cL="rgb"
    N=1
    if plotS:
        N=2
    for i in range(len(waveform['staIndexL'][0])):
        if waveform['pTimeL'][0][i]!=0:
            for comp in range(3):
                W=waveform['pWaveform'][i,:,comp]
                print(N)
                plt.subplot(N,3,comp+1)

                if W.max()<=0:
                    continue
                W=W/np.linalg.norm(W)
                spec,fL=calSpec(W,waveform['deltaL'][0,i])
                plt.plot(fL,abs(spec),cL[comp],alpha=alpha)
                plt.xlim([0,fL[-1]/2])
                if plotS:
                    plt.subplot(N,3,3+comp+1)
                else:
                    continue
                W=waveform['sWaveform'][i,:,comp]
                if W.max()<=0:
                    continue
                W=W/np.linalg.norm(W)
                spec,fL=calSpec(W,waveform['deltaL'][0,i])
                plt.plot(fL,abs(spec),cL[comp],alpha=alpha)
                plt.xlim([0,fL[-1]/2])
    plt.show()

'''
this part is designed to prepare gan's data
'''
def preGan(waveformL,maxCount=10000,indexL=np.arange(-200,200)):
    realFile='gan/input/waveform4.mat'
    resFile='gan/output/genWaveform.mat'
    modelFile='gan/model/phaseGen'
    boardDir='gan/boardDir/'
    if not os.path.exists(os.path.dirname(realFile)):
        os.mkdir(os.path.dirname(realFile))
    if not os.path.exists(os.path.dirname(resFile)):
        os.mkdir(os.path.dirname(resFile))
    if not os.path.exists(os.path.dirname(modelFile)):
        os.mkdir(os.path.dirname(modelFile))
    if not os.path.exists(os.path.dirname(boardDir)):
        os.mkdir(os.path.dirname(boardDir))
    waveformAll=np.zeros((maxCount,indexL.size,1,3))
    count = 0
    for waveform in waveformL:
        count
        if count>=maxCount:
                break
        for i in range(waveform['pWaveform'].shape[0]):
            index0=np.where(waveform['indexL'][0,:]==0)[0]+int(200-100*np.random.rand(1))
            w=waveform['pWaveform'][i,indexL+index0,:]
            if min(abs(w).max(axis=1))<=0:
                print('badWaveform')
                continue
            w=w/np.linalg.norm(w)*3
            waveformAll[count,:,:,:]=w.reshape((indexL.size,1,3))
            count+=1
            if count>=maxCount:
                break
    sio.savemat(realFile,{'waveform':waveformAll[:count,:,:,:]})

def plotGan():
    realFile='gan/input/waveform4.mat'
    resFile='gan/output/genWaveform.mat'
    outDir=os.path.dirname(resFile)
    waveformO=sio.loadmat(realFile)['waveform']
    waveformGen=sio.loadmat(resFile)['genWaveform']
    timeL=np.arange(400)*0.02
    for i in range(4):
        plt.subplot(2,2,i+1)
        for comp in range(3):
            plt.plot(timeL,waveformO[i,:,0,comp]+2-comp,'b')
        plt.yticks(np.arange(3),['Z','N','E'])
        plt.suptitle('real waveforms')
    plt.savefig(outDir+'/real.png')
    plt.close()

    for i in range(4):
        plt.subplot(2,2,i+1)
        for comp in range(3):
            plt.plot(timeL,waveformGen[i,:,0,comp]+2-comp,'b')
        plt.yticks(np.arange(3),['Z','N','E'])
        plt.suptitle('fake waveforms')
    plt.savefig(outDir+'/fake.png')
    plt.close()

'''
this part is designed to compare with WLX's method
'''
def findNear(time,timeL,maxD=5):
    if np.abs(timeL-time).min()<maxD:
        return np.abs(timeL-time).argmin()
    else:
        return -1

def compareQuakeL(quakeL1, quakeL2,recordF=None):
    PS1=getQuakeInfoL(quakeL1)
    PS2=getQuakeInfoL(quakeL2)
    #print(PS1[:,0])
    #print(PS2[:,0])
    h1,=plt.plot(PS1[:,2],PS1[:,1],'.b')
    h2,=plt.plot(PS2[:,2],PS2[:,1],'.r')
    for i in range(len(quakeL1)):
        index= findNear(PS1[i,0],PS2[:,0])
        if index>=0:
            laL=np.array([PS1[i,1],PS2[index,1]])
            loL=np.array([PS1[i,2],PS2[index,2]])
            hh,=plt.plot(loL,laL,'g')
            if recordF!=None:
                recordF.write("%02d %02d %.2f %7.4f %7.4f %7.4f %4.2f %4.2f\n"%(i,\
                    index,PS1[i,0],PS1[i,0]-PS2[index,0],PS1[i,1]-PS2[index,1],\
                    PS1[i,2]-PS2[index,2],quakeL1[i].cc,quakeL2[index].cc))
    return h1,h2,hh

def onlyQuake(quakeL,quakeRefL):
    qL=[]
    for quake in quakeL:
        isM=False
        for quakeRef in quakeRefL:
            if quake.tmpName==quakeRef.filename:
                isM=True
                break
        if isM:
            qL.append(quake)
    return qL

def analysisMFT(quakeL1,quakeL2,quakeRefL,filename='wlx/MFTCompare.png',recordName='tmp.res'):
    quakeL1=onlyQuake(quakeL1,quakeRefL)
    quakeL2=onlyQuake(quakeL2,quakeRefL)
    with open(recordName,'w+') as f:
        f.write("i1 i2 oTime          dTime   dLa     dLo    cc1  cc2\n")
        h1,h2,hh=compareQuakeL(quakeL1,quakeL2,recordF=f)
    PSRef=getQuakeInfoL(quakeRefL)
    hRef,=plt.plot(PSRef[:,2],PSRef[:,1],'^k') 
    plt.legend((h1,h2,hh,hRef),('Jiang','WLX','same','Ref'))
    plt.xlabel('lo')
    plt.ylabel('la')
    plt.xlim([-0.025,0.025])
    plt.ylim([-0.025,0.025])
    plt.savefig(filename,dpi=300)
    plt.close()

def analysisMFTAll(quakeL1,quakeL2,quakeRefL,outDir='wlx/compare/'):
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for quakeRef in quakeRefL:
        figName=outDir+'/'+quakeRef.filename.split('/')[-1].split('.')[0]+'.png'
        recordName=outDir+'/'+quakeRef.filename.split('/')[-1].split('.')[0]+'.res'
        analysisMFT(quakeL1,quakeL2,[quakeRef],filename=figName,recordName=recordName)

def dTimeQuake(quake,quakeRef,staInfos,filename='test.png',quake2=None):
    ishh1=False
    for R in quake:
        for RR in quakeRef:
            if R.getStaIndex()==RR.getStaIndex() and R.pTime()!=0 and RR.pTime()!=0:
                staInfo=staInfos[R.getStaIndex()]
                AZ=(DistAz(quake.loc[0],quake.loc[1],\
                    staInfo['la'],staInfo['lo']).getAz()+180)/180*np.pi
                dTime=(R.pTime()-quake.time-(RR.pTime()-quakeRef.time))/30
                dLa=dTime*np.cos(AZ)
                dLo=dTime*np.sin(AZ)
                laL=np.array([0,dLa])+quake.loc[0]
                loL=np.array([0,dLo])+quake.loc[1]
                if R.getPCC()>0.5:
                    hh,=plt.plot(loL,laL,'k')
                else:
                    hh1,=plt.plot(loL,laL,'y')
                    ishh1=True
    h,=plt.plot(quake.loc[1],quake.loc[0],'.b')
    hR,=plt.plot(quakeRef.loc[1],quakeRef.loc[0],'^r')
    hL=(h,hR,hh)
    nameL=('jiang','Ref','dTime/30 cc>0.5')
    if ishh1:
        hL=hL+(hh1,)
        nameL=nameL+('dTime/30 cc<0.5',)
    if quake2!=None:
        h2,=plt.plot(quake2.loc[1],quake2.loc[0],'*b')
        hL=hL+(h2,)
        nameL=nameL+('WLX',)
    plt.legend(hL,nameL)
    plt.xlim([quakeRef.loc[1]-0.015,quakeRef.loc[1]+0.015])
    plt.ylim([quakeRef.loc[0]-0.015,quakeRef.loc[0]+0.015])
    plt.savefig(filename,dpi=300)
    plt.close()

def dTimeQuakeByRef(quakeL,quakeRef,staInfos,outDir='wlx/dTime/',quakeL2=None):
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    quakeL=onlyQuake(quakeL,[quakeRef])
    quakeL2=onlyQuake(quakeL2,[quakeRef])
    PS2=getQuakeInfoL(quakeL2)
    for quake in quakeL:
        figName=outDir+'/'+quake.filename.split('/')[-1].split('.')[0]+'.png'
        index= findNear(quake.time,PS2[:,0])
        if index>=0:
            dTimeQuake(quake,quakeRef,staInfos,filename=figName,quake2=quakeL2[index])

'''
to generate the training set

'''
def genKY(sacDir='/home/jiangyr/accuratePickerV3/hiNet/event/',delta=0.02):
    N=20000
    pxFile='PX3.mat'
    pyFile='PY3.mat'
    sxFile='SX3.mat'
    syFile='SY3.mat'
    dTimeFile='dTimeV2.mat'
    PX=np.zeros((N,3000,3))
    PY=np.zeros((N,3000,3))
    SX=np.zeros((N,3000,3))
    SY=np.zeros((N,3000,3))
    dTime=np.zeros((N,2))
    pCount=0
    sCount=0
    indexO=1500
    iO=400000
    indexL=np.arange(-1500,1500)
    pY0=np.exp(-((np.arange(800000)-iO)/5)**2)
    sY0=np.exp(-((np.arange(800000)-iO)/10)**2)
    f0=int(1/delta)
    for monthDir in glob(sacDir+'2*/'):
        print(monthDir)
        for eventDir in glob(monthDir+'/D*/'):
            file=glob(eventDir+'/D*.txt')[0]
            with open(file,encoding='ISO-8859-1') as f:
                lines=f.readlines()
                print(lines[1])
                la=float(lines[1].split()[-1][:-1])
                lo=float(lines[2].split()[-1][:-1])
            print(la,lo)
            for sacZ in glob(eventDir+'/*U.SAC'):
                strL='ENZ'
                sacE=sacZ[:-5]+'E'+sacZ[-5+1:]
                sacN=sacZ[:-5]+'N'+sacZ[-5+1:]
                sacFileL=[sacE,sacN,sacZ]
                #print(sacFileL)
                isF=True
                for sac in sacFileL:
                    if not os.path.exists(sac):
                        isF=False
                #if os.path.exists()
                if not isF:
                    continue
                sacL=[obspy.read(sac)[0] for sac in sacFileL]
                downSampleRate=delta/sacL[0].stats['delta']
                stla=sacL[0].stats['sac']['stla']
                stlo=sacL[0].stats['sac']['stlo']
                if downSampleRate<1 or np.abs(round(downSampleRate) -downSampleRate)>0.01 :
                    continue
                downSampleRate=int(round(downSampleRate))
                bTime=sacL[0].stats['starttime'].timestamp
                eTime=sacL[0].stats['endtime'].timestamp
                pTime=bTime+sacL[0].stats['sac']['t0']-sacL[0].stats['sac']['b']
                sTime=bTime+sacL[0].stats['sac']['t1']-sacL[0].stats['sac']['b']

                oTime=pTime+(np.random.rand()-0.5)*40
                oTimeP=min(max(oTime,bTime+30.1),eTime-30.1)
                dIndexP=int(round((oTimeP-pTime)/delta))
                dIndexS=int(round((oTimeP-sTime)/delta))
                if pTime>bTime+1000:
                    continue
                if sTime>bTime+1000:
                    continue
                #print(sacL[0].stats['sac']['t0'],pTime,oTimeP,bTime,eTime,dIndexP)
                PY[pCount,:,0]=pY0[iO+dIndexP+indexL]
                PY[pCount,:,1]=sY0[iO+dIndexS+indexL]
                PY[pCount,:,2]=1-PY[pCount,:,0]-PY[pCount,:,1]
                oTime=sTime+(np.random.rand()-0.5)*40
                oTimeS=min(max(oTime,bTime+30.1),eTime-30.1)
                dIndexP=int(round((oTimeS-pTime)/delta))
                dIndexS=int(round((oTimeS-sTime)/delta))
                #PY[pCount,:,0]=pY0[iO+dIndexP+indexL]
                SY[sCount,:,0]=sY0[iO+dIndexP+indexL]
                SY[sCount,:,1]=sY0[iO+dIndexS+indexL]
                SY[sCount,:,2]=1-SY[sCount,:,0]-SY[sCount,:,1]
                try :
                    for comp in range(3):
                        sac=sacL[comp]
                        sac.decimate(downSampleRate)
                        oIndexP=int(round((oTimeP-bTime)*f0))
                        PX[pCount,:,comp]=sac.data[oIndexP+indexL]
                        oIndexS=int(round((oTimeS-bTime)*f0))
                        SX[sCount,:,comp]=sac.data[oIndexS+indexL]
                except:
                    print('wrong')
                    continue
                else:
                    pass

                dTime[pCount,0]=sTime-pTime
                dTime[pCount,1]=DistAz(la,lo,stla,stlo).degreesToKilometers\
                    (DistAz(la,lo,stla,stlo).getDelta())
                pCount+=1
                sCount+=1
                if pCount%10==0:
                    print(pCount,sCount)
                if pCount%10==0:
                    x=PX[pCount-1,:,0]
                    plt.plot(x/x.max(),linewidth=0.3)
                    plt.plot(PY[pCount-1,:,0]-1,linewidth=0.3)
                    plt.plot(PY[pCount-1,:,1]-2,linewidth=0.3)
                    plt.plot(PY[pCount-1,:,2]-3,linewidth=0.3)
                    x=SX[sCount-1,:,0]
                    plt.plot(x/x.max()-4,linewidth=0.3)
                    plt.plot(SY[sCount-1,:,0]-5,linewidth=0.3)
                    plt.plot(SY[sCount-1,:,1]-6,linewidth=0.3)
                    plt.plot(SY[sCount-1,:,2]-7,linewidth=0.3)
                    plt.savefig('fig/%d.png'%pCount,dpi=500)
                    plt.close()
                if pCount==N:
                    break
            if pCount==N:
                break
        if pCount==N:
            break
    if pCount<10000:
        print('No')
        return
    h5py.File(pxFile,'w')['px']=PX[:pCount]
    h5py.File(pyFile,'w')['py']=PY[:pCount]
    h5py.File(sxFile,'w')['sx']=SX[:pCount]
    h5py.File(syFile,'w')['sy']=SY[:pCount]
    h5py.File(dTimeFile,'w')['dTime']=dTime[:pCount]


def genKYSC():
    delta=0.02
    sacDir='/home/jiangyr/WC_mon78/'
    phaseLst='../phaseLst0'
    staName0=''
    fileName='SC.mat'
    date0=0
    N=1000
    PX=np.zeros((N,3000,3))
    PY=np.zeros((N,3000,1))
    SX=np.zeros((N,3000,3))
    SY=np.zeros((N,3000,1))
    pCount=0
    sCount=0
    indexO=1500
    iO=2500
    indexL=np.arange(-1500,1500)
    pY0=np.exp(-((np.arange(5000)-iO)/5)**2)
    sY0=np.exp(-((np.arange(5000)-iO)/10)**2)
    with open(phaseLst) as f:
        for line in f.readlines():
            if line[0]=='2':
                continue
            staName=line[:3]
            Y=int(line[5:9])
            M=int(line[9:11])
            D=int(line[11:13])
            h=int(line[13:15])
            m=int(line[15:17])
            s=float(line[17:21])
            phase=line[22]
            time=UTCDateTime(Y,M,D,h,m,s).timestamp-3600*8
            if staName!=staName0 or np.floor(time/86400)!=date0:
                print(staName0,time)
                #XX.MXI.2008189000000.BHZ
                timeStr=UTCDateTime(time).strftime('%Y%j')
                print('%s/*.%s*%s*Z'%(sacDir,staName,timeStr))
                tmp=glob('%s/*.%s*%s*Z'%(sacDir,staName,timeStr))
                if len(tmp)<1:
                    continue
                sacZ=tmp[0]
                sacN=sacZ[:-1]+'N'
                sacE=sacZ[:-1]+'E'
                sacFileL=[sacE,sacN,sacZ]
                isF=True
                for sac in sacFileL:
                    if not os.path.exists(sac):
                        isF=False
                #if os.path.exists()
                if not isF:
                    continue
                staName0=staName
                date0=np.floor(time/86400)
                sacL=[obspy.read(sac)[0] for sac in sacFileL]
                [sac.decimate(2) for sac in sacL]
            oTime=time+(np.random.rand()-0.5)*40
            try:
                for comp in range(3):
                    sac=sacL[comp]
                    bTime=sac.stats['starttime'].timestamp
                    #print(bTime)
                    bIndex=int(round((oTime-bTime)/delta))
                    dIndex=int(round((oTime-time)/delta))
                    #print(bIndex,dIndex)
                    if phase =='P':
                        PX[pCount,:,comp]=sac.data[bIndex+indexL]
                        PY[pCount,:,0]=pY0[iO+dIndex+indexL]
                    else:
                        SX[sCount,:,comp]=sac.data[bIndex+indexL]
                        SY[sCount,:,0]=sY0[iO+dIndex+indexL]
            except:
                print('wrong')
                continue
            else:
                pass
            if phase=='P':
                if ((PX[pCount,:,:]**2).sum(axis=0)==0).sum()>0:
                    print('wrong data')
                    continue
                pCount+=1
            else:
                if ((SX[sCount,:,:]**2).sum(axis=0)==0).sum()>0:
                    print('wrong data')
                    continue
                sCount+=1

            if pCount%100==0:
                    print(pCount,sCount)
                    print((PX[max(pCount-1,0),:,:]**2).sum(axis=0))
            if pCount%10==0:
                x=PX[pCount-1,:,0]
                plt.plot(x/x.max(),linewidth=0.3)
                plt.plot(PY[pCount-1,:,0]-1,linewidth=0.3)
                x=SX[sCount-1,:,0]
                plt.plot(x/x.max()-2,linewidth=0.3)
                plt.plot(SY[sCount-1,:,0]-3,linewidth=0.3)
                plt.savefig('fig/SC_%d.png'%pCount,dpi=500)
                plt.close()
            if pCount==N:
                break
        sio.savemat(fileName,{'px':PX[:pCount],'py':PY[:pCount],'sx':SX[:sCount],'sY':SY[:sCount]})

'''
test and output part 
'''
def fileRes(fileName,phase='p'):
    data=sio.loadmat(fileName)
    y0JP=data['%sy0'%phase]
    yJP=data['out%sy'%phase]
    y0SC=data['%sy0Test'%phase]
    ySC=data['out%syTest'%phase]
    return cmpY(y0JP,yJP,phase=phase),cmpY(y0SC,ySC,phase=phase)
def cmpY(y0,y,delta=0.02,phase='p'):
    if phase=='p':
        i0=250
        i1=1250
    else:
        i0=500
        i1=1500#1250
    y0=y0.reshape(-1,y0.shape[1])
    y=y.reshape(-1,y.shape[1])
    index0=y0.argmax(axis=1)
    v0=y0[:,i0:i1].max(axis=1)
    index=y[:,i0:i1].argmax(axis=1)+i0
    v=y[:,i0:i1].max(axis=1)
    t0=index0*delta
    t0[v0<0.99]=-100000
    t=index*delta
    V0=v0
    V=v
    return t0,V0,t,V

def fileResJP(fileName,phase='p',dTimeFile='data/dTime.mat',threshold=0.5,\
    isPlot=True,isSave=True,N0=3,N1=1,NL=[1,2,3],strL='abc',\
    isLegend=False,iL=np.arange(0,2000,100)):
    if phase=='p':
        cI=0
    else:
        cI=1
    data=sio.loadmat(fileName)
    y0JP=data['%sy0'%phase]
    x0JP=data['out%sx'%phase]
    yJP=data['out%sy'%phase]
    yJPO=yJP
    x0JPO=x0JP
    print(yJP.shape)
    if y0JP.shape[-1]>1 and y0JP.shape[-1]<5:
        y0JP=y0JP[:,:,:,cI:cI+1]
    if yJP.shape[-1]>1 and yJP.shape[-1]<5:
        yJP=yJP[:,:,:,cI:cI+1]
    dTime=h5py.File(dTimeFile)['dTime'][1000:11000]
    if phase=='p':
        i0=x0JP.shape[1]/8
        i1=x0JP.shape[1]/8*7
    else:
        i0=x0JP.shape[1]/4
        i1=x0JP.shape[1]/4*3
    if x0JP.shape[1]==1500:
        if phase=='p':
            i0=2000/8
            i1=2000/8*7
        else:
            i0=2000/4
            i1=2000/4*3
    if phase=='p':
        i0=250
        i1=1000#1250
    else:
        i0=250
        i1=1000#1250
    
    SNR,index0L,indexL=calSNR(x0JP,y0JP,yJP)
    if not isPlot:
        return index0L,indexL,x0JPO,yJPO
    if isSave:
        plt.close()
        plt.figure(figsize=[6,12])
    ax1=plt.subplot(N0,N1,NL[0])
    ax1,ax2=DP(index0L,indexL,dTime,np.arange(0,80,10),phase,threshold,0.02,\
        i0,i1,ax1,is30=True)
    ax1.set_xlabel('ts-tp/s')
    ax1.set_ylabel('Number')
    plt.text(0.01,0.93,'(%s)'%strL[0],transform=ax1.transAxes)
    ax2.set_ylabel('Rate')
    ax2.set_ylim([0.6,1.03])
    ax1=plt.subplot(N0,N1,NL[1])
    ax1,ax2=DP(index0L,indexL,SNR,np.arange(-1,6,1),phase,threshold,0.02,\
        i0,i1,ax1)
    ax1.set_xlabel('log(SNR)')
    ax1.set_ylabel('Number')
    plt.text(0.01,0.93,'(%s)'%strL[1],transform=ax1.transAxes)
    ax2.set_ylabel('Rate')
    ax2.set_ylim([0.1,1.1])
    ax1=plt.subplot(N0,N1,NL[2])
    ax1,ax2=DP(index0L,indexL,index0L*0.02,iL*0.02,phase,threshold, 0.02,\
        0,2000,ax1,isLegend=isLegend)
    ax1.set_xlabel('arrival time/s ')
    ax1.set_ylabel('Number')
    plt.text(0.01,0.93,'(%s)'%strL[2],transform=ax1.transAxes)
    ax2.set_ylabel('Rate')
    ax2.set_ylim([0,1.1])
    if isSave:
        plt.savefig('fig/dTime_SNR_time_%s.eps'%phase)
        plt.close()

def plotFileRes():
    plt.close()
    fig=plt.figure(figsize=[12.3,7.3])
    fig.tight_layout()
    fileResJP('resDir/resDataP_320000_0-2-15.mat',threshold=0.5,\
        isSave=False,N0=3,N1=2,NL=[1,3,5],strL='ace')
    fileResJP('resDir/resDataS_320000_0-2-15.mat',phase='s',\
        threshold=0.5,isSave=False,N0=3,N1=2,NL=[2,4,6],\
        strL='bdf',isLegend=True)
    plt.savefig('fig/fileResAll.eps')
def plotFileResSoft():
    plt.close()
    fig=plt.figure(figsize=[11,7])
    fig.tight_layout()
    fileResJP('resDir/resDataP_320000_0-2-15-Soft.mat',threshold=0.5,\
        isSave=False,N0=3,N1=2,NL=[1,3,5],strL='ace')
    fileResJP('resDir/resDataS_320000_0-2-15-Soft.mat',phase='s',\
        threshold=0.5,isSave=False,N0=3,N1=2,NL=[2,4,6],\
        strL='bdf',isLegend=True)
    plt.savefig('fig/fileResAll_soft.eps')

def plotFileResShorter():
    plt.close()
    plt.figure(figsize=[11.5,7])
    fileResJP('resDir/resDataP_320000_0-2-15-Shorter.mat',threshold=0.5,\
        isSave=False,N0=3,N1=2,NL=[1,3,5],strL='ace',\
        iL=np.arange(0,1200,100))
    fileResJP('resDir/resDataS_320000_0-2-15-Shorter.mat',phase='s',\
        threshold=0.5,isSave=False,N0=3,N1=2,NL=[2,4,6],\
        strL='bdf',isLegend=True,iL=np.arange(0,1200,100))
    plt.savefig('fig/fileResAll_shorter.eps')

def plotFileRes1500():
    plt.close()
    plt.figure(figsize=[11.5,7])
    fileResJP('resDir/resDataP_320000_0-2-15-1500.mat',threshold=0.5,\
        isSave=False,N0=3,N1=2,NL=[1,3,5],strL='ace',\
        iL=np.arange(0,1500,100))
    fileResJP('resDir/resDataS_320000_0-2-15-1500.mat',phase='s',\
        threshold=0.5,isSave=False,N0=3,N1=2,NL=[2,4,6],\
        strL='bdf',isLegend=True,iL=np.arange(0,1500,100))
    plt.savefig('fig/fileResAll_1500.eps')

def DP(index0L,indexL,D,DL,phase='p',threshold=0.25,\
    delta=0.02,i0=250,i1=1750,ax1=None,isLegend=False,\
    isPrint=False,is30=False):
    dIndex=int(threshold/delta)
    vL=np.where((index0L>i0)*(index0L<i1))[0]
    index0L=index0L[vL]
    indexL=indexL[vL]
    dIndexL=np.abs(index0L-indexL)
    dIndexLO=(index0L-indexL)
    m=dIndexLO[dIndexL<(2/delta)].mean()*delta
    std=dIndexLO[dIndexL<(threshold/delta)].std()*delta
    dIndexL=np.abs(dIndexLO-m*delta)
    D=D[vL]
    DNew=D[dIndexL<dIndex]
    #ax1 = plt.subplot()
    plt.hist([D,DNew],DL,color=[[0.01,0.01,0.01],[0.5,0.5,0.5]])
    num=np.histogram(D,DL)[0]
    numNew=np.histogram(DNew,DL)[0]
    if is30:
        print('dTime>20:',(DNew>20).sum()/(D>20).sum())
    TP=len(DNew)
    r=TP/len(D)
    p=TP/(((indexL>0)*(indexL<100000)).sum())
    F1=2/(1/r+1/p)
    #m=dIndexLO[dIndexL<(2/delta)].mean()*delta
    #std=dIndexLO[dIndexL<(2/delta)].std()*delta
    print('p:%f r: %f F1: %f m: %f std： %f'%(p,r,F1,m,std))
    DL=(DL[:-1]+DL[1:])/2
    DL=DL[num>0]
    numNew=numNew[num>0]
    num=num[num>0]
    if isLegend:
        plt.legend(['total','detected'],bbox_to_anchor=[1.25,0.25])
    #return ax1,ax2
    ax2=ax1.twinx()
    ax2.plot(DL,numNew/num,'-ok',linewidth=0.8,markersize=4)
    rate=numNew/num
    print(rate)
    #plt.legend(['rate'])
    return ax1,ax2
def showXY(x,y,channelL):
    plt.plot(x[:,:,2]+1)
    for i in channelL:
        plt.plot(y[:,:,i]-i)
def cmpTwo(file1='resDir/resDataP_320000_0-2-15.mat',\
    file2='resDir/resDataP_320000_0-2-15-Soft.mat',\
    phase='p',threshold=0.5,delta=0.02):
    index0L1,indexL1,x01,y01=fileResJP(file1,phase=phase,isPlot=False)
    index0L2,indexL2,x02,y02=fileResJP(file2,phase=phase,isPlot=False)
    dIndex=threshold/delta
    dIndexL1=np.abs(index0L1-indexL1)
    dIndexL2=np.abs(index0L2-indexL2)
    timeL=np.arange(x02.shape[1])*delta
    iL=[156,1235]
    plt.close()
    #print(i)
    plt.figure(figsize=[6,6])
    strL='bacd'
    for ii in range(2):
        i=iL[ii]
        if index0L1[i]>50*10:
            if (dIndexL1[i]>dIndex and dIndexL2[i]<dIndex ) or \
            (dIndexL1[i]<dIndex and dIndexL2[i]>dIndex ):
                plotWaveformCmp(x01[i],y01[i],y02[i],delta=0.02,figName='fig/cmp/%d_cmp.eps'%i,phase='p',text='(%s)'%strL[ii])
                if False:
                    ax1=plt.subplot(2,2,ii+1)
                    plt.plot(timeL,x01[i,:,0,2],'k',linewidth=0.3)
                    plt.xlabel('t/s')
                    plt.ylabel('Amplitude')
                    plt.text(0.05,0.92,'(%s)'%strL[ii],transform=ax1.transAxes)
                    ax2=plt.subplot(2,2,ii+3)
                    h1,=plt.plot(timeL,y01[i,:,0,0],'--k',linewidth=0.8)
                    h2,=plt.plot(timeL,y02[i,:,0,0],'b',linewidth=0.8)
                    h3,=plt.plot(timeL,y02[i,:,0,1],'r',linewidth=0.8)
                    plt.text(0.05,0.92,'(%s)'%strL[ii+2],transform=ax2.transAxes)
                    plt.xlabel('t/s')
                    plt.ylabel('Probability')
                    plt.ylim([-0.1,1.1])
                    plt.legend((h1,h2,h3),['P','P\'','S\''])
    #plt.savefig('fig/cmp/cmp.jpg',dpi=300)
    #plt.close()
def cmpTwoS(file1='resDir/resDataS_320000_0-2-15.mat',\
    file2='resDir/resDataS_320000_0-2-15-Soft.mat',\
    phase='s',threshold=0.5,delta=0.02):
    index0L1,indexL1,x01,y01=fileResJP(file1,phase=phase,isPlot=False)
    index0L2,indexL2,x02,y02=fileResJP(file2,phase=phase,isPlot=False)
    dIndex=threshold/delta
    dIndexL1=np.abs(index0L1-indexL1)
    dIndexL2=np.abs(index0L2-indexL2)
    timeL=np.arange(x02.shape[1])*delta
    iL=[5432,6818]#np.arange(10000)
    #print(i)
    plt.figure(figsize=[6,6])
    strL='cb'*10000
    count=0
    for ii in range(2):
        i=iL[ii]
        if index0L1[i]>50*10:
            if (dIndexL1[i]>dIndex and dIndexL2[i]<dIndex ) or \
            (dIndexL1[i]<dIndex and dIndexL2[i]>dIndex ):
                plotWaveformCmp(x01[i],y01[i],y02[i],delta=0.02,\
                    figName='fig/cmp/%d_cmpS.eps'%i,phase='s',text='(%s)'%strL[count])
                count+=1
                if count==10000:
                    continue
def loadPRF1(file='PRF1'):
    trainSetN=8
    testSetN=2
    pN=5
    thresN=3
    phaseN=2
    M=np.loadtxt(file)
    return M.reshape(phaseN,testSetN,thresN,pN,trainSetN)

def plotPRF1(M=loadPRF1()):
    trainSetN=8
    testSetN=2
    pN=5
    thresN=3
    phaseN=2
    pL=[0,1,2,3,4]
    strL='abcdefghijklmn'
    thresIndex=1
    rgbL=['o-k','^--k','s-k','d--k']
    legendL=['P on JP','P on SC','S on JP','S on SC']
    trainSetL=np.arange(trainSetN)+1
    plt.close()
    #plt.figure(figsize=[10.5,5]).tight_layout()
    plt.figure(figsize=[7,10]).tight_layout()
    ylabelL=['P','R','F1','mean(s)','std(s)']
    for i in range(5):
        p=pL[i]
        '''
        if i<3:
            ax=plt.subplot(2,3,i+1)
        else:
            ax=plt.subplot(2,2,i)
        '''
        ii = int(i/2)
        jj = int(i%2)
        ax = plt.subplot(8,4,ii*8+jj*2+1)
        hL=[None for tmp in range(4)]
        for j in range(phaseN):
            for k in range(testSetN):
                styleI=j*testSetN+k
                h,=plt.plot(trainSetL,M[j,k,thresIndex,p],rgbL[styleI],linewidth=0.8,markersize=4)
                hL[styleI]=h
        ##hL=set(hL)
        plt.xlabel('training set')
        plt.ylabel(ylabelL[i])
        plt.text(0.01,0.93,'(%s)'%strL[i+4],transform=ax.transAxes)
       # ax.transAxes
        if i==4:
            #plt.ylim([0.08,0.217])
            plt.legend(hL,legendL,bbox_to_anchor=[1.2,0.25])
    plt.savefig('fig/PRF1V2.eps')

def calSNR(x0,y0,y):
    x0=x0.reshape([-1,x0.shape[1],3])
    y0=y0.reshape([-1,y0.shape[1]])
    y=y.reshape([-1,y.shape[1]])
    SNR=np.zeros(x0.shape[0])+1e-3
    index0L=np.zeros(x0.shape[0])
    indexL=np.zeros(x0.shape[0])
    for i in range(x0.shape[0]):
        if y0[i].max()>0.99:
            index0=y0[i].argmax()
            if y[i].max()>0.5:
                index=y[i].argmax()
            else:
                index=-1000
        else:
            index0=-1000
            index=-1000
        if y0[i,60:-60].max()>0.99:
            A0=x0[i,max(0,index0-500):index0-50].std()
            A1=x0[i,index0:min(2000,index0+200)].std()
            SNR[i]=np.log(A1/(A0+1e-8))
        index0L[i]=index0
        indexL[i]=index
    return SNR,index0L,indexL

def calRes(t0,V0,t,V,minDT=0.5):
    dT=t-t0
    m=dT[np.abs(dT)<2].mean()
    dT=dT-dT[np.abs(dT)<2].mean()
    Tp=((np.abs(dT)<=minDT) * (V>0.5)*(V0>0.99)).sum()
    Fp=((np.abs(dT)>minDT) * (V>=0.5)*(V0>0.99)).sum()
    #Fp+=((np.abs(dT)>minDT) * (V>=0.5)*(V0<0.01)).sum()
    #( (V>0.5)*(V0<0.0001)).sum()
    Fn=((V<0.5) * (V0>0.5)).sum()+((V>=0.5) * (V0>0.5)*(np.abs(dT)>minDT)).sum()
    p=Tp/(Tp+Fp)
    r=Tp/(V0>0.99).sum()
    F1=2*p*r/(p+r)
    dTNew=dT[(np.abs(dT)<=0.5) * (V>0.5)]
    return p,r,F1,dTNew.mean()+m,dTNew.std()

def calResAll():
    minDTL=[1,0.5,0.25]
    mulL=[128,64,32,16,4,1]
    pFileL=['resDataP_80000_1000-2-15']+\
    ['resDataP_%d_0-2-15'%(320000/mulL[i]) for i in range(6)]+\
    ['resDataP_320000_100-2-15']#+['resDataP_320000_100']
    sFileL=['resDataS_80000_1000-2-15']+\
    ['resDataS_%d_0-2-15'%(320000/mulL[i]) for i in range(6)]+\
    ['resDataS_320000_100-2-15']#+['resDataS_320000_100']
    cmdStrL=['b','-.g','g','-.r','r','-.k','k','y']
    strL=['SC']+['1/%d JP'%(mulL[i]) for i in range(6)]+['SC + 1/1 JP']#+['no Filter']
    timeBinL=np.arange(-1.5,1.51,0.1)
    pJPL=[]
    sJPL=[]
    pSCL=[]
    sSCL=[]
    sL='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    resPJP='resPJP-2-15'
    resPSC='resPSC-2-15'
    resSJP='resSJP-2-15'
    resSSC='resSSC-2-15'
    excelName='resPSJPSCV2.xlsx'
    for pFile in pFileL:
        pJP,pSC=fileRes('resDir/'+pFile,'p')
        pJPL.append(pJP)
        pSCL.append(pSC)

    for sFile in sFileL:
        sJP,sSC=fileRes('resDir/'+sFile,'s')
        sJPL.append(sJP)
        sSCL.append(sSC)
    textL='abcd'
    psL=[pJPL,pSCL,sJPL,sSCL]
    fileL=[resPJP,resPSC,resSJP,resSSC]
    plt.close()
    fig=plt.figure(figsize=[7,10])#fig=plt.figure(figsize=[8,4.5])
    fig.tight_layout()
    excel=Workbook()
    for index in range(4):
        ps=psL[index]
        fileName=fileL[index]
        sheet=excel.create_sheet(fileName, index=index)
        ii = int(index/2)
        jj = int(index%2)
        plt.subplot(8,4,ii*8+jj*2+1)#plt.subplot(2,2,index+1)
        plt.xlim([-1.5,1.5])
        plt.ylim([0,0.6])
        plt.text(-1.42,0.57,'(%s)'%textL[index])
        hL=()
        for i in range(len(ps)):
            tmp=ps[i]
            tmpStr=strL[i]
            cmdStr=cmdStrL[i]
            dTime=(tmp[2]-tmp[0])[(tmp[1]>0.5) * (tmp[3]>0.5)]
            tmpFL=np.histogram(dTime,timeBinL,normed=True)[0]*(timeBinL[1]-timeBinL[0])
            h,=plt.plot((timeBinL[:-1]+timeBinL[1:])/2,tmpFL,cmdStr,linewidth=0.5)
            hL+=(h,)
            #f.write('%11s : '%tmpStr)
            row=[]
            for minDT in minDTL:
                p,r,F1,m,s=calRes(tmp[0],tmp[1],tmp[2],tmp[3],minDT)
                #f.write(' %5.3f %5.3f %5.3f %5.3f %5.3f |'%(p,r,F1,m,s))
                row+=[p,r,F1,m,s]
            #f.write('\n')
            sheet.append(row)
        if index==3:
            plt.legend(hL,strL,bbox_to_anchor=[1.3,1])
        plt.xlabel('t(s)')
        plt.ylabel('frequency density')
    excel.save(excelName)
    plt.savefig('fig/trainChangeV2.eps')
    plt.close()

def plotWaveform(x0,y0,y,delta=0.02,figName='test.eps',phase='p',text='(a)'):
    timeL=np.arange(x0.shape[0])*0.02
    x0=processX(x0)
    x0=x0/np.abs(x0).max(axis=(1,2,3),keepdims=True)*0.4
    y0=y0.reshape((-1))
    if phase=='p':
        y0=y0**0.25
    y=y.reshape((-1))
    t0=(y0[:].argmax()+0)*delta
    t=(y[:].argmax()+0)*delta
    plt.figure(figsize=[3,3])
    for comp in range(3):
        plt.plot(timeL,x0[0,:,0,comp]+comp,'k',linewidth=0.5)
    plt.plot(timeL,y0-1.5,'-.k',linewidth=0.5)
    plt.xlim([timeL[0],timeL[-1]])
    plt.yticks(np.arange(-2,3),['q(x)','p(x)','E','N','Z'])
    plt.xlabel('t/s')
    h0,=plt.plot(np.array([t0,t0]),np.array([-1.2,-0.4]),'k',linewidth=0.5)
    #plt.legend((h0,),{'t0'})
    plt.ylim([-1.6,3.2])
    plt.savefig(figName[:-4]+'_0.eps')
    h1,=plt.plot(np.array([t,t]),np.array([-2.2,-1.4]),'k',linewidth=0.5)
    plt.ylim([-2.6,3.2])
    plt.plot(timeL,y-2.5,'-.k',linewidth=0.5)
    #plt.legend((h0,h1),{'t0','t'})
    plt.text(1,2.95,text)
    plt.savefig(figName)
    print(figName)
    plt.close()

def plotWaveformCmp(x0,y1,y2,delta=0.02,figName='test.eps',phase='p',text='(a)'):
    timeL=np.arange(x0.shape[0])*0.02
    x0=processX(x0)
    x0=x0/np.abs(x0).max(axis=(1,2,3),keepdims=True)*0.4
    y1=y1.reshape((-1))
    #if phase=='p':
    #    y0=y0**0.25
    y2=y2.reshape((y2.shape[0],-1))
    t1=(y1[250:1750].argmax()+250)*delta
    t2=(y2[250:1750].argmax()+250)*delta
    plt.figure(figsize=[3.3,3])
    for comp in range(3):
        plt.plot(timeL,x0[0,:,0,comp]+comp,'k',linewidth=0.5)
    hL=[None,None,None]
    hL[0],=plt.plot(timeL,y1-1.5,'-k',linewidth=0.5)
    plt.xlim([timeL[0],timeL[-1]])
    plt.yticks(np.arange(-3,3),['q_s1(x)','q_p1(x)','q_%s0(x)'%phase,'E','N','Z'])
    plt.xlabel('t/s')
    plt.ylim([-3.6,3.2])
    strL=['']
    strL=['-k','-k']
    for i in range(2):
        hL[i+1],=plt.plot(timeL,y2[:,i]-2.5-i,strL[i],linewidth=0.5)
    #plt.legend(set(hL),['P','P\'','S\''])
    plt.text(1,2.85,text)
    plt.savefig(figName)
    plt.close()

def plotTestOutput(fileName='../resDataP_320000_100-2-15',phase='p',outDir='fig/testFig/',N=100):
    data=sio.loadmat(fileName)
    y0JP=data['%sy0'%phase]
    x0JP=data['out%sx'%phase]
    yJP=data['out%sy'%phase]
    y0SC=data['%sy0Test'%phase]
    x0SC=data['out%sxTest'%phase]
    ySC=data['out%syTest'%phase]
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    for i in range(200):
        if phase=='p':
            strTmp='(a)'
        else:
            strTmp='(c)'
        plotWaveform(x0JP[i],y0JP[i],yJP[i],figName='%s/JP%d.eps'%(outDir,i),phase=phase,text=strTmp)
    for i in range(200):
        if phase=='p':
            strTmp='(b)'
        else:
            strTmp='(d)'
        plotWaveform(x0SC[i],y0SC[i],ySC[i],figName='%s/SC%d.eps'%(outDir,i),phase=phase,text=strTmp)

def dayTimeDis(quakeLs,staInfos,mlL0,minCover=0.5,minSta=3,isBox=False):
    #timeL=[]
    depL=[]
    mlL=[]
    numL=[]
    numLS=[]
    timeL=[]
    laL=[]
    loL=[]
    plt.close()
    plt.figure(figsize=[9,3])
    for quakeL in quakeLs:
        for quake in quakeL:
            if quake.time>UTCDateTime(2015,1,1).timestamp:
                continue
            if len(quake)<minSta or quake.calCover(staInfos)<minCover:
                continue
            la=quake.loc[0]
            lo=quake.loc[1]
            if isBox and (la<38.7 or la>42.2 or lo<97.5 or lo>103.8):
                continue
            timeL.append(quake.time)
            mlL.append(quake.ml)
            depL.append(quake.loc[2])
            laL.append(quake.loc[0])
            loL.append(quake.loc[1])
            numL.append(len(quake))
            numLS.append(np.sign(quake.getSTimeL(staInfos)).sum())
    depL=np.array(depL)
    mlL=np.array(mlL)
    numL=np.array(numL)
    numLS=np.array(numLS)
    plt.subplot(1,3,1)
    plt.hist(mlL,np.arange(-1,6,0.2),color='k',log=True)
    #plt.hist(mlL0,np.arange(-1,6,0.2),color='r',log=True,)
    #plt.legend((h2,h1),['catalog','auto pick'])
    plt.xlabel('ml')
    plt.ylabel('count')
    a=plt.ylim()
    b=plt.xlim()
    plt.text(-1.1,a[1]*0.7,'(a)')

    plt.subplot(1,3,2)
    #plt.hist(mlL,np.arange(-1,6,0.2),color='b',log=True)
    plt.hist(mlL0,np.arange(-1,6,0.2),color='k',log=True,)
    #plt.legend((h2,h1),['catalog','auto pick'])
    plt.xlabel('ml')
    plt.ylabel('count')
    a=plt.ylim(a)
    b=plt.xlim(b)
    plt.text(-1.1,a[1]*0.7,'(b)')
    plt.subplot(1,3,3)
    plt.hist(numL,np.arange(0,100,1),color='k',log=True)
    plt.ylabel('count')
    plt.xlabel('n')
    a=plt.ylim(a)
    plt.text(-1.5,a[1]*0.7,'(c)')
    #plt.ylabel('count')
    plt.savefig('fig/ml_n.eps')
    plt.savefig('fig/ml_n.tiff',dpi=600)
    plt.close()
    print(len(numL),numL.sum(),numLS.sum())
    return np.array(timeL),np.array(laL),np.array(loL)

def getCatalog(fileName='../NM/catalog.txt'):
    timeL=[]
    mlL=[]
    laL=[]
    loL=[]
    laL0=[]
    loL0=[]
    mlL0=[]
    with open(fileName,encoding='ISO-8859-1') as f:
        for line in f.readlines():
            la=float(line[24:30])
            lo=float(line[32:39])
            #print(la,lo)
            time=UTCDateTime(line[:22]).timestamp-3600*8
            if time<UTCDateTime(2015,1,1).timestamp and time>=UTCDateTime(2014,1,1).timestamp:
                if la<37.75 or la>40.7 or lo<96.2 or lo>104.2:
                    continue
                laL.append(la)
                loL.append(lo)
                mlL.append(float(line[45:49]))
                if la<38.7 or la>42.2 or lo<97.5 or lo>103.8:
                    continue
                timeL.append(time)
                laL0.append(la)
                loL0.append(lo)
                mlL0.append(float(line[45:49]))
                
    return np.array(timeL),np.array(mlL),np.array(laL),np.array(loL),np.array(laL0),np.array(loL0),np.array(mlL0)

def compareTime(timeL,timeL0,laL,loL,laL0,loL0,mL0,maxDT=10,maxD=1):
    count=0
    laLN=[]
    loLN=[]
    mLN=[]
    timeLN=[]
    for i0 in range(len(timeL0)):
        time0=timeL0[i0]
        sortI=np.abs(timeL-time0).argsort()
        isAny=False
        for i in sortI:
            if np.abs(timeL[i]-time0)>maxDT:
                break
            if np.abs(laL[i]-laL0[i0])<maxD and np.abs(loL[i]-loL0[i0])<maxD:
                count+=1
                isAny=True
                break
        if not isAny:
            laLN.append(laL0[i0])
            loLN.append(loL0[i0])
            mLN.append(mL0[i0])
            timeLN.append(time0)
    return count,np.array(laLN),np.array(loLN),np.array(mLN),np.array(timeLN)

def getDateCon(filename):
    sDate=UTCDateTime(2013,1,1).timestamp
    eDate=UTCDateTime(2017,1,1).timestamp
    dateN=int((eDate-sDate)/86400)
    dateD={}
    with open(filename) as f:
        for line in f.readlines():
            #15615 14.132_BH /media/jiangyr/XIMA_I/XIMA_I/BU/15615/R132.01
            tmp=line.split()
            sta=tmp[0]
            if sta not in dateD:
                dateD[sta]=np.zeros(dateN)
            y=2000+int(tmp[1][:2])
            j=int(tmp[1][3:6])
            date=UTCDateTime(y,1,1).timestamp+(j-1)*86400
            dateI=int((date-sDate)/86400)
            if dateI <0:
                print(line)
                continue
            dateD[sta][dateI]=1
    return dateD

#719529
def getDateConBtz(dirname):
    dateD={}
    sDate=UTCDateTime(2013,1,1).timestamp
    eDate=UTCDateTime(2017,1,1).timestamp
    dateN=int((eDate-sDate)/86400)
    date0=719529
    for file in glob(dirname+'/*.txt'):
        sta=os.path.basename(file)[:-4]
        dateD[sta]=np.zeros(dateN)
        with open(file) as f:
            line0=f.readline()
            num=int(line0.split()[-1])
            for i in range(int(num/2)):
                line0=f.readline()
                line1=f.readline()
                dateI0=int(int(line0)-date0-sDate/86400)
                dateI1=int(int(line1)-date0-sDate/86400)
                #print(dateI0,dateI1)
                dateD[sta][dateI0:dateI1+1]=1
    return dateD

def compDateD():
    dateD=getDateCon('include/fileLst')
    dateDBtz=getDateConBtz('include/date_continuity/')
    cmpD={}
    for sta in dateDBtz.keys():
        if sta not in dateD:
            print('noSta',sta)
            continue
        date=dateD[sta]
        dateBtz=dateDBtz[sta]
        num=date.sum()
        numBtz=dateBtz.sum()
        d=date-dateBtz
        dnum=(d==1).sum()
        dnumBtz=(d==-1).sum()
        if dnumBtz>0:
            print('BtzmoreDate',sta,dnumBtz)
            if dnumBtz==1:
                print(np.where(d==-1))
        if dnum>0:
            #print('moreDate',sta,dnum)
            pass
        cmpD[sta]=[num, numBtz, dnum, dnumBtz]
    return cmpD
