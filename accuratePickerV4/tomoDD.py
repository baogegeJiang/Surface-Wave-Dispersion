import numpy as np
from distaz import DistAz
from obspy import UTCDateTime
from tool import getYmdHMSj
import matplotlib.pyplot as plt
from mathFunc import xcorr
#from cudaFunc import  torchcorrnn as xcorr
SPRatio=0.5
wkdir='TOMODD'
def preEvent(quakeL,staInfos,filename='abc'):
    if filename=='abc':
        filename='%s/input/event.dat'%wkdir
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
            ml=0
            if quake.ml!=None :
                if quake.ml>-2:
                    ml=quake.ml
            Y=getYmdHMSj(UTCDateTime(quake.time))
            f.write("%s  %s%02d   %.4f   %.4f    % 7.3f % 5.2f   0.15    0.51  % 5.2f   % 8d %1d\n"%\
                (Y['Y']+Y['m']+Y['d'],Y['H']+Y['M']+Y['S'],int(quake.time*100)%100,\
                    quake.loc[0],quake.loc[1],quake.loc[2],ml,1,i,0))

def preABS(quakeL,staInfos,filename='abc',isNick=True,notTomo=False):
    if filename=='abc':
        filename='%s/input/ABS.dat'%wkdir
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            quake=quakeL[i]
            ml=0
            if quake.ml!=None :
                if quake.ml>-2:
                    ml=quake.ml
            if notTomo:
                Y=getYmdHMSj(UTCDateTime(quake.time))
                f.write("%s %s %s  %s %s %s.%02d   %.4f   %.4f    % 7.3f % 5.2f   0.15    0.51  % 5.2f   % 8d %1d\n"%\
                    (Y['Y'],Y['m'],Y['d'],Y['H'],Y['M'],Y['S'],int(quake.time*100)%100,\
                    quake.loc[0],quake.loc[1],quake.loc[2],ml,1,i,0))
            else:
                f.write('#  % 8d\n'%i)
            for record in quake:
                staIndex=record.getStaIndex()
                staInfo=staInfos[staIndex]
                if not isNick:
                    staName=staInfo['sta']
                else:
                    staName=staInfo['nickName']
                if record.pTime()!=0:
                    if not isNick:
                        f.write('%8s    %7.2f   %5.3f   P\n'%\
                            (staName,record.pTime()-quake.time,1.0))
                    else:
                        f.write('%s     %7.2f   %5.3f   P\n'%\
                            (staName,record.pTime()-quake.time,1.0))
                if record.sTime()!=0:
                    if not isNick:
                        f.write('%8s    %7.2f   %5.3f   S\n'%\
                            (staName,record.pTime()-quake.time,1.0))
                    else:
                        f.write('%s     %7.2f   %5.3f   S\n'%\
                            (staName,record.sTime()-quake.time,SPRatio))

def preSta(staInfos,filename='abc',isNick=True):
    if filename=='abc':
        filename='%s/input/station.dat'%wkdir
    with open(filename,'w+') as f:
        for staInfo in staInfos:
            if not isNick:
                staName=staInfo['sta']
            else:
                staName=staInfo['nickName']
            f.write('%s %7.4f %8.4f %.0f\n'\
                %(staName,staInfo['la'],staInfo['lo'],staInfo['dep']))

def preDTCC(quakeL,staInfos,dTM,maxD=0.5,minSameSta=5,minPCC=0.75,minSCC=0.75,\
    perCount=500,filename='abc'):
    if filename=='abc':
        filename='%s/input/dt.cc'%wkdir
    N=len(quakeL)
    countL=np.zeros(N)
    with open(filename,'w+') as f:
        for i in range(len(quakeL)):
            print(i)
            pTime0=quakeL[i].getPTimeL(staInfos)
            sTime0=quakeL[i].getSTimeL(staInfos)
            time0=quakeL[i].time
            if countL[i]>perCount:
                continue
            jL=np.arange(i+1,len(quakeL))
            np.random.shuffle(jL) 
            for j in jL:
                if dTM[i][j]==None:
                    continue
                if countL[j]>perCount:
                    continue
                if DistAz(quakeL[j].loc[0],quakeL[j].loc[1],quakeL[j].loc[0],\
                    quakeL[j].loc[1]).getDelta()>maxD:
                    continue
                pTime1=quakeL[j].getPTimeL(staInfos)
                sTime1=quakeL[j].getSTimeL(staInfos)
                time1=quakeL[j].time
                if len(sameSta(pTime0,pTime1))<minSameSta:
                    continue                  
                for dtD in dTM[i][j]:
                    dt,maxC,staIndex,phaseType=dtD
                    if phaseType==1 and maxC>minPCC:
                        dt=pTime0[staIndex]-time0-(pTime1[staIndex]-time1+dt)
                        f.write("% 9d % 9d %s %8.3f %6.4f %s\n"%(i,j,\
                            staInfos[staIndex]['nickName'],dt,maxC*maxC,'P'))
                        countL[i]+=1
                        countL[j]+=1
                    if phaseType==2 and maxC>minSCC:
                        dt=sTime0[staIndex]-time0-(sTime1[staIndex]-time1+dt)
                        f.write("% 9d % 9d %s %8.3f %6.4f %s\n"%(i,j,\
                            staInfos[staIndex]['nickName'],dt,maxC*maxC*SPRatio,'S'))
                        countL[i]+=1
                        countL[j]+=1


def preMod(R,nx=8,ny=8,nz=12,filename='abc'):
    if filename=='abc':
        filename='%s/MOD'%wkdir
    with open(filename,'w+') as f:
        vp=np.array([4   ,5.0,5.5,5.8, 5.91, 6.1,  6.3, 6.50, 6.7,  8.0, 8.6, 9.0])
        #vs=[2.4,2.67, 3.01,  4.10, 4.24, 4.50, 5.00, 5.15, 6.00,6.1]
        z =[-150, -2,  0,  5,   10,  15,   25,   35,  50,   60,  80, 500]
        vs=vp/1.71
        x=np.zeros(nx)
        y=np.zeros(ny)
        #z=[-150,-2, 0, 5,10, 15,25, 35, 50, 60,80, 500]
        f.write('0.1 %d %d %d\n'%(nx,ny,nz))
        x[0]=R[2]-5
        x[-1]=R[3]+5
        y[0]=R[0]-5
        y[-1]=R[1]+5
        x[1]=(x[0]+R[2])/2
        x[-2]=(x[-1]+R[3])/2
        y[1]=(y[0]+R[0])/2
        y[-2]=(y[-1]+R[1])/2
        x[2:-2]=np.arange(R[2],R[3]+0.001,(R[3]-R[2])/(nx-5))
        y[2:-2]=np.arange(R[0],R[1]+0.001,(R[1]-R[0])/(ny-5))
        #f.write("\n")
        for i in range(nx):
            f.write('%.4f '%x[i])
        f.write('\n')
        for i in range(ny):
            f.write('%.4f '%y[i])
        f.write('\n')
        for i in range(nz):
            f.write('%.4f '%z[i])
        f.write('\n')

        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    f.write('%.2f '%vp[i])
                f.write('\n')

        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    f.write('%.2f '%(vp[i]/vs[i]))
                f.write('\n')

def sameSta(timeL1,timeL2):
    return np.where(np.sign(timeL1*timeL2)>0)[0]

def calDT(quake0,quake1,waveform0,waveform1,staInfos,bSec0=-2,eSec0=3,\
    bSec1=-3,eSec1=4,delta=0.02,minC=0.6,maxD=0.3,minSameSta=5):
    '''
    dT=[[dt,maxCC,staIndex,phaseType],....]
    dT is a list containing the dt times between quake0 and quake1
    dt is the travel time difference
    maxCC is the peak value of the normalized cross-correlation 
    staIndex is the index of the station
    phaseType: 1 for P; 2 for S
    '''
    pTime0=quake0.getPTimeL(staInfos)
    sTime0=quake0.getSTimeL(staInfos)
    pTime1=quake1.getPTimeL(staInfos)
    sTime1=quake1.getSTimeL(staInfos)
    sameIndex=sameSta(pTime0,pTime1)
    if len(sameIndex)<minSameSta:
        return None
    if DistAz(quake0.loc[0],quake0.loc[1],quake1.loc[0],quake1.loc[1]).getDelta()>maxD:
        return None
    dT=[];
    timeL0=np.arange(bSec0,eSec0,delta)
    indexL0=(timeL0/delta).astype(np.int64)-waveform0['indexL'][0][0]
    timeL1=np.arange(bSec1,eSec1,delta)
    indexL1=(timeL1/delta).astype(np.int64)-waveform1['indexL'][0][0]
    for staIndex in sameIndex:
        if pTime0[staIndex]!=0 and pTime1[staIndex]!=0:
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            pWave0=waveform0['pWaveform'][index0,indexL0,2]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            #print(index1)
            pWave1=waveform1['pWaveform'][index1,indexL1,2]
            c=xcorr(pWave1,pWave0)#########
            maxC=c.max()
            if maxC>minC:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC,staIndex,1])
        if sTime0[staIndex]!=0 and sTime1[staIndex]!=0:
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            sWave0=waveform0['sWaveform'][index0,indexL0,0]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            sWave1=waveform1['sWaveform'][index1,indexL1,0]
            c=xcorr(sWave1,sWave0)##########
            maxC0=c.max()
            if maxC0>minC:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC0,staIndex,2])
            index0=np.where(waveform0['staIndexL'][0]==staIndex)[0]
            sWave0=waveform0['sWaveform'][index0,indexL0,1]
            index1=np.where(waveform1['staIndexL'][0]==staIndex)[0]
            sWave1=waveform1['sWaveform'][index1,indexL1,1]
            c=xcorr(sWave1,sWave0)##########
            maxC1=c.max()
            if maxC1>minC and maxC1>maxC0:
                maxIndex=c.argmax()
                dt=timeL1[maxIndex]-timeL0[0]
                dT.append([dt,maxC1,staIndex,2])
    return dT

def calDTM(quakeL,waveformL,staInfos,maxD=0.3,minC=0.6,minSameSta=5,\
    isFloat=False,bSec0=-2,eSec0=3,bSec1=-3,eSec1=4):
    '''
    dTM is 2-D list contianing the dT infos between each two quakes
    dTM[i][j] : dT in between quakeL[i] and quakeL[j]
    quakeL's waveform is contained by waveformL
    '''
    if isFloat:
        for waveform in waveformL:
            waveform['pWaveform']=waveform['pWaveform'].astype(np.float32)
            waveform['sWaveform']=waveform['sWaveform'].astype(np.float32)
    dTM=[[None for quake in quakeL]for quake in quakeL]
    for i in range(len(quakeL)):
        print(i)
        for j in range(i+1,len(quakeL)):
            dTM[i][j]=calDT(quakeL[i],quakeL[j],waveformL[i],waveformL[j],\
                staInfos,maxD=maxD,minC=minC,minSameSta=minSameSta,\
                bSec0=bSec0,eSec0=eSec0,bSec1=bSec1,eSec1=eSec1)
    return dTM

def plotDT(waveformL,dTM,i,j,staInfos,bSec0=-2,eSec0=3,\
    bSec1=-3,eSec1=4,delta=0.02,minSameSta=5):
    plt.close()
    waveform0=waveformL[i]
    waveform1=waveformL[j]
    timeL0=np.arange(bSec0,eSec0,delta)
    indexL0=(timeL0/delta).astype(np.int64)-waveform0['indexL'][0][0]
    timeL1=np.arange(bSec1,eSec1,delta)
    indexL1=(timeL1/delta).astype(np.int64)-waveform1['indexL'][0][0]
    count=0
    staIndexL0=waveform0['staIndexL'][0].astype(np.int64)
    staIndexL1=waveform1['staIndexL'][0].astype(np.int64)
    for dT in dTM[i][j]:
        staIndex=dT[2]
        tmpIndex0=np.where(staIndexL0==staIndex)[0][0]
        tmpIndex1=np.where(staIndexL1==staIndex)[0][0]
        print(tmpIndex0,tmpIndex1)
        if dT[3]==1:
            w0=waveform0['pWaveform'][int(tmpIndex0),indexL0,2]
            w1=waveform1['pWaveform'][int(tmpIndex1),indexL1,2]
        else:
            continue
            w0=waveform0['sWaveform'][int(tmpIndex0),indexL0,0]
            w1=waveform1['sWaveform'][int(tmpIndex1),indexL1,0]
        plt.plot(timeL0+dT[0],w0/(w0.max())*0.5+count,'r')
        print(xcorr(w1,w0).max())
        plt.plot(timeL1,w1/(w1.max())*0.5+count,'b')
        plt.plot(timeL0-dT[0],w0/(w0.max())*0.5+count+2,'r')
        #print(w0.max())
        plt.plot(timeL1,w1/(w1.max())*0.5+count+2,'b')
        #plt.plot(+count,'g')
        #print((w1/w1.max()).shape)
        plt.text(timeL1[0],count+0.5,'cc=%.2f dt=%.2f '%(dT[1],dT[0]))
        count+=1
        plt.savefig('fig/TOMODD/%d_%d_%d_dT.png'%(i,j,staIndex),dpi=300)
        plt.close()

def saveDTM(dTM,filename):
    N=len(dTM)
    with open(filename,'w+') as f:
        f.write("# %d\n"%N)
        for i in range(N):
            for j in range(N):
                if dTM[i][j]==None:
                    continue
                f.write("i %d %d\n"%(i,j))
                for dt in dTM[i][j]:
                    f.write("%f %f %d %d\n"%(dt[0],dt[1],dt[2],dt[3]))
def loadDTM(filename='dTM'):
    with open(filename) as f:
        for line in f.readlines():
            if line.split()[0]=='#':
                N=int(line.split()[1])
                dTM=[[None for i in range(N)]for i in range(N)]
                continue
            if line.split()[0]=='i':
                i=int(line.split()[1])
                j=int(line.split()[2])
                dTM[i][j]=[]
                continue
            staIndex=int(line.split()[2])
            dTM[i][j].append([float(line.split()[0]),float(line.split()[1]),\
            int(line.split()[2]),int(line.split()[3])])
    return dTM

def reportDTM(dTM):
    plt.close()
    N=len(dTM)
    sumN=np.zeros(N)
    quakeN=np.zeros(N)
    dTL=[]
    for i in range(N):
        for j in range(N):
            if dTM[i][j]!=None:
                if len(dTM)<=0:
                    continue
                sumN[i]+=len(dTM[i][j])
                sumN[j]+=len(dTM[i][j])
                quakeN[i]+=1
                quakeN[j]+=1
                for dT in dTM[i][j]:
                    dTL.append(dT[0])
    plt.subplot(3,1,1)
    plt.plot(sumN)
    plt.subplot(3,1,2)
    plt.plot(quakeN)
    plt.subplot(3,1,3)
    plt.hist(np.array(dTL))
    plt.savefig('fig/TOMODD/dTM_report.png',dpi=300)
    plt.close()

def getReloc(quakeL,filename='abc'):
    if filename=='abc':
        filename='%s/tomoDD.reloc'%wkdir
    quakeRelocL=[]
    with open(filename) as f:
        for line in f.readlines():
            line=line.split()
            time=quakeL[0].tomoTime(line)
            index=int(line[0])
            print(quakeL[index].time-time)
            quakeRelocL.append(quakeL[index])
            quakeRelocL[-1].getReloc(line)
    return quakeRelocL

