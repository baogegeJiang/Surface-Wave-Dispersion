import numpy as np
from numba import jit,float32, int64
import scipy.signal  as signal
from scipy import fftpack
from scipy.optimize import curve_fit
from scipy import stats
nptype=np.float32
rad2deg=1/np.pi*180
@jit
def xcorr(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1)
    tb=0
    for i in range(lb):
        tb+=b[i]*b[i]
    for i in range(la-lb+1):
        ta=0
        tc=0
        ta= (a[i:(i+lb)]*a[i:(i+lb)]).sum()
        tc= (a[i:(i+lb)]*b[0:(0+lb)]).sum()
        if ta!=0 and tb!=0:
            c[i]=tc/np.sqrt(ta*tb)
    return c

@jit
def xcorrSimple(a,b):
    
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1)
    for i in range(la-lb+1):
        tc= (a[i:(i+lb)]*b[0:(0+lb)]).sum()
        c[i]=tc
    return c

def xcorrAndDe(a,b):
    la = a.size
    lb = b.size
    lab = min(la,lb)
    if la!=lab or lb != lab:
        a = a[:lab]
        b = b[:lab]
    A = np.fft.fft(a) 
    B = np.fft.fft(b)
    c0 =  signal.correlate(a,b,'full')[lab-1:]
    absB = np.abs(B)
    threshold = absB.max()*(1e-13)
    #c0 = np.real(xcorrFrom0(a,b))
    #np.real(np.fft.ifft(np.conj(B)*A))
    C1 = A.copy()
    C1[absB>threshold]/=C1[absB>threshold]/B[absB>threshold]
    c1 = np.real(np.fft.ifft(C1))
    c0 /= c0.max()
    c1 /= c1.max()
    return c0+c1*1j

def xcorrAndDeV2(a,b):
    la = a.size
    lb = b.size
    lab = min(la,lb)
    if la!=lab or lb != lab:
        a = a[:lab]
        b = b[:lab]
    A = np.fft.fft(a) 
    B = np.fft.fft(b)
    labMid = int(lab/2)
    #B[labMid:]= np.conj(B[labMid:])
    c0 =  signal.correlate(a,b,'full')[lab-1:]
    absB = np.abs(B)
    threshold = absB.max()*(1e-13)
    #c0 = np.real(xcorrFrom0(a,b))
    #np.real(np.fft.ifft(np.conj(B)*A))
    C1 = A.copy()
    C1[absB>threshold]/=B[absB>threshold]
    C1[absB<=threshold] = 0
    c1 = np.real(np.fft.ifft(C1))
    c0 /= c0.max()
    c1 /= c1.max()
    return c0+c1*1j
def xcorrAndDeV3(a,b):
    la = a.size
    lb = b.size
    lab = min(la,lb)
    if la!=lab or lb != lab:
        a = a[:lab]
        b = b[:lab]
    #a = fftpack.hilbert(a)*1j+a
    #b = fftpack.hilbert(b)*1j+b
    A = np.fft.fft(a) 
    B = np.fft.fft(b)
    absA = np.abs(A)
    absB = np.abs(B)
    thresholdA = absA.std()*(1e-1)
    thresholdB = absB.std()*(1e-1)
    C1=np.conj(B)*A
    #c0 = np.real(xcorrFrom0(a,b))
    #np.real(np.fft.ifft(np.conj(B)*A))
    #C1 = A.copy()
    #C1/=B
    C1[absA<=thresholdA] = 0
    C1[absB<=thresholdB] = 0
    return np.fft.ifft(C1)

def xcorrFrom0(a,b):
    la = a.size
    lb = b.size
    x =  signal.correlate(a,b,'full')
    return x[lb-1:]

def xcorrAndConv(a,b):
    la = a.size
    lb = b.size
    x0 =  signal.correlate(a,b,'full')
    x1 =  signal.convolve(b,a,'full')
    return x0[lb-1:]+1j*x1[lb-1:]


@jit
def xcorrComplex(a,b):
    a = fftpack.hilbert(a)*1j+a
    b = fftpack.hilbert(b)*1j+b
    la=a.size
    lb=b.size
    c=np.zeros(la-lb+1).astype(np.complex)
    for i in range(la-lb+1):
        tc= (a[i:(i+lb)]*b[0:(0+lb)].conj()).sum()
        c[i]=tc
    return c

@jit
def xcorrEqual(a,b):
    la=a.size
    lb=b.size
    c=np.zeros(la)
    tb0=(b*b).sum()
    for i in range(la):
        i1=min(i+lb,la)
        ii1=i1-i
        #print(ii1)
        tc= (a[i:i1]*b[0:ii1]).sum()
        tb=tb0
        if ii1!=lb:
            tb=(b[0:ii1]*b[0:ii1]).sum()
        c[i]=tc/np.sqrt(tb)
    return signal.correlate()

def corrNP(a,b):
    a=a.astype(nptype)
    b=b.astype(nptype)
    if len(b)==0:
        return a*0+1
    c=signal.correlate(a,b,'valid')
    tb=(b**2).sum()**0.5
    taL=(a**2).cumsum()
    ta0=taL[len(b)-1]**0.5
    taL=(taL[len(b):]-taL[:-len(b)])**0.5
    c[1:]=c[1:]/tb/taL
    c[0]=c[0]/tb/ta0
    return c,c.mean(),c.std()

@jit
def getDetec(x, minValue=0.2, minDelta=200):
    indexL = [-10000]
    vL = [-1]
    for i in range(len(x)):
        if x[i] <= minValue:
            continue
        if i > indexL[-1]+minDelta:
            vL.append(x[i])
            indexL.append(i)
            continue
        if x[i] > vL[-1]:
            vL[-1] = x[i]
            indexL[-1] = i
    if vL[0] == -1:
        indexL = indexL[1:]
        vL = vL[1:]
    return np.array(indexL), np.array(vL)

def matTime2UTC(matTime,time0=719529):
    return (matTime-time0)*86400

@jit(int64(float32[:],float32,int64,int64,float32[:]))
def cmax(a,tmin,winL,laout,aM):
    i=0 
    while i<laout:
        if a[i]>tmin:
            j=0
            while j<min(winL,i):
                if a[i]>a[i-j]:
                    a[i-j]=a[i]
                j+=1
        if i>=winL:
            aM[i-winL]+=a[i-winL]
        i+=1
    while i<laout+winL:
        aM[i-winL]+=a[i-winL]
        i+=1
    return 1

def cmax_bak(a,tmin,winL,laout,aM):
    i=0 
    indexL=np.where(a>tmin)[0]
    for i in indexL:
        a[max(i-winL,0):i]=np.fmax(a[max(i-winL,0):i],a[i])
    aM[:laout]+=a[:laout]

def CEPS(x):
    #sx=fft(x);%abs(fft(x)).^2;
    #logs=log(sx);
    #y=abs(fft(logs(1:end)));
    spec=np.fft.fft(x)
    logspec=np.log(spec*np.conj(spec))
    y=abs(np.fft.ifft(logspec))
    return y

def flat(z,vp,vs,rho,m=-2,R=6371):
    z = np.array(z)
    zmid = z.mean()
    miu  = vs**2*rho
    lamb = vp**2*rho-2*miu
    r = R-zmid
    zNew = R*np.log(R/(R-z))
    lambNew =  ((r/R)**(m-1))*lamb
    miuNew  =  ((r/R)**(m-1))*miu
    rhoNew  =  ((r/R)**(m+1))*rho
    vpNew   =  ((lambNew+2*miuNew)/rhoNew)**0.5
    vsNew   =  (miuNew/rhoNew)**0.5
    return zNew,vpNew,vsNew,rhoNew

@jit
def validL(v,prob, minProb = 0.7,minV=2,maxV=6):
    l    = []
    tmp  = []
    for i in range(len(v)):
        if v[i] > minV and v[i]<maxV and\
         prob[i]>minProb and (i==0 or np.abs(prob[i]-prob[i-1])/prob[i]<0.2):
            tmp.append(i)
            if i == len(v)-1:
                l.append(tmp)
            continue
        elif len(tmp)>0:
            l.append(tmp)
            tmp=[]
    return l


def randomSource(i,duraCount,data):
    if i==0:
        data[:duraCount] += 1
        data[:duraCount] += np.random.rand()*0.3*np.random.rand(duraCount)
    if i ==1:
        mid = int(duraCount/2)
        data[:mid] = np.arange(mid)
        data[mid:2*mid] = np.arange(mid-1,-1,-1)
        data[:duraCount] += np.random.rand()*0.3*np.random.rand(duraCount)*mid
    if i==2:                
        rise = 0.1+0.3*np.random.rand()
        mid = int(duraCount/2)
        i0 = int(duraCount*rise)
        data[:duraCount] += i0
        data[:i0] = np.arange(i0)
        data[duraCount-i0:duraCount] = np.arange(i0-1,-1,-1)
        data[:duraCount] += np.random.rand()*0.3*np.random.rand(duraCount)*i0
    if i ==3:
        T  = np.random.rand()*60+5
        T0 = np.random.rand()*2*np.pi
        data[:duraCount] = np.sin(np.arange(duraCount)/T*2*np.pi+T0)+1
        data[:duraCount] += (np.random.rand(duraCount)-0.5)*0.1
        data[:duraCount] *= np.random.rand(duraCount)+4
    if i == 4:
        T  = (np.random.rand()**3)*100+5
        T0 = np.random.rand()*2*np.pi
        data[:duraCount] = np.sin(np.arange(duraCount)/T*2*np.pi+T0)
        data[:duraCount] += (np.random.rand(duraCount)-0.5)*0.1
        data[:duraCount] *= np.random.rand(duraCount)+2
@jit
def gaussian(x,A, t0, sigma):
    return A*np.exp(-(x - t0)**2 / sigma**2)
@jit
def fitexp(y):
    N = len(y)
    x = np.arange(N)
    ATS,pcov = curve_fit(gaussian,x,y,p0=[1,N/2,1.5],\
        bounds=(0.1, [3, N, 8]),maxfev=40)
    A = ATS[0]
    t0 = ATS[1]
    sigma = ATS[2]
    #print(pcov)
    return t0

def findPos(y, moreN = 10):
    yPos  = y.argmax( axis=1).astype(np.float32)
    yMax = y.max(axis=1)
    for i in range(y.shape[0]):
        for j in range(y.shape[-1]):
            pos0 = int(yPos[i,0,j])
            max0 = yMax[i,0,j]
            if max0 > 0.5 and pos0>=moreN and pos0+moreN<y.shape[1] :
                try:
                    pos =  fitexp(y[i,pos0-moreN:pos0+moreN,0,j])+pos0-moreN
                except:
                    pass
                else:
                    if np.abs(pos-pos0)<0.5:
                        yPos[i,0,j]=pos
    return yPos, yMax

def disDegree(dis,maxD = 100, maxTheta=20):
    delta = dis/110.7
    theta0 = maxD/110.7
    theta = theta0/np.sin(delta/180*np.pi)
    return min(theta,maxTheta)

def disDegreeBak(dis,maxD = 100, maxTheta=20):
    delta = dis/110.7
    if delta >90:
        delta = 90
    theta0 = maxD/110.7
    theta = theta0/np.sin(delta/180*np.pi)
    return min(theta,maxTheta)

def QC_bak(data,threshold=2.5):
    if len(data)<6:
        return data.mean(),999,len(data)
    #if len(data)<10:
    #    return data.mean(),data.std(),len(data)
    mData = np.median(data)
    d = np.abs(data - mData)
    lqr = stats.iqr(data)
    Threshold = lqr*threshold
    if (d>Threshold).sum()==0:
        return data.mean(),data.std(),len(data)
    else:
        return QC(data[d<Threshold],threshold)

def QC(data,threshold=2.5,it=20):
    if it==0:
        print('***********************************reach depest*******************')
    if len(data)<6 or it==0:
        return data.mean(),999,len(data)
    #if len(data)<10:
    #    return data.mean(),data.std(),len(data)
    mData = np.median(data)
    d = np.abs(data - mData)
    lqr = stats.iqr(data)
    Threshold = lqr*threshold
    mData = np.mean(data[d<Threshold])
    d = np.abs(data - mData)
    if (d>Threshold).sum() ==0 :
        return data[d<Threshold].mean(),data[d<Threshold].std(),len(data)
    else:
        return QC(data[d<Threshold],threshold,it-1)