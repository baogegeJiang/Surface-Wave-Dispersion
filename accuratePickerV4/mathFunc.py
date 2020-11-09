import numpy as np
from numba import jit,float32, int64
import scipy.signal  as signal
from scipy.optimize import curve_fit
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
    return c

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

def UTC2MatTime(time,time0=719529):
    return time/86400+time0

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
    return i

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
def cart2pol(x,y):
    r=x+y*1j
    return np.abs(r),np.angle(r)
def pol2cart(R,theta):
    r=R*np.exp(theta*1j)
    return np.real(r),np.imag(r)
def angleBetween(x0,y0,x1,y1):
    X=x0*x1+y0*y1
    X=X/((x0**2+y0**2)*(x1**2+y1**2))**0.5
    Y=x0*y1-y0*x1
    Y=Y/((x0**2+y0**2)*(x1**2+y1**2))**0.5
    r,theta=cart2pol(X,Y)
    #print(X[np.where(X<0)[0]])
    return theta
    

class  R:
    """docstring for  R"""
    def __init__(self, pL):
        super( R, self).__init__()
        if pL.shape[0]==2:
            x0=pL[0,0]
            y0=pL[0,1]
            x1=pL[1,0]
            y1=pL[1,1]
            xL=np.array([x0,x1,x1,x0])
            yL=np.array([y0,y0,y1,y1])
        elif pL.shape[0]==3:
            x0=pL[0,0]
            y0=pL[0,1]
            x1=pL[1,0]
            y1=pL[1,1]
            x2=pL[2,0]
            y2=pL[2,1]
            x3=x2+x0-x1
            y3=y2+y0-y1
            xL=np.array([x0,x1,x2,x3]).reshape(-1,1)
            yL=np.array([y0,y1,y2,y3]).reshape(-1,1)
        self.xyL=np.concatenate([xL,yL],axis=1)
    def isIn(self,p):
        x0=p[0]
        y0=p[1]
        dxL=self.xyL[:,0]-x0
        dyL=self.xyL[:,1]-y0
        dxL=np.append(dxL,dxL[0])
        dyL=np.append(dyL,dyL[0])
        thetaL=angleBetween(dxL[:-1],dyL[:-1],\
            dxL[1:],dyL[1:])
        #print(dxL,dyL,thetaL/np.pi*180,np.abs(thetaL.sum()/np.pi-2))
        if np.abs(thetaL.sum()/np.pi-2)<0.2:
            return True
        else:
            return False

def prob2color(prob,color0=np.array([1,1,1])*0.8):
    # blue for no prob; gray for differetn p(>0.5);red for p(<0.5)
    # green for not detect phase
    if prob > 1:
        return np.array([0,1,0])
    elif prob >0.5:
        pN = (1-prob)*2
        return color0*pN
    elif prob > -1:
        return np.array([1,0,0])
    else:
        return np.array([0,0,1])

def gaussian(x,A, t0, sigma):
    return A*np.exp(-(x - t0)**2 / sigma**2)
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