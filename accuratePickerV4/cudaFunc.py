import torch
import numpy as np
'''
this module is designed to calculate correlation and weak-max by torch
'''
#5.96×10^-8~6.55×10^4
tentype=torch.cuda.HalfTensor
dtype=torch.float16
nptype=np.float16
nptypeO=np.float32
minF=0.0##1e-7
maxF=5e4
minFT=torch.tensor(minF)
maxFT=torch.tensor(maxF)
convert=2e-3
isHalf=False
if not isHalf:
    tentype=torch.cuda.FloatTensor
    dtype=torch.float32
    nptype=np.float32
    nptypeO=np.float32
    minF=0.0##1e-7
    maxF=5e8
    minFT=torch.tensor(minF)
    maxFT=torch.tensor(maxF)
    convert=1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(tentype)

def calAS(a,lb):
    la=a.size
    aR=torch.tensor(a)
    aC=torch.cumsum(aR**2,0)
    aC[:la-lb+1]=1/(aC[lb-1:]-aC[:la-lb+1]+1e-9)
    return aC.cpu().numpy()**(0.5)

def calCC(aT,bT):
    N=aT.shape[0]
    cT=torch.zeros((N,2))
    aT=torch.fft(aT,1)
    bT=torch.fft(bT,1)
    cT[:,0]=aT[:,0]*bT[:,0]+aT[:,1]*bT[:,1]
    cT[:,1]=aT[:,0]*bT[:,1]-aT[:,1]*bT[:,0]
    return torch.ifft(cT,1)[:,0]
    

def torchcorr(a,b,mul=10):
    '''
    use fft to calculate cross-correlation
    when to do on a long data, the precision is worth attention
    '''
    la=a.shape[0]
    lb=b.shape[0]
    if not a.dtype==nptype:
        a=a.astype(nptype)
    if not b.dtype==nptype:
        b=b.astype(nptype)
    aR=torch.tensor(a)
    aR=aR/torch.norm(aR)
    bR=torch.tensor(b)
    tb=torch.norm(bR)
    if tb !=0:
        bR=bR*(1/tb)
    aT=torch.zeros((la,2))
    bT=torch.zeros((la,2))
    aT[:,0]=aR
    bT[:lb,0]=bR
    c=calCC(bT,aT)
    aT=torch.zeros((la,2))
    bT=torch.zeros((la,2))
    ac=torch.zeros(la)
    aT[:,0]=aR.pow(2)
    bT[:lb,0]=1

    ac=calCC(bT,aT)
    ac=torch.rsqrt(torch.abs(ac)+1e-9)
    c=ac*c
    return c[:la-lb+1].cpu().numpy(),c[la-lb+1].mean().cpu().numpy(),\
    c[:la-lb+1].std().cpu().numpy()


def torchcorrnn(a,b):
    '''
    directly calculate cross-correlation based on torch.nn.functional.conv1d
    '''
    la=a.shape[0]
    lb=b.shape[0]
    if not isinstance(a,torch.Tensor):
        aR=torch.tensor(a,dtype=dtype)
    else:
        aR=a

    if not isinstance(b,torch.Tensor):
        bR=torch.tensor(b,device=aR.device,dtype=dtype)
        if isHalf:
            bR=bR/bR.abs().max()
    else:
        bR=b

    tb=torch.norm(bR)
    if tb !=0:
        bR=bR*(1/tb)
    aT=aR.reshape(1,1,-1)
    bT=bR.reshape(1,1,-1)
    c=torch.nn.functional.conv1d(aT,bT)
    aT=aT.pow(2)
    bT[0,0,:]=1
    ac=torch.nn.functional.conv1d(aT,bT)
    ac=torch.rsqrt(ac+minF)
    c*=ac
    c=torch.where(torch.isnan(c),torch.tensor(-1000.0,dtype=dtype),c)
    c=torch.where(torch.isinf(c),torch.tensor(-1000.0,dtype=dtype),c)
    return c[0,0,:],c[0,0,c[0,0,:]>-2].mean().cpu().numpy(),\
            c[0,0,c[0,0,:]>-2].std().cpu().numpy()

def torchMax(a,tmin,winL,aM):
    if not isinstance(a,torch.Tensor):
        aR=torch.tensor(a,dtype=dtype)
    else:
        aR=a
    la=aR.shape[0]
    aMax=torch.nn.functional.max_pool1d(aR.reshape(1,1,-1),winL,1)[0,0,:]
    aR[:la-winL+1]=torch.where(aMax>tmin,aMax,aR[:la-winL+1])
    if isinstance(aM,torch.Tensor):
        if aM.device==aR.device:
            aM[:la]+=aR[:la]
        else:
            aM[:la]+=aR[:la].cuda(device=aM.device)
    else:
        aM[:la]+=aR[:la].cpu().numpy()
