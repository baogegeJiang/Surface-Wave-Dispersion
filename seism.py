import obspy 
import numpy as np
def tolist(s,d='/'):
    return s.split(d)
nickStrL='1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
strType={'S':str,'f':float,'i':int, 'l':tolist}
NoneType = type(None)


class Dist:
    def __init__(self,*argv,**kwargs):
        self.defaultSet()
        self.l = [None for i in range(len(self.keys))]
        if 'keysIn' in kwargs :
            self.keysIn     = kwargs['keysIn']
        for i in range(len(argv)):
            self[self.keysIn[i]] = argv[i]
        if 'line' in kwargs:
            self.setByLine(kwargs['line'])
        for key in self.keys:
            if key in kwargs:
                self[key]=kwargs[key]
    def defaultSet(self):
        self.keys = ['']
        self.keysType =['s']
        self.keysIn = ['']
    def index(self,key):
        if key in self.keys:
            return self.keys.index(key)
        return -1
    def __getitem__(self,key):
        if not key in self.keys:
            print('no ',key)
        return self.l[self.index(key)]
    def __setitem__(self,key,value):
        if not key in self.keys:
            print('no ',key)
        self.l[self.index(key)] = value
    def setByLine(self, line):
        tmp = line.split()
        for i in range(len(tmp)):
            index = self.index(self.keysIn[i])
            self[self.keysIn[i]] = strType[self.keysType[index][0]](tmp[i])
    def __str__(self,*argv):
        line = ''
        if len(argv)>0:
            keysOut = argv[0]
        else:
            keysOut =self.keysIn
        for key in keysOut:
            line += str(self[key])+' '
        return line
    def __iter__(self):
        return self.keys
    def copy(self):
        type(self)()
        inD = {'keysIn':self.keys}
        for key in self:
            inD[key]  = self[key]
        return type(self)(**inD)

class Station(Dist):
    def __init__(self,*argv,**kwargs):
        super().__init__(*argv,**kwargs)
        if isinstance(self['compBase'],NoneType): 
            self['comp'] = [self['compBase']+s for s in 'ENZ' ]
        if isinstance( self['index'],NoneType) and isinstance( self['nickName'],NoneType):
            self['nickName'] = self.getNickName(self['index'])
    def defaultSet(self):
        self.keysIn   = 'net sta compBase lo la erroLo erroLa dep erroDep '.split()
        self.keys     = 'net sta compBase lo la erroLo erroLa dep erroDep nickName comp index nameFunc'.split()
        self.keysType ='S S S f f f f f f S l f F'.split()
    def getNickName(self, index):
        nickName = ''
        N      = len(nickStrL)
        for i in range(4):
            tmp    = index%N
            nickName += nickStrL[N]
            count  = int(index/N)
        return nickName


class StationList(list):
    def __init__(self,*argv,**kwargs):
        super().__init__()
        if isinstance(argv[0],list):
            for sta in argv[0]:
                self.append(sta)
        if isinstance(argv[0],str):
            self.setByFile(argv[0])
    def setByFile(self,fileName):
        self.header = []
        with open(fileName,'r') as staFile:
            lines = staFile.readlines()
        inD = {}
        if lines[0][0]=='#':
            keys = lines[0][1:].split()
            lines= lines[1:]
            inD['keysIn'] = keys
        index=0
        for line in lines:
            inD['line'] = line
            inD['index']=index
            self.append(Station(**inD))
            index+=1
        self.inD = inD
    def save(self,fileName,*argv):
        with open(fileName,'w+') as f:
            keysOut =''
            if 'keysIn' in self.inD:
                keysOut = '#'+self.inD['keysIn']+'\n'
            if len(argv)>0:
                keysOut = '#'+argv[0]+'\n'
            f.write(keysOut)
            for sta in self:
                f.write(sta)
                f.write('\n')
    def __str__(self):
        line =''
        for sta in self:
            line += '%s\n'%sta
        return line 
        
class Record(Dist):
    def __init__(self,*argv,**kwargs):
        super().__init__()
    def defaultSet(self):
        self.keysIn   = 'staIndex pTime sTime pProb sProb'.split()
        self.keys     = 'staIndex pTime sTime pProb sProb pCC sCC pM pS sM sS staName'.split()
        self.keysType = 'f        f     f     f      f    f   f   f  f  f  f  S'.split()

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