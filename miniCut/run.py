import dispersion as d
from imp import reload
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mathFunc
import fcn
import seism
plt.switch_backend('agg')
orignExe='/home/jiangyr/program/fk/'
absPath = '/home/jiangyr/home/Surface-Wave-Dispersion/'
srcSacDir='/home/jiangyr/Surface-Wave-Dispersion/srcSac/'
srcSacDirTest='/home/jiangyr/Surface-Wave-Dispersion/srcSacTest/'
T=np.array([0.5,1,2,5,8,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200,225,250,275,300])
para={'freq'      :[1/6],'filterName':'lowpass'}
dConfig=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=110*10,maxDist=110*170,minDDist=200,\
        maxDDist=10000,para=para,isFromO=True,removeP=True)

class runConfig:
	def __init__(self,para={}):
		sacPara = {'pre_filt': (1/400, 1/300, 1/2, 1/1.5),\
                   'output':'VEL','freq':[1/300, 1/5],\
                   'filterName':'bandpass',\
                   'corners':4,'toDisp':False,\
                   'zerophase':True,'maxA':1e15}
		self.para={ 'quakeFileL'  : ['phaseLPickCEA'],\
		            'stationFileL': ['stations/CEA.sta_sel'],\
		            'oRemoveL'    : [False],\
		            'avgPairDirL' : ['models/ayu/Pairs_avgpvt/'],\
		            'pairDirL'    : ['models/ayu/Pairs_avgpvt/'],\
		            'minSNRL'     : [8],\
		            'isByQuakeL'  : [True],\
		            'remove_respL': [True],\
		            'isLoadFvL'   : [False],\
		            'byRecordL'   : [False],
		            'maxCount'    : 4096*3,\
		            'trainDir'    : 'predict/0930/',
		            'resDir'      : 'results/0930'
		            'refModel'    : 'models/prem',\
		            'sacPara'     : sacPara,\
		            'dConfig'     : dConfig,\
		            'T'           : (16**np.arange(0,1.000001,1/49))*10,\
		            'lalo'        : [0,90,0,180]}
		            
		self.para.update(sacPara)

class run:
    def __init__(self,config=runConfig()):
		self.config = config
		self.model  = None
	def loadCorr(self):
		config     = self.config
		corrL      = []
		station    = []
		quakes     = []
		fvDAvarage = {}
		fvD        = {}
		para       = config.para
		N          = len(para['stationFileL'])
		fvDAvarage['models/prem']=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		for i in range(N):
			sta     = seism.StationList(para['stationFileL'][i])
			sta.inR(para['lalo'])
			sta.set('oRemove', para['oRemoveL'][i])
			sta.getInventory()
			station += sta
			q       = seism.QuakeL(para['quakeFileL'][i])
			quakes  += q
			fvDA    = para['dConfig'].loadNEFV(sta,fvDir=para['avgPairDirL'][i])
			fvDAvarage.update(fvDA)
			fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,fvDir=para['pairDirL'][i])
			d.replaceByAv(fvd,fvDA)
			fvD.update(fvd)
			corrL0  = para['dConfig'].config.quakeCorr(q,sta,\
    				byRecord=para['byRecordL'][0],remove_resp=para['remove_respL'][0],\
    				minSNR=para['minSNRL'][0],isLoadFv=para['isLoadFvL'][0],\
    				fvD=fvDA,isByQuake=para['isByQuakeL'][0],para=para['sacPara'])
			corrL   += corrL0
		self.corrL  = d.corrL(corrL,maxCount=para['maxCount'])
		self.corrL1 = d.corrL(self.corrL,maxCount=para['maxCount'],fvD=fvD)
		self.fvD    = fvD
		self.fvDAvarage = fvDAvarage
		self.quakes = quakes
		self.stations = stations
	def train(self):
		fvL = [key for key in self.fvDAvarage]
		random.shuffle(fvL)
		fvN = len(fvL)
		fvn = int(fvN/10)
		fvTrain = fvL[fvn*2:]
		fvTest  = fvL[fvn:fvn*2]
		fvVaild = fvL[:fvn]
		para    = self.config.para
		specThreshold = 0.0
		corrLTrain = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTrain)
		corrLTest  = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTest)
		#corrLQuakePTrain = d.corrL(corrLQuakePCEA[:-1000])
		corrLValid = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvVaild)
		#corrLQuakePTest  = d.corrL(corrLQuakePNE)
		#random.shuffle(corrLQuakePTrain)
		#random.shuffle(corrLQuakePValid)
		#random.shuffle(corrLQuakePTest)
		tTrain = para['T']
		corrLTrain.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False)
		corrLTest.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=4096*3,\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False)
		corrLValid.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=4096*3,\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False)
		self.model = self.loadModel()
		fcn.trainAndTest(self.model,corrLTrain,corrLValid,corrLTest,\
    		outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=200,count0=5,w0=3)

    def loadModel(self,file=''):
    	if self.model == None:
    		self.model = fcn.model(channelList=[0,2,3])
    	if file != '':
    		self.model.load_weights(file, by_name= True)
    def calRes(self):
    	para = self.config.para
		self.corrL.setTimeDis(self.fvDAvarage,para[T],sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=True,\
		set2One=True,move2Int=False,modelNameO=para['refModel'])
		self.corrL.getAndSave(self.model,'%s/CEA_P_'%para['resDirs'],self.stations\
    	,isPlot=True,isLimit=True,isSimple=False,D=0.2)
