import dispersion as d
import os
from imp import reload
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mathFunc
import fcn
import seism
import random
import DSur
#主要还是需要看如何分批次更新
#提前分配矩阵可能不影响
#插值之后绘图，避免不对应
#剩最后一部分校验
#以个点为中心点，方块内按节点插值
#是z数量的1.5倍左右即可
#是否去掉SH台站
#尝试通过射线路径确定分辨率范围
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
        maxDDist=1801,para=para,isFromO=True,removeP=True)

class runConfig:
	def __init__(self,para={}):
		sacPara = {'pre_filt': (1/400, 1/300, 1/2, 1/1.5),\
                   'output':'VEL','freq':[1/200,1/8],\
                   'filterName':'bandpass',\
                   'corners':4,'toDisp':False,\
                   'zerophase':True,'maxA':1e15}
		self.para={ 'quakeFileL'  : ['phaseLPickCEA'],\
		            'stationFileL': ['stations/CEA.sta_know_few'],#'stations/CEA.sta_know_few'\
		            'oRemoveL'    : [False],\
		            'avgPairDirL' : ['models/ayu/Pairs_avgpvt/'],\
		            'pairDirL'    : ['models/ayu/Pairs_pvt/'],\
		            'minSNRL'     : [6],\
		            'isByQuakeL'  : [True],\
		            'remove_respL': [True],\
		            'isLoadFvL'   : [False],#False\
		            'byRecordL'   : [False],
		            'maxCount'    : 4096*3,\
		            'trainDir'    : 'predict/1010_0.95_0.05_1.8/',
		            'resDir'      : 'results/1011_YNSC_V1/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
		            'refModel'    : 'models/prem',\
		            'sacPara'     : sacPara,\
		            'dConfig'     : dConfig,\
		            'perN'        : 10,\
		            'eventDir'    : '/HOME/jiangyr/eventSac/',\
		            'T'           : (16**np.arange(0,1.000001,1/49))*10,\
		            'lalo'        : [0,90,0,180],\
		            'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
		            'z'           : [5,10,20,30,45,60,80,100,125,150,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
		            'surPara'     : { 'nxyz':[40,50,0], 'lalo':[36,96],#[40,60,0][55,108]\
		                            'dlalo':[0.5,0.5], 'maxN':100,#[0.5,0.5]\
		        					'kmaxRc':0,'rcPerid':[],'threshold':0.01\
		        					,'maxIT':32,'nBatch':16,'smoothDV':20,'smoothG':40},\
		        	'runDir'      : 'DS/1005_CEA160_YNSC/',#_man/',\
		        	'gpuIndex'    : 0,\
		        	'gpuN'        : 1,\
		        	'lalo'        :[20,34,96,108],#[-1,180,-1,180],\
		        	'threshold'   :0.08,\
		        	'minProb'     :0.7}
		os.environ["CUDA_VISIBLE_DEVICES"]=str(self.para['gpuIndex'])
		self.para['surPara']['nxyz'][2]=len(self.para['z'])
		self.para['surPara'].update({\
		    	'kmaxRc':len(self.para['tSur']),'rcPerid': self.para['tSur'].tolist()})
		self.para.update(sacPara)
class run:
	def __init__(self,config=runConfig(),self1 = None):
		self.config = config
		self.model  = None
		if self1 != None:
			self.corrL  =  self1.corrL
			self.corrL1 =  self1.corrL1
			self.fvD    =  self1.fvD
			self.fvDAvarage = self1.fvDAvarage
			self.quakes = self1.quakes
			self.stations = self1.stations
	def loadCorr(self):
		config     = self.config
		corrL      = []
		stations   = seism.StationList([])
		quakes     = seism.QuakeL()
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
			stations += sta
			q       = seism.QuakeL(para['quakeFileL'][i])
			quakes  += q
			fvDA    = para['dConfig'].loadNEFV(sta,fvDir=para['avgPairDirL'][i])
			fvDAvarage.update(fvDA)
			fvd, q0 = para['dConfig'].loadQuakeNEFV(sta,quakeFvDir=para['pairDirL'][i])
			d.replaceByAv(fvd,fvDA)
			fvD.update(fvd)
			corrL0  = para['dConfig'].quakeCorr(q,sta,\
    				byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],\
    				minSNR=para['minSNRL'][i],isLoadFv=para['isLoadFvL'][i],\
    				fvD=fvD,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],resDir=para['eventDir'])
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
		self.loadModel()
		fcn.trainAndTest(self.model,corrLTrain,corrLValid,corrLTest,\
	   		outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=200,count0=5,w0=3)
	def loadModel(self,file=''):
		if self.model == None:
			self.model = fcn.model(channelList=[0,2,3])
		if file != '':
			self.model.load_weights(file, by_name= True)
	def calResOneByOne(self):
		config     = self.config
		para       = config.para
		N          = len(para['stationFileL'])
		fvDAvarage = {}
		fvDAvarage['models/prem']=d.fv(para['refModel']+'_fv_flat_new_p_0','file')
		for i in range(N):
			sta     = seism.StationList(para['stationFileL'][i])
			sta.inR(para['lalo'])
			sta.set('oRemove', para['oRemoveL'][i])
			sta.getInventory()
			q       = seism.QuakeL(para['quakeFileL'][i])
			self.fvDAvarage = fvDAvarage
			self.stations = sta
			q.sort()
			perN= self.config.para['perN']
			for j in range(self.config.para['gpuIndex'],int(len(q)/perN),self.config.para['gpuN']):#self.config.para['gpuIndex']
				corrL0  = para['dConfig'].quakeCorr(q[j*perN:min(len(q)-1,j*perN+perN)],sta,\
	    				byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],\
	    				minSNR=para['minSNRL'][i],isLoadFv=para['isLoadFvL'][i],\
    					fvD=fvDAvarage,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],\
    					resDir=para['eventDir'])
				self.corrL  = d.corrL(corrL0,maxCount=para['maxCount'])
				if len(self.corrL)==0:
					continue
				self.calRes()
				corrL0 = 0
				self.corrL = 0
	def calRes(self):
		para = self.config.para
		self.corrL.setTimeDis(self.fvDAvarage,para['T'],sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=False,rThreshold=0.0,byAverage=True,\
		set2One=True,move2Int=False,modelNameO=para['refModel'],noY=True)
		self.corrL.getAndSaveOld(self.model,'%s/CEA_P_'%para['resDir'],self.stations\
		,isPlot=False,isLimit=False,isSimple=True,D=0.2,minProb = para['minProb'])
	def loadRes(self):
		stations = []
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		self.stations = seism.StationList(stations)
		self.stations.inR(self.config.para['lalo'])
		print(len(self.stations))
		para    = self.config.para
		fvDGet,quakesGet = para['dConfig'].loadQuakeNEFV(self.stations,quakeFvDir=para['resDir'])
		fvMGet  =d.fvD2fvM(fvDGet,isDouble=True)
		fvAvGet = d.fvM2Av(fvMGet)
		for fv in fvAvGet:
			fvAvGet[fv].qc(threshold=self.config.para['threshold'])
		self.fvDGet  = fvDGet
		self.fvMGet  = fvMGet
		self.fvAvGet = fvAvGet
		d.qcFvD(fvAvGet)
	def preDS(self):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir'])
		self.DS = DS
		indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations,1/tSur)
		self.indexL = indexL
		self.vL   = vL
		DS.test(vL,indexL,self.stations)
	def loadAndPlot(self):
		self.DS.loadRes()
		self.DS.plotByZ()

	def test(self):
		self.loadCorr()
		self.train()
