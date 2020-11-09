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
#注意最浅能到多少，最多能到多少
#如何挑选与控制
#160_w+_model
#降采样
#能load 但是损失函数未load
#验证新fv
#注意画图的节点和方框对应（shading）
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
		            'stationFileL': ['stations/CEA.sta_sel'],#**********'stations/CEA.sta_know_few'\
		            'oRemoveL'    : [False],\
		            'avgPairDirL' : ['models/ayu/Pairs_avgpvt/'],\
		            'pairDirL'    : ['models/ayu/Pairs_pvt/'],\
		            'minSNRL'     : [6],\
		            'isByQuakeL'  : [True],\
		            'remove_respL': [True],\
		            'isLoadFvL'   : [False],#False********\
		            'byRecordL'   : [False],
		            'maxCount'    : 4096*3,\
		            'trainDir'    : 'predict/1015_0.95_0.05_3.2_randMove_W+/',
		            'resDir'      : '/fastDir/results/1015_all_V?',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
		            'refModel'    : 'models/prem',\
		            'sacPara'     : sacPara,\
		            'dConfig'     : dConfig,\
		            'perN'        : 20,\
		            'eventDir'    : '/HOME/jiangyr/eventSac/',\
		            'T'           : (16**np.arange(0,1.000001,1/49))*10,\
		            'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
		            'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,240],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
		            'surPara'     : { 'nxyz':[50,75,0], 'lalo':[55,108],#[40,60,0][55,108]\
		                            'dlalo':[0.4,0.4], 'maxN':100,#[0.5,0.5]\
		        					'kmaxRc':0,'rcPerid':[],'threshold':0.01\
		        					,'maxIT':8,'nBatch':16,'smoothDV':10,'smoothG':20},\
		        	'runDir'      : 'DS/1026_CEA160_NE/',#_man/',\
		        	'gpuIndex'    : 0,\
		        	'gpuN'        : 1,\
		        	'lalo'        :[-1,180,-1,180],#[20,34,96,108][]*******,\
		        	'nlalo'        :[-1,-1,-1,-1],\
		        	'threshold'   :0.05,\
		        	'qcThreshold':2,\
		        	'minProb'     :0.5,\
		        	'minP'        :0.5,\
		        	'laL'         : [],\
		        	'loL'         : [],\
		        	'areasLimit'  :  3}
		self.para.update(sacPara)
		self.para.update(para)
		os.environ["CUDA_VISIBLE_DEVICES"]=str(self.para['gpuIndex'])
		self.para['surPara']['nxyz'][2]=len(self.para['z'])
		self.para['surPara'].update({\
		    	'kmaxRc':len(self.para['tSur']),'rcPerid': self.para['tSur'].tolist()})
		
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
	def loadCorr(self,isLoad=True):
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
			if isLoad:
				corrL0  = para['dConfig'].quakeCorr(q,sta,\
	    				byRecord=para['byRecordL'][i],remove_resp=para['remove_respL'][i],\
	    				minSNR=para['minSNRL'][i],isLoadFv=para['isLoadFvL'][i],\
	    				fvD=fvD,isByQuake=para['isByQuakeL'][i],para=para['sacPara'],resDir=para['eventDir'])
				corrL   += corrL0
		if isLoad:
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
		self.corrLTrain = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTrain)
		self.corrLTest  = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvTest)
		#corrLQuakePTrain = d.corrL(corrLQuakePCEA[:-1000])
		self.corrLValid = d.corrL(self.corrL1,specThreshold=specThreshold,fvD=fvVaild)
		#corrLQuakePTest  = d.corrL(corrLQuakePNE)
		#random.shuffle(corrLQuakePTrain)
		#random.shuffle(corrLQuakePValid)
		#random.shuffle(corrLQuakePTest)
		tTrain = para['T']
		self.corrLTrain.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=para['maxCount'],\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLTest.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=4096*3,\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLValid.setTimeDis(self.fvD,tTrain,sigma=1.5,maxCount=4096*3,\
		byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
		set2One=True,move2Int=False,randMove=True)
		self.corrLTrain(np.arange(10))
		print(self.corrLTrain.t0L)
		self.loadModel()
		fcn.trainAndTest(self.model,self.corrLTrain,self.corrLValid,self.corrLTest,\
	   		outputDir=para['trainDir'],sigmaL=[1.5],tTrain=tTrain,perN=200,count0=5,w0=3)#w0=3
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
		print(self.corrL.t0L)
	def loadRes(self):
		stations = []
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		self.stations = seism.StationList(stations)
		self.stations.inR(self.config.para['lalo'])
		print(len(self.stations))
		para    = self.config.para
		fvDGet,quakesGet = para['dConfig'].loadQuakeNEFV(self.stations,quakeFvDir=para['resDir'])
		self.fvDGet  = fvDGet
		d.qcFvD(self.fvDGet)
		self.getAv()
	def loadResAv(self):
		stations = []
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		self.stations = seism.StationList(stations)
		self.stations.inR(self.config.para['lalo'])
		self.stations.notInR(self.config.para['nlalo'])
		print(len(self.stations))
		para    = self.config.para
		self.fvAvGet,self.quakesGet = para['dConfig'].loadQuakeNEFVAv(self.stations,quakeFvDir=para['resDir'],\
			threshold=self.config.para['threshold'],minP=self.config.para['minP'])
		for fv in self.fvAvGet:
			self.fvAvGet[fv].qc(threshold=para['threshold'])
		d.qcFvD(self.fvAvGet)
	def loadAv(self,fvDir ='models/all/',mode='NEFileNew'):
		stations = []
		para    = self.config.para
		for staFile in self.config.para['stationFileL']:
			stations+=seism.StationList(staFile)
		stations = seism.StationList(stations)
		stations.inR(para['lalo'])
		stations.notInR(self.config.para['nlalo'])
		self.stations = seism.StationList(stations)
		self.fvAvGet = para['dConfig'].loadNEFV(stations,fvDir=fvDir,mode=mode)
	def getAv(self):
		for fv in self.fvDGet:
			self.fvDGet[fv].qc(threshold=-self.config.para['minP'])
		self.fvMGet  =d.fvD2fvM(self.fvDGet,isDouble=True)
		self.fvAvGet = d.fvM2Av(self.fvMGet)
		for fv in self.fvAvGet:
			self.fvAvGet[fv].qc(threshold=self.config.para['threshold'])
		d.qcFvD(self.fvAvGet)
	def getAV(self):
		self.fvAvGetL = [self.fvAvGet[key] for key in self.fvAvGet]
		self.FVAV     = d.averageFVL(self.fvAvGetL)
	def limit(self,threshold=3):
		for key in self.fvAvGet:
			self.FVAV.limit(self.fvAvGet[key],threshold=threshold)
		d.qcFvD(self.fvAvGet)
	def getAreas(self):
		self.areas=d.areas(laL=self.config.para['laL'],\
			loL=self.config.para['loL'],stations=self.stations)
	def areasLimit(self):
		#self.areas = self.getAreas()
		self.areas.Insert(self.fvAvGet)
		self.areas.getAv()
		self.areas.limit(self.fvAvGet,threshold=self.config.para['areasLimit'])
		d.qcFvD(self.fvAvGet)
	def preDS(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir'])
		self.DS = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations[-1::-1],1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.test(vL,indexL,self.stations[-1::-1])
	def preDSSyn(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'syn/',mode='syn')
		self.DS = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations[-1::-1],1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.testSyn(vL,indexL,self.stations[-1::-1])
	def preDSSynOld(self,do=True):
		para    = self.config.para
		tSur = para['tSur']
		z= para['z']
		surPara= para['surPara']
		DSConfig = DSur.config(para=surPara,z=z)
		DS = DSur.DS(config=DSConfig,runPath=para['runDir']+'syn/',mode='syn')
		self.DS = DS
		if do:
			indexL,vL = d.fvD2fvL(self.fvAvGet,self.stations,1/tSur)
			self.indexL = indexL
			self.vL   = vL
			DS.testSyn(vL,indexL,self.stations)
	def preDSOld(self):
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


paraOrdos={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 20,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [5,10,20,30,45,60,80,100,125,150,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'surPara'     : { 'nxyz':[40,47,0], 'lalo':[43,102],#[40,60,0][55,108]\
                    'dlalo':[0.3,0.3], 'maxN':100,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01\
					,'maxIT':8,'nBatch':4,'smoothDV':20,'smoothG':40},\
	'runDir'      : 'DS/1026_CEA160_Ordos_0.03/',#_man/',\
	'gpuIndex'    : 0,\
	'gpuN'        : 1,\
	'lalo'        :[32,42,103,115],#[20,34,96,108][-1,180,-1,180]*******,\
	'nlalo'        :[-1,-1,-1,-1],\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7}
paraYNSC={ 'quakeFileL'  : ['phaseLPickCEA'],\
		            'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
		            'oRemoveL'    : [False],\
		            'avgPairDirL' : ['models/ayu/Pairs_avgpvt/'],\
		            'pairDirL'    : ['models/ayu/Pairs_pvt/'],\
		            'minSNRL'     : [6],\
		            'isByQuakeL'  : [True],\
		            'remove_respL': [True],\
		            'isLoadFvL'   : [False],#False********\
		            'byRecordL'   : [False],
		            'maxCount'    : 4096*3,\
		            'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',
		            'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
		            'refModel'    : 'models/prem',\
		            'perN'        : 20,\
		            'eventDir'    : '/HOME/jiangyr/eventSac/',\
		            'T'           : (16**np.arange(0,1.000001,1/49))*10,\
		            'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
		            'z'           : [5,10,20,30,45,60,80,100,125,150,175,200,250,300,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
		            'surPara'     : { 'nxyz':[40,35,0], 'lalo':[36,96],#[40,60,0][55,108]\
		                            'dlalo':[0.4,0.4], 'maxN':100,#[0.5,0.5]\
		        					'kmaxRc':0,'rcPerid':[],'threshold':0.01\
		        					,'maxIT':32,'nBatch':16,'smoothDV':20,'smoothG':40},\
		        	'runDir'      : 'DS/1013_CEA160_YNSC/',#_man/',\
		        	'gpuIndex'    : 0,\
		        	'gpuN'        : 1,\
		        	'lalo'        :[20,34,96,108],#[20,34,96,108][-1,180,-1,180]*******,\
		        	'threshold'   :0.05,\
		        	'minProb'     :0.5,\
		        	'minP'        :0.5}

paraAll={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
                    'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : 'DS/1015_CEA160_all/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,-1,180],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3}
paraAll2={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
                    'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : 'DS/1026_CEA160_all/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,-1,180],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [35,30, 28,  35, 45],\
	'loL'         : [95,108,118,115,125],\
	'areasLimit'  :  3}

paraWest={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[56,88,0], 'lalo':[56,70],#[40,60,0][55,108]\
                    'dlalo':[0.8,0.8], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : 'DS/1026_CEA160_west/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,-1,100],#[20,34,96,108][]*******,\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [35,30, 28,  35, 45],\
	'loL'         : [95,108,118,115,125],\
	'areasLimit'  :  3}

paraEest={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[112,96,0], 'lalo':[56,102],#[40,60,0][55,108]\
                    'dlalo':[0.4,0.4], 'maxN':800,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.3,\
					'maxIT':100,'nBatch':100,'smoothDV':80,'smoothG':160},\
	'runDir'      : 'DS/1026_CEA160_east/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [28,  35, 45],\
	'loL'         : [110,115,125],\
	'areasLimit'  :  3}

paraNECE={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/NEsta_all.locSensorDas'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : 'models/NEFVSEL/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,240],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[21,52,0], 'lalo':[48,115],#[40,60,0][55,108]\
                    'dlalo':[0.4,0.4], 'maxN':129,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 0.4,\
					'maxIT':8,'nBatch':8,'smoothDV':10,'smoothG':20},\
	'runDir'      : 'DS/1026_CEA160_NECE_SEL/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[-1,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,-1,-1,-1],\
	'threshold'   :0.03,\
	'minProb'     :0.5,\
	'minP'        :0.7,\
	'laL'         : [],\
	'loL'         : [],\
	'areasLimit'  :  3}

paraNorth={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[65,96,0], 'lalo':[56,102],#[40,60,0][55,108]\
                    'dlalo':[0.4,0.4], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':30,'smoothDV':20,'smoothG':40},\
	'runDir'      : 'DS/1026_CEA160_north/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[32,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.05,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'areasLimit'  :  3}

paraNorthLager={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[52,71,0], 'lalo':[56,102],#[40,60,0][55,108]\
                    'dlalo':[0.5,0.5], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':30,'smoothDV':20,'smoothG':40},\
	'runDir'      : 'DS/1026_CEA160_north_lager/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[32,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.05,\
	'minProb'     :0.5,\
	'minP'        :0.5,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'areasLimit'  :  3}

paraNorthLagerNew={ 'quakeFileL'  : ['phaseLPickCEA'],\
    'stationFileL': ['stations/CEA.sta_know_few'],#**********'stations/CEA.sta_know_few'\
    'isLoadFvL'   : [False],#False********\
    'byRecordL'   : [False],\
    'trainDir'    : 'predict/1010_0.95_0.05_3.2_randMove/',\
    'resDir'      : '/fastDir/results/1015_all_V?/',#'models/ayu/Pairs_pvt/',#'results/1001/',#'results/1005_allV1/',\
    'perN'        : 1,\
    'eventDir'    : '/HOME/jiangyr/eventSac/',\
    'z'           : [0,5,10,15,20,25,30,35,45,55,65,80,100,130,160,200,270,350],#[5,10,20,30,45,60,80,100,125,150,175,200,250,300,350](350**(np.arange(0,1.01,1/18)+1/18)).tolist(),\
    'tSur'        : (16**np.arange(0,1.000001,1/24.5))*10,\
    'surPara'     : { 'nxyz':[52,71,0], 'lalo':[56,102],#[40,60,0][55,108]\
                    'dlalo':[0.5,0.5], 'maxN':350,#[0.5,0.5]\
					'kmaxRc':0,'rcPerid':[],'threshold':0.01,'sparsity': 1,\
					'maxIT':30,'nBatch':30,'smoothDV':20,'smoothG':40},\
	'runDir'      : 'DS/1026_CEA160_north_lager_new/',#_man/',\
	'gpuIndex'    : 1,\
	'gpuN'        : 2,\
	'lalo'        :[32,180,103,180],#[20,34,96,108][]*******,\
	'nlalo'        :[-1,35,-1,106],\
	'threshold'   :0.04,\
	'minProb'     :0.5,\
	'minP'        :0.6,\
	'laL'         : [40,38,45],\
	'loL'         : [110,120,125],\
	'areasLimit'  :  3}
