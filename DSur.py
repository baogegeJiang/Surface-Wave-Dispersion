import numpy as np
import os 
from scipy import interpolate,stats
from matplotlib import pyplot as plt
from distaz import DistAz
from netCDF4 import Dataset
import mapTool as mt
import pycpt
cmap = pycpt.load.gmtColormap('cpt/temperatureInv')
##程序中又控制的大bug
##必须按顺序(period source)
faultL = mt.readFault('Chinafault_fromcjw.dat')
class config:
	def __init__(self,para={},name='ds',z=[10,20,40,80,120,160,200,320]):
		self.name = name
		self.z    = z
		config.keyList = ['dataFile', 'nxyz', 'lalo', 'dlalo', 'maxN','damp',\
		'sablayers','minmaxV', 'maxIT','sparsity','kmaxRc','rcPerid','kmaxRg','rgPeriod',\
		'kmaxLc','lcPeriod','kmaxLg','lgPeriod','isSyn','noiselevel','threshold',\
		'vnn']
		config.keyList = ['dataFile', 'nxyz', 'lalo', 'dlalo', 'sablayers','minmaxV',\
		'maxN','sparsity', 'maxIT','iso','c','smoothDV','smoothG','Damp',\
		'c','kmaxRc','rcPerid','nBatch']
		config.keyListSyn = ['dataFile', 'nxyz', 'lalo', 'dlalo','maxN', 'sablayers',\
		'sparsity','rayT','kmaxRc','rcPerid','noise']
		self.para = {'dataFile':name+'in', 'nxyz':[18,18,9], 'lalo':[130,30],\
		 'dlalo':[0.01,0.01], 'maxN':[20],'damp':[4.0,1.0],\
		'sablayers':3,'minmaxV':[1,7],'maxIT':10, 'sparsity':0.4,\
		'kmaxRc':10,'rcPerid':np.arange(1,11).tolist(),'kmaxRg':0,'rgPeriod':[],\
		'kmaxLc':0,'lcPeriod':[],'kmaxLg':0,'lgPeriod':[],'isSyn':0,'noiselevel':0.02,'threshold':0.05,\
		'vnn':[0,100,50],'iso':'F','c':'c','smoothDV':10,'smoothG':20,'Damp':0,'perN':[6,6,4],'perA':0.05,\
		'modelPara': {'config':self,'mode':'prem','runPath':'','file':'models/prem','la':'','lo':'','z':'','self1':'',\
		},'rayT':'F','noise':0,\
		'GSPara': {'config':self,'mode':'GS','runPath':'','file':'','la':'','lo':'','z':'','self1':'',\
		},\
		'GCPara': {'config':self,'mode':'GC','runPath':'','file':'','la':'','lo':'','z':'','self1':'',\
		},}
		self.para.update(para)
	def output(self):
		nxyz = self.para['nxyz']
		la = self.para['lalo'][0]-np.arange(nxyz[0])*self.para['dlalo'][0]
		lo = self.para['lalo'][1]+np.arange(nxyz[1])*self.para['dlalo'][1]
		return nxyz,la,lo,self.z
	def outputP(self):
		nxyz = self.para['nxyz'][:2]+[len(self.para['rcPerid'])]
		la = self.para['lalo'][0]-np.arange(nxyz[0])*self.para['dlalo'][0]
		lo = self.para['lalo'][1]+np.arange(nxyz[1])*self.para['dlalo'][1]
		return nxyz,la,lo,np.array(self.para['rcPerid'])
	def findLaLo(self,la,lo):
		laNew = -round((self.para['lalo'][0]-la)/self.para['dlalo'][0])*self.para['dlalo'][0]+self.para['lalo'][0]+0.001
		loNew =  round((lo-self.para['lalo'][1])/self.para['dlalo'][1])*self.para['dlalo'][1]+self.para['lalo'][1]+0.001
		return la,lo
class DS:
	"""docstring for ClassName"""
	def __init__(self,runPath='DS/',config=config(),mode='real'):
		self.runPath = runPath
		if not os.path.exists(runPath):
			os.mkdir(runPath)
		self.config = config
		self.config.para['modelPara']['runPath']= runPath
		self.config.para['GSPara']['runPath']= runPath
		self.config.para['GCPara']['runPath']= runPath
		self.mode = mode

	def writeInput(self,perI=-1):
		'''
		cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
		c SurfAnisoForward Input
		cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
		Surfphase_RV3_5_40s_1s.dat          c: blank traveltime data file
		17 17 4                             c: nx ny nz (grid number in lat lon and depth directions)
		26.5  101.25                        c: goxd gozd (upper left point,[lat,lon])
		0.25 0.25                           c: dvxd dvzd (grid interval in lat and lon directions)
		5000                                c: max(sources, receivers)
		2                                   c: sublayers (2~5)
		0.2                                 c: sparsity fraction
		F                                   c: T: output raypath; F: not output raypath
		36                                  c: kmaxRc (followed by periods)
		5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
		0                                   c: noise level e.g.: 0.5s

		cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
		c INPUT PARAMETERS
		cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
		surfdataTB.dat                   c: data file
		18 18 9                          c: nx ny nz (grid number in lat lon and depth direction)
		25.2  121.35                     c: goxd gozd (upper left point,[lat,lon])
		0.015 0.017                      c: dvxd dvzd (grid interval in lat and lon direction)
		20                               c: max(sources, receivers)
		4.0  1.0                         c: weight damp
		3                                c: sablayers (for computing depth kernel, 2~5)
		0.5 2.8                          c: minimum velocity, maximum velocity (a priori information)
		10                               c: maximum iteration
		0.2                              c: sparsity fraction
		26                               c: kmaxRc (followed by periods)
		0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0
		0                                c: kmaxRg
		0                                c: kmaxLc
		0                                c: kmaxLg
		0                                c: synthetic flag(0:real data,1:synthetic)
		0.02                             c: noiselevel
		0.05                             c: threshold
		0 100 50                         c: vorotomo,ncells,nrelizations
		'''
		with open(self.runPath+self.config.name, 'w+') as f:
		    f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
		    f.write('c INPUT PARAMETERS\n')
		    f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
		    for key in self.config.keyList:
		    	value = self.config.para[key]
		    	if isinstance(value,list):
		    		valueNew = ''
		    		for v in value:
		    			valueNew += str(v)+' '
		    		value = valueNew
		    	value = str(value)
		    	if value =='':
		    		continue
		    	f.write(value+' c: '+key+'\n')
		with open(self.runPath+self.config.name+'syn', 'w+') as f:
		    f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
		    f.write('c INPUT PARAMETERS\n')
		    f.write('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n')
		    for key in self.config.keyListSyn:
		    	value = self.config.para[key]
		    	if isinstance(value,list):
		    		valueNew = ''
		    		for v in value:
		    			valueNew += str(v)+' '
		    		value = valueNew
		    	value = str(value)
		    	if value =='':
		    		continue
		    	if perI >=0:
			    	if key =='kmaxRc':
			    		value = '1'
			    	if key =='rcPerid':
			    		value = str(self.config.para[key][perI])
		    	f.write(value+' c: '+key+'\n')
	def writeData(self,fvLL,indexL,stations,waveType=0,perI=-1,perJ=0,M=1):
		staN = len(stations)
		distM = np.zeros([staN, staN])
		for i in range(staN):
			for j in range(i+1):
				dist = DistAz(0,0,0,0).degreesToKilometers(\
					DistAz(stations[i]['la'],stations[i]['lo'],\
						stations[j]['la'],stations[j]['lo']).getDelta())
				distM[i,j] = dist
				distM[j,i] = dist
		with open(self.runPath+'/'+self.config.para['dataFile'],'w') as f:
			for j in range(self.config.para['kmaxRc']):
				if perI >=0:
					if j != perI:
						continue
				for i in range(staN):
					if i%M!=perJ:
						continue
					if len(indexL[i])==0:
						continue
					vL =np.zeros(len(indexL[i]))
					for k in range(len(indexL[i])):
						vL[k]=fvLL[i][k][j]
					nvL = (vL>2).sum()
					if nvL<2:
						continue
					la,lo=self.config.findLaLo(stations[i]['la'],\
						stations[i]['lo'])
					if perI>=0:
						f.write('# %.3f %.3f %d 2 0\n'%(la,lo,j+1-perI))
					else:
						f.write('# %.3f %.3f %d 2 0\n'%(la,lo,j+1))
					for k in range(len(indexL[i])):
						kk = indexL[i][k]
						if vL[k]>2 and vL[k]<6:
							la,lo=self.config.findLaLo(stations[kk]['la'],\
							stations[kk]['lo'])
							f.write('%.3f %.3f %f\n'%(la,lo,vL[k]))
	def writeMod(self):
		nx,ny,nz=self.config.para['nxyz']
		dep1=np.array(self.config.z)
		nxyz,la,lo,z = self.config.output()
		#dep1=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.3,1.5,1.8,2.1,2.5])
		nz=len(dep1)
		#ends
		vs1=np.zeros(nz)
		self.model = Model(**self.config.para['modelPara'])
		self.model.write(self.runPath+'/MOD',la,lo,z)
		if self.mode == 'syn':
			self.GC    = Model(**self.config.para['GCPara'])
			self.GS    = Model(**self.config.para['GSPara'])
			self.model.write(self.runPath+'/MODVs.true',la,lo,z,isDiff=True,N=self.config.para['perN']\
				,A=self.config.para['perA'])
			self.GS.write(self.runPath+'/MODGs.true',la,lo,z,isDiff=True,N=self.config.para['perN']\
				,A=self.config.para['perAGs'])
			self.GC.write(self.runPath+'/MODGc.true',la,lo,z,isDiff=True,N=self.config.para['perN']\
				,A=self.config.para['perAGc'])
		for i in range(nz):
		  print (dep1[i]),
	def test(self,fvLL,indexL,stations):
		self.writeInput()
		self.writeMod()
		self.writeData(fvLL,indexL,stations)
	def testSyn(self,fvLL,indexL,stations):
		self.writeMod()
		periods = ['' for i in range(self.config.para['kmaxRc'])]
		for i in range(self.config.para['kmaxRc']):
			self.writeInput(perI=i)
			for j in range(3):
				self.writeData(fvLL,indexL,stations,perI=i,perJ=j,M=3)
				os.system('cd %s;SurfAAForward dssyn'%self.runPath)
				with open('%s/surfphase_forward.dat'%self.runPath,'r') as res:
					for line in res.readlines():
						if line[0] == '#':
							# 39.914 116.241 11 2 0
							tmp = line.split()
							tmp[-3] = '%d'%(i+1)
							line='#'
							for t in tmp[1:]:
								line += ' '+t
							line += '\n'
						else:
							tmp = line.split()
							v   = float(tmp[-1])
							v   *= 1+2*(np.random.rand()-0.5)*0.008
							tmp[-1] = '%.6f'%v
							line=''
							for t in tmp:
								line += ' '+t
							line += '\n'
						periods[i]+=line
						if np.random.rand()<0.001:
							print(line)
			with open('%s/dsin_syn'%self.runPath,'w+') as f:
				for period in periods:
					f.write(period)
	def plotByZ(self):
		if self.mode == 'syn':
			self.modelTrue.plotByZ(self.runPath,head='true')
		self.modelPeriod.plotByZ(self.runPath,head='period')
		self.modelRes.plotByZ(self.runPath)
	def plotTK(self):
		nxyz,la,lo,z = self.config.output()
		z0,la0,lo0,vsv0= loadModelTK()
		resDir = self.runPath+'/'+'plot/'
		if not os.path.exists(resDir):
			os.mkdir(resDir)
		for i in range(nxyz[-1]):
			index = np.abs(z[i]-z0).argmin()
			v = interpolate.interp2d(lo0,la0,vsv0[index])(lo,la) 
			plt.close()
			plt.pcolor(lo,la,-v,cmap=cmap)
			plt.colorbar()
			plt.title('%f.jpg'%z[i])
			plt.savefig('%s/TK_%f.jpg'%(resDir,z[i]),dpi=200)
			plt.ylim([35,55])
			plt.close()
	def loadRes(self):
		if self.mode == 'syn':
			self.modelTrue = Model(self.config,mode='DSFile',runPath=self.runPath,file='MODVs.true')
		self.modelPeriod = Model(self.config,mode='DSP',runPath=self.runPath,file='period_Azm_tomo.inv')
		#self.GsTrue = Model(self.config,mode='GSO',runPath=self.runPath,file='MODGs.true')
		#self.GcTrue = Model(self.config,mode='GCO',runPath=self.runPath,file='MODGc.true')
		#self.modelInit = Model(self.config,mode='DSFile',runPath=self.runPath,file='MOD')
		self.modelRes = Model(self.config,mode='DS',runPath=self.runPath,file='Gc_Gs_model.inv')
		#self.GsRes = Model(self.config,mode='GS',runPath=self.runPath,file='Gc_Gs_model.inv')
		#self.GcRes = Model(self.config,mode='GC',runPath=self.runPath,file='Gc_Gs_model.inv')

class Model:
	def __init__(self,config=None,mode='DS',runPath='',file='',la='',lo='',z='',self1=''):
		self.mode = mode
		self.config=config
		if mode =='DS':
			data = np.loadtxt(runPath+file)
			nxyz,la,lo,z = config.output()
			z    = np.array(z)
			v = np.zeros(nxyz)*0-1
			for i in range(data.shape[0]):
				Lo = data[i,0]
				La = data[i,1]
				Z  = data[i,2]
				V  = data[i,3]
				i0 = np.abs(la-La).argmin()
				i1 = np.abs(lo-Lo).argmin()
				i2 = np.abs(z-Z).argmin()
				v[i0,i1,i2]=V
		if mode =='DSP':
			data = np.loadtxt(runPath+file)
			nxyz,la,lo,z = config.outputP()
			z    = np.array(z)
			v = np.zeros(nxyz)*0-1
			for i in range(data.shape[0]):
				Lo = data[i,0]
				La = data[i,1]
				Z  = data[i,2]
				V  = data[i,3]
				i0 = np.abs(la-La).argmin()
				i1 = np.abs(lo-Lo).argmin()
				i2 = np.abs(z-Z).argmin()
				v[i0,i1,i2]=V
		if mode == 'DSFile':
			nxyz,la,lo,z = config.output()
			v = np.loadtxt(runPath+file,skiprows=1)
			v=v.reshape(nxyz[2],nxyz[1],nxyz[0]).transpose([2,1,0])
		if mode =='TK':
			z,la,lo,v=loadModelTK()
			#.reshape([-1])
			#shape = self.vsv
			#self.vsv=self.vsv.reshape([-1])
			v=v.transpose([1,2,0])
		if mode=='byModel':
			v = self1.output(la,lo,z)
		if mode=='prem':
			data = np.loadtxt(file)
			la = np.array([-90,90])
			lo = np.array([-180,180])
			z = data[:,0]
			V = data[:,2]
			v = np.zeros([2,2,len(V)])
			for i in range(2):
				for j in range(2):
					v[i,j] = V
		if mode=='GC' or mode=='GS':
			nxyz,la,lo,z = config.output()
			v = np.zeros(nxyz)*0+0.004
			if file!='':
				data = np.loadtxt(runPath+file)
				for i in range(data.shape[0]):
					Lo = data[i,0]
					La = data[i,1]
					Z  = data[i,2]
					gc = data[i,6] 
					gs = data[i,7] 
					i0 = np.abs(la-La).argmin()
					i1 = np.abs(lo-Lo).argmin()
					i2 = np.abs(z-Z).argmin()
					if mode == 'GC':
						v[i0,i1,i2]=gc/100
					else:
						v[i0,i1,i2]=gs/100
		if mode=='GCO' or mode=='GSO':
			nxyz,la,lo,z = config.output()
			v = np.zeros(nxyz)*0+0.01
			if file!='':
				v = np.loadtxt(runPath+file)
				v=v.reshape(nxyz[2],nxyz[1],nxyz[0]).transpose([2,1,0])
		self.nxyz = [len(la),len(lo),len(z)]
		self.z  =  z#.reshape([-1,1,1])
		self.la = la#.reshape([1,-1,1])
		self.lo = lo#.reshape([1,1,-1])
		self.v  = v
	def __call__(self,la,lo,z):
		i0 = np.abs(self.la - la).argmin()
		i1 = np.abs(self.lo - lo).argmin()
		i2 = np.abs(self.z  - z).argmin()
		v = self.v[i0,i1,i2]
		return v 
	def output(self,la,lo,z):
		nxyz    = [len(la),len(lo),len(z)]
		nxyzTmp = [len(la),len(lo),len(self.z)]
		v       = np.zeros(nxyz)
		vTmp    = np.zeros(nxyzTmp)
		#print(z)

		for i in range(nxyzTmp[-1]):
			vTmp[:,:,i] = interpolate.interp2d(self.lo, self.la, self.v[:,:,i],bounds_error=False,fill_value=1e-8)(lo,la)
		if la[-1]<la[0]:
			vTmp = vTmp[::-1]
		if lo[-1]<lo[0]:
			vTmp = vTmp[:,::-1]
		for i in range(nxyz[0]):
			for j in range(nxyz[1]): 
				v[i,j,:] = interpolate.interp1d(self.z,vTmp[i,j])(z)
		return v	
	def denseLaLo(self,per,N=500):
		dLa = (self.la[-1]-self.la[0])/N
		dLo = (self.lo[-1]-self.lo[0])/N
		la  = np.arange(self.la[0],self.la[-1]+1e-5*dLa,dLa)[-1::-1]
		lo  = np.arange(self.lo[0],self.lo[-1]+1e-5*dLa,dLo)
		per = interpolate.interp2d(self.lo, self.la, per)(lo,la)
		return la, lo, per
	def plotByZ(self,runPath='DS',head='res'):
		resDir = runPath+'/'+'plot/'
		if not os.path.exists(resDir):
			os.mkdir(resDir)
		nxyz,la,lo,z = self.config.output()
		if self.mode=='DSP':
			nxyz,la,lo,z = self.config.outputP()
		V            =  self.output(la,lo,z)
		for i in range(self.nxyz[-1]):
			plt.close()
			plt.figure(figsize=[12,8])
			R = [la.min(),la.max(),lo.min(),lo.max()]
			m = mt.genBaseMap(R)
			v = V[:,:,i]
			#print(v[:10,:10])
			laN, loN = v.shape
			midLaN = int(1.2/4*laN)
			midLoN = int(1.2/4*loN)
			mean = v[midLaN:-midLaN,midLoN:-midLoN].mean()
			print(mean)
			v[v<0]=mean
			Per   = (v-mean)/mean
			la,lo,per=denseLaLo(self.la,self.lo,Per)
			x,y= m(lo,la)
			vmin=-np.abs(Per[midLaN:-midLaN,midLoN:-midLoN]).max()
			vmax=np.abs(Per[midLaN:-midLaN,midLoN:-midLoN]).max()
			plotPlane(m,x,y,per,R,z[i],mean,vmin,vmax,isFault=True,head=head)
			plt.savefig('%s/%s_%f.jpg'%(resDir,head,self.z[i]),dpi=500)
			plt.close()
	def write(self,filename,la,lo,z,isDiff=False,N=[2,2,2],A=0.05):
		with open(filename,'w') as fp:
			if self.mode != 'GS' and self.mode!='GC':
			    for i in range(len(z)):
			        fp.write('%9.1f' % (z[i]))
			    fp.write('\n')
			v = self.output(la,lo,z)
			for k in range(len(z)):
				for j in range(len(lo)):
					for i in range(len(la)):
						per=self.diff(isDiff,N,i,j,k,A)
						#print(N,i,j,k,per)
						fp.write('%9.3f' % (v[i,j,k]*(1+per)))
					fp.write('\n')
	def diff(self,isDiff,N,i,j,k,A):
		if isDiff:
			I = int(float(i)/float(N[0]))
			J = int(float(j)/float(N[1]))
			K = int(float(k)/float(N[2]))
			#print(I+J+K)
			return ((I+J+K)%2-0.5)*2*A
		else:
			return 0
	def compare(self,self1,runPath='DS',head='compare'):
		resDir = runPath+'/'+'plot/'
		if not os.path.exists(resDir):
			os.mkdir(resDir)
		z = self.z.tolist()
		z.append(0)
		nxyz,la,lo,z = self.config.output()
		V            =  self.output(la,lo,z)
		V1           =  self1.output(la,lo,z)
		for i in range(self.nxyz[-1]):
			plt.close()
			plt.figure(figsize=[26,8])
			R = [la.min(),la.max(),lo.min(),lo.max()]
			v  = V[:,:,i]
			v1 = V1[:,:,i]
			laN, loN = v.shape
			midLaN = int(1.2/4*laN)
			midLoN = int(1.2/4*loN)
			mean = v[midLaN:-midLaN,midLoN:-midLoN].mean()
			#print(mean)
			dLa,dLo=getDlaDlo(R)
			v[v<0]=mean
			v1[v1<0]=mean
			#mean = v.mean()
			Per   = (v-mean)/mean
			Per1  = (v1-mean)/mean
			la,lo,per=denseLaLo(la,lo,Per[:,:])
			la,lo,per1=denseLaLo(la,lo,Per1[:,:])
			vmax=max(np.abs(per[midLaN:-midLaN,midLoN:-midLoN]).max(),\
				np.abs(per1[midLaN:-midLaN,midLoN:-midLoN]).max())
			vmin=-vmax
			dLa,dLo=getDlaDlo(R)
			plt.subplot(1,3,1)
			m = mt.genBaseMap(R)
			x,y= m(lo,la)
			plotPlane(m,x,y,per,R,z,mean,vmin,vmax,isFault=True,head=head+'o')
			plt.subplot(1,3,2)
			m = mt.genBaseMap(R)
			x,y= m(lo,la)
			plotPlane(m,x,y,per1,R,z,mean,vmin,vmax,isFault=True,head=head+'1')
			plt.subplot(1,3,3)
			dper = per - per1
			m = mt.genBaseMap(R)
			x,y= m(lo,la)
			plotPlane(m,x,y,dper,R,z,0,-0.02,0.02,isFault=True,head=head+'diff')
			plt.subtitle('depth: %.2f km mean: %.3f'%(self.z[i],mean))
			plt.savefig('%s/%s_%f.jpg'%(resDir,head,self.z[i]),dpi=500)
			#plt.ylim([35,55])
			plt.close()

def plotPlane(m,x,y,per,R,z,mean,vmin=-0.05,vmax=0.05,isFault=True,head='res'):
	dLa,dLo=getDlaDlo(R)
	if isFault:
		for fault in faultL:
			if fault.inR(R):
				fault.plot(m,markersize=0.3)
	m.pcolor(x,y,per,cmap=cmap,shading='auto')
	m.drawcoastlines(linewidth=0.8, linestyle='dashdot', color='k')
	plt.clim(vmin=vmin,vmax=vmax)
	plt.colorbar()
	plt.title('%s %.2f km mean: %.3f'%(head,z,mean))
	if head=='period':
		plt.title('%s %.2f s mean: %.3f'%(head,z,mean))
	dLa,dLo=getDlaDlo(R)
	plotLaLoLine(m,dLa,dLo)

def plotLaLoLine(m,dLa=10,dLo=10):
	parallels = np.arange(0.,90,dLa)
	m.drawparallels(parallels,labels=[False,True,True,False])
	meridians = np.arange(10.,360.,dLo)
	plt.gca().yaxis.set_ticks_position('right')
	m.drawmeridians(meridians,labels=[True,False,False,True])

def getDlaDlo(R):
	DLA = R[1] -R[0]
	DLO = R[3] -R[2]
	if DLA<10:
		dLa = 2
	elif DLA<20:
		dLa = 4
	elif DLA<40:
		dLa = 5
	else:
		dLa = 10

	if DLO<10:
		dLo = 2
	elif DLO<20:
		dLo = 4
	elif DLO<40:
		dLo = 5
	else:
		dLo = 10
	return dLa,dLo

def loadModel(file='models/prem'):
	data = np.loadtxt(file)
	z = data[:,0]
	vs = data[:,2]
	return interpolate.interp1d(z,vs)

def loadModelTK(file = 'models/tk.nc'):
	nc  = Dataset(file,'r')
	z   =  nc.variables['depth'][:]
	la  =  nc.variables['latitude'][:]
	lo  =  nc.variables['longitude'][:]
	vsv =  nc.variables['vsv'][:]
	return z,la,lo,vsv



def denseLaLo(La,Lo,Per,N=500):
	dLa = (La[-1]-La[0])/N
	dLo = (Lo[-1]-Lo[0])/N
	la  = np.arange(La[0],La[-1],dLa)[-1::-1]
	lo  = np.arange(Lo[0],Lo[-1],dLo)
	per = interpolate.interp2d(Lo, La, Per)(lo,la)
	return la, lo, per 



