import numpy as np
import os 
from scipy import interpolate
from matplotlib import pyplot as plt
from distaz import DistAz
##程序中又控制的大bug
##必须按顺序(period source)
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
		'c','kmaxRc','rcPerid']
		self.para = {'dataFile':name+'in', 'nxyz':[18,18,9], 'lalo':[130,30],\
		 'dlalo':[0.01,0.01], 'maxN':[20],'damp':[4.0,1.0],\
		'sablayers':3,'minmaxV':[2,7],'maxIT':10, 'sparsity':0.8,\
		'kmaxRc':10,'rcPerid':np.arange(1,11).tolist(),'kmaxRg':0,'rgPeriod':[],\
		'kmaxLc':0,'lcPeriod':[],'kmaxLg':0,'lgPeriod':[],'isSyn':0,'noiselevel':0.02,'threshold':0.05,\
		'vnn':[0,100,50],'iso':'F','c':'c','smoothDV':20,'smoothG':40,'Damp':0,}
		self.para.update(para)
	def output(self):
		nxyz = self.para['nxyz']
		la = self.para['lalo'][0]-np.arange(nxyz[0])*self.para['dlalo'][0]
		lo = self.para['lalo'][1]+np.arange(nxyz[1])*self.para['dlalo'][1]
		return nxyz,la,lo,self.z
	def findLaLo(self,la,lo):
		laNew = -int((self.para['lalo'][0]-la)/self.para['dlalo'][0])*self.para['dlalo'][0]+self.para['lalo'][0]+0.001
		loNew =  int((lo-self.para['lalo'][1])/self.para['dlalo'][1])*self.para['dlalo'][1]+self.para['lalo'][1]+0.001
		return la,lo
class DS:
	"""docstring for ClassName"""
	def __init__(self,runPath='DS/',config=config()):
		self.runPath = runPath
		if not os.path.exists(runPath):
			os.mkdir(runPath)
		self.config = config
	def writeInput(self):
		'''
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
	def writeData(self,fvLL,indexL,stations,waveType=0):
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
				for i in range(staN):
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
					f.write('# %.3f %.3f %d 2 0\n'%(la,lo,j+1))
					for k in range(len(indexL[i])):
						kk = indexL[i][k]
						if vL[k]>2:
							la,lo=self.config.findLaLo(stations[kk]['la'],\
							stations[kk]['lo'])
							f.write('%.3f %.3f %f\n'%(la,lo,vL[k]))
	def writeMod(self,mod=[]):
		nx,ny,nz=self.config.para['nxyz']
		dep1=np.array(self.config.z)
		#dep1=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.3,1.5,1.8,2.1,2.5])
		nz=len(dep1)
		#ends
		vs1=np.zeros(nz)
		model = loadModel()
		if len(mod) ==0:
			mod=np.zeros((nx,ny,nz))
			for k in range(nz):
				for j in range(ny):
					for i in range(nx):
					  mod[i,j,k] = model(dep1[k])
		else:
			mod = model
		 
		with open(self.runPath+'/MOD','w') as fp:
		    for i in range(nz):
		        fp.write('%9.1f' % (dep1[i]))
		    fp.write('\n')
		    for k in range(nz):
		        for j in range(ny):
		            for i in range(nx):
		                fp.write('%9.3f' % (mod[i,j,k]))
		            fp.write('\n')
		for i in range(nz):
		  print (dep1[i]),
	def test(self,fvLL,indexL,stations):
		self.writeData(fvLL,indexL,stations)
		self.writeInput()
		self.writeMod()
	def loadRes(self,it=-1):
		if it<0:
			filename = '%s/Gc_Gs_model.inv'%(self.runPath)
			
		else:
			filename = '%s/%sMeasure.dat.iter0%d'%(self.runPath,self.config.name,it)
		self.vModel = Model(filename, self.config)
		return self.vModel
	def plotByZ(self):
		self.vModel.plotByZ(self.runPath)


class Model:
	def __init__(self,file,config,runPath='DS/'):
		data0 = np.loadtxt(runPath+'/MOD',skiprows=1)
		data = np.loadtxt(file)
		nxyz,la,lo,z = config.output()
		self.nxyz = nxyz
		self.la   = la
		self.lo   = lo
		self.z    = np.array(z)
		la = la.tolist()
		lo = lo.tolist()
		#z  = z.tolist()
		self.v = data0.reshape(nxyz)*0-1
		for i in range(data.shape[0]):
			Lo = data[i,0]
			La = data[i,1]
			Z  = data[i,2]
			v  = data[i,3]
			i0 = np.abs(la-La).argmin()
			i1 = np.abs(lo-Lo).argmin()
			i2 = np.abs(z-Z).argmin()
			self.v[i0,i1,i2]=v
		for i in range(nxyz[-1]):
			self.v[self.v[:,:,i]<0,i] = self.v[self.v[:,:,i]>0,i].mean()

		
	def plotByZ(self,runPath='DS'):
		resDir = runPath+'/'+'plot/'
		if not os.path.exists(resDir):
			os.mkdir(resDir)
		for i in range(self.nxyz[-1]):
			plt.close()
			plt.pcolor(self.lo,self.la,-self.v[:,:,i],cmap='bwr')
			plt.colorbar()
			plt.title('%f.jpg'%self.z[i])
			plt.savefig('%s/%f.jpg'%(resDir,self.z[i]),dpi=200)
			plt.ylim([35,55])
			plt.close()

def loadModel(file='models/prem'):
	data = np.loadtxt(file)
	z = data[:,0]
	vs = data[:,2]
	return interpolate.interp1d(z,vs)