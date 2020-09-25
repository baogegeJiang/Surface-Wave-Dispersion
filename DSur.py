import numpy as np
import os 

class config:
	def __init__(self,para,name='ds'):
		self.name = name
		config.keyList = ['dataFile', 'nxyz', 'lalo', 'dlalo', 'maxN','damp',\
		'sablayers','minmaxV', 'sparsity','kmaxRc','rcPerid','kmaxRg','rgPeriod',\
		'kmaxLc','lcPeriod','kmaxLg','lgPeriod','isSyn','noiselevel','threshold',\
		'vnn']
		self.para = {'dataFile':name+'in', 'nxyz':[18,18,9], 'lalo':[130,30],\
		 'dlalo':[0.01,0.01], 'maxN':[20],'damp':[4.0,1.0],\
		'sablayers':3,'minmaxV':[2,7], 'sparsity':0.2,\
		'kmaxRc':10,'rcPerid':np.arange(1,11).tolist(),'kmaxRg':0,'rgPeriod':[],\
		'kmaxLc':0,'lcPeriod':[],'kmaxLg':0,'lgPeriod':[],'isSyn':0,'noiselevel':0.02,'threshold':0.05,\
		'vnn':[0,100,50]}

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
		with open(self.runPath+config.name, 'w+') as f:
				for key in config.keyList:
					value = config.para[key]
					if isinstance(value,list):
						valueNew = ''
						for v in value:
							valueNew += v+' '
						value = value
					f.write(value+'\n')
