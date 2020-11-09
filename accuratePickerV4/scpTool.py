import os
from glob import glob
def get(remotePath,localPath,cmd\
	='scp -p 120 jiangyr@162.105.155.181:%s %s'):
	os.system(cmd%(remotePath,localPath))
	return glob(localPath+'/*[mENZCcenz]')

def put(localPath,remotePath,cmd\
	='scp -p 120 %s jiangyr@162.105.155.181:%s'):
	os.system(cmd%(localPath,remotePath))