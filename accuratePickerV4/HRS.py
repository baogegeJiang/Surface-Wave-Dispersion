import obspy 
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
filesL=[glob('*/XA.RD11[1-5].00.DN%s.20190831.SAC'%comp) for comp in 'ENZ']
sacsL=[[obspy.read(file)[0] for file in files] for files in filesL]
#for sacs in sacsL:
#	for sac in sacs:
#		sac.filter('bandpass',corners=2,freqmin=0.05,freqmax=240)

index=4450000
indexL=np.arange(-500*12,500*12)+index
timeL=indexL/500
delta=1/500
fL=np.arange(len(indexL))/len(indexL)*1/delta
fmax=0.5/delta
for i in range(3):
	plt.figure(1)
	#plt.clear()
	plt.figure(2)
	plt.figure(3)
	#plt.clear()
	sacs=sacsL[i]
	comp='ENZ'[i]
	for j in range(5):
		sac=sacs[j]
		data=sac.data[indexL]
		data=data-data.mean()
		data/=5e3#1e5
		plt.figure(1)
		plt.plot(timeL,data+j,'k',linewidth=0.3)
		plt.xlabel('t/s')
		plt.ylabel('sta')
		plt.figure(2)
		plt.subplot(5,1,5-j)
		if j>0:
			plt.xticks([])
		if j==0:
			plt.xlabel('f/Hz')
		plt.plot(fL,np.abs(np.fft.fft(data)),'k',linewidth=0.3)
		plt.xlim([0,fmax])
		plt.figure(3)
		plt.subplot(5,1,5-j)
		if j>0:
			plt.xticks([])
		if j==0:
			plt.xlabel('f/Hz')
		plt.plot(fL,np.abs(np.fft.fft(data)),'k',linewidth=0.3)
		plt.xlim([0,30])
	plt.figure(1)
	plt.savefig('%s_waveform.jpg'%('ENZ'[i]),dpi=300)
	plt.close()
	plt.figure(2)
	plt.savefig('%s_spec.jpg'%('ENZ'[i]),dpi=300)
	plt.close()
	plt.figure(3)
	plt.savefig('%s_spec_zoom.jpg'%('ENZ'[i]),dpi=300)
	plt.close()