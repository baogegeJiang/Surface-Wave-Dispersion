import numpy as np
from matplotlib import pyplot as plt
'''
a = np.zeros(1000)
b = np.zeros(1000)
a[100:200]=1
b[200:300]=1

#A close
#B far
A = np.fft.fft(a) 
B = np.fft.fft(b)
C = np.fft.ifft(B/(A+1e-13))
CC = np.fft.ifft(np.conj(A)*B)
plt.plot(C)
plt.plot(CC)
plt.show()
'''
from mathFunc import *
from imp import reload

N=50
t = np.arange(1,10000).astype(np.float)
w = np.random.rand(N)*0.02+0.01
theta = np.random.rand(N)*3.14*2
A = np.random.rand(N)*0.8+0.2
a = t*0
b = t*0
for i in range(N):
	a += np.sin((t+0)*w[i]+theta[i])*A[i]
	b += np.sin((t+100+0*w[i]/0.01*50)*w[i]+theta[i])*A[i]
b += np.cos((t+100)*1/200+theta[0])*100
for xcorr in [xcorrFrom0,xcorrAndDe,xcorrAndDeV2,xcorrAndDeV3]:
	c0  = xcorr(a,b)
	c1  = xcorr(b,a)

	plt.plot(np.real(c0))
	plt.plot(np.imag(c0))
	plt.xlim([0,500])
	plt.show()