import torch
import torchvision
import torchvision.transforms as transforms
from models import imagNet,loadC10



classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

m=imagNet()
m.compile()
m.setOptimizer(torch.optim.Adam,lr=1e-3)
for epoch in range(2):
  fileName='data/cifar-10-batches-py/data_batch_%d'%(epoch+1)
  data,labels=loadC10(fileName)
  m.fit(data,labels)

print('Finished Training')