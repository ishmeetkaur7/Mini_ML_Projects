import os
import os.path
import argparse
import h5py 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



#h5py.run_tests()
filename = "/home/ishii/Desktop/ML/Assignment2/data_5.h5"

file= h5py.File(filename,'r');

x = file.get('x')
#x_embedded = TSNE(n_components=2,random_state=0).fit_transform(x)


print(file.name)
#print(x_embedded.shape)
#print(file.keys())
y= file.get('y')
			

plt.scatter(x[:,0],x[:,1], c= y)
plt.savefig("/home/ishii/Desktop/plot5.png")
#plt.show()
file.close();