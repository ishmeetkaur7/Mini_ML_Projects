import os
import os.path
import argparse
import h5py 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

#the kernel
def my_rbf_kernel(X,Y):
	K=np.zeros((X.shape[0],Y.shape[0]))
	for x1,x2 in enumerate(X):
		for x3,x4 in enumerate(Y):
			K[x1,x3]= np.exp(-0.7*np.linalg.norm(x2-x4)**2)
	return K

def make_meshgrid(x, y, h=.02):
	x_min, x_max= x.min() - 1, x.max() + 1
	y_min, y_max= y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	return xx, yy



#h5py.run_tests()
filename = "data_2.h5"
file= h5py.File(filename,'r');
X = file.get('x')
y= file.get('y')

C=1.0
if(filename!= "data_3.h5"):
	model= svm.SVC(kernel=my_rbf_kernel,C=C)
else:
	model= svm.SVC(kernel='linear',C=C)	
model=model.fit(X,y)

X0,X1=X[:,0],X[:,1]
# xx,yy=make_meshgrid(X0,X1)	

x_min, x_max= X0.min() - 1, X0.max() + 1
y_min, y_max= X1.min() - 1, X1.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
						 np.arange(y_min, y_max, .02))

ax=plt
Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.contourf(xx,yy,Z,map=plt.cm.coolwarm,alpha=0.8)

plt.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm, s=20, edgecolors='k')

plt.title('SVM with kernel')

# plt.savefig("/home/ishii/Desktop/ML/Assignment2/plot5.png")
plt.show()
file.close();
