import os
import os.path
import argparse
import h5py 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing



#h5py.run_tests()
filename = "segmentation.data"

file= open(filename,'r');
lines= file.read().splitlines();
x = []; y=[];
smallarray=[]
for var in lines:
	smallarray=[];
	line=var.split(',');
	for i in range(len(line)):
		if(i!=0):
			smallarray.append(float(line[i]));
		else:
			y.append(line[i]);
	x.append(smallarray);

x = preprocessing.robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True);

for i in range(len(y)):
	if(y[i]=="BRICKFACE"):
		y[i]=0;
	elif(y[i]=="SKY"):
		y[i]=1;
	elif(y[i]=="FOLIAGE"):
		y[i]=2;
	elif(y[i]=="CEMENT"):
		y[i]=3;
	elif(y[i]=="WINDOW"):
		y[i]=4;
	elif(y[i]=="PATH"):
		y[i]=5;
	else:
		y[i]=6;

print len(x);
print len(y);

X_embedded = TSNE(n_components=2,random_state=0).fit_transform(x)	



# i=0
# j=0
# k=0
# colors = []
# # print(Y[0][1])
# # #print(len(X[0]))
# # print(len(Y))
# for i in range(len(x)):
# 	if(y[i]==1):
# 		colors.append(i)
			

plt.scatter(X_embedded[:,0],X_embedded[:,1], c= y)
# #plt.savefig(args.plots_save_dir + "visualisationplotC.png")
# plt.show()
# file.close();

#plt.scatter(x[:,0],x[:,1], c= y)
# plt.savefig("/home/ishii/Desktop/plot5.png")
plt.show()
# file.close();