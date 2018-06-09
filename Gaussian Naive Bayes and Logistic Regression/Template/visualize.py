import os
import os.path
import argparse
import h5py 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()



#h5py.run_tests()
filename = args.data

file= h5py.File(filename,'r');

X = file.get('X')
X_embedded = TSNE(n_components=2,random_state=0).fit_transform(X)


print(file.name)
print(X_embedded.shape)
#print(file.keys())
Y= file.get('Y')

i=0
j=0
k=0
colors = []
# print(Y[0][1])
#print(len(X[0]))
print(len(Y))
for i in range(len(X)):
	k=k+1
	for j in range(len(Y[0])):
		k+=2
		if(Y[i][j]==1):
			k+=3
			colors.append(j)
			

plt.scatter(X_embedded[:,0],X_embedded[:,1], c= colors)
plt.savefig(args.plots_save_dir + "visualisationplotC.png")
file.close();