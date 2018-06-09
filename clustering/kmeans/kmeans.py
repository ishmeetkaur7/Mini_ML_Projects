from random import uniform, shuffle
import math
# import sys
# import os
# import os.path
#from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
from sklearn import metrics
# from sklearn.cluster import KMeans


#step1: read data

filename = "iris.data"

file= open(filename,'r');
lines= file.read().splitlines();
x = []; y=[];
smallarray=[]
for var in lines:
	smallarray=[];
	line=var.split(',');
	for i in range(len(line)):
		if(i!=len(line)-1):
			smallarray.append(float(line[i]));
		else:
			y.append(line[i]);
	x.append(smallarray);

#shuffle(x);
		
for i in range(len(y)):
	if(y[i]=="Iris-setosa"):
		y[i]=2;
	elif(y[i]=="Iris-versicolor"):
		y[i]=1;
	else:
		y[i]=0;

# print len(x);
# print len(y);

#to find clusters, we'll first calculate means, then divide each item to a cluster.

# kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
# print kmeans.labels_

n= len(x[0]); #no of attributes
min=[]; max=[]; #min max for each col.
for i in range(n):
	min.append( 9999999999999);
	max.append(-9999999999999);
for i in range(len(x)):
	for j in range(n):
		if(x[i][j]>max[j]):
			max[j]=x[i][j];
		if(x[i][j]<min[j]):
			min[j]=x[i][j];

# print min
# print max;

K=3;

#INITIALISE K MEANS AS RANDOM.
means=[];
for i in range(K):
	temp=[]
	for j in range(n):
		temp.append(0);
	means.append(temp);

for i in range(len(means)):
	for j in range(len(means[0])):
		means[i][j]=uniform(min[j]+1,max[j]-1);

# print means;

#FINISH MEANS: CLASSIFY ITEMS INTO CLUSTERS AND UPDATE MEANS...
clustersizes=[]
belongsto=[]
for i in range(len(means)):
	clustersizes.append(0);

for i in range(len(x)):
	belongsto.append(0);

maxiter=3000;
change=0;
arr1=[]
for i in range(maxiter):
	#if no change then end
	print i
	change=0
	arr1.append(sum);
	sum=0

	for j in range(len(x)):
		#classify x[j].
		tempmin=99999999999999; 
		ans=-1; 
		for k in range(len(means)):
			#find distance
			dist=0;
			for l in range(len(x[j])):
				dist+=math.pow(x[j][l]-means[k][l],2)
			dist=math.sqrt(dist);
			
			if(dist<tempmin):
				tempmin=dist;
				sum+=dist
				ans=k;
		
		clustersizes[ans]+=1;
		#means[ans]= UpdateMean(clusterSizes[index],means[index],item);
		#UPDATE MEANS
		for k in range(len(means[ans])):
			m = means[ans][k];
			n= clustersizes[ans]
			m = (m*(n-1)+x[j][k])/float(n);
			means[ans][k]=round(m,3);

		if(ans != belongsto[j]):
			change=1;
		belongsto[j]=ans;
	print sum
	if(change==0):
		break;
	# break;

print("Labels are as follows:\n")
for i in range(len(x)):
        print belongsto[i]


clusters=[]

for i in range(len(means)):
	clusters.append([]);


for i in range(len(x)):
	tempmin=99999999999999; 
	ans=-1; 
	for k in range(len(means)):
		#find distance
		dist=0;
		for l in range(len(x[i])):
			dist+=math.pow(x[i][l]-means[k][l],2)
		dist=math.sqrt(dist);
		if(dist<tempmin):
			tempmin=dist;
			ans=k;
	clusters[ans].append(x[i])

print("Items in clusters:\n")
for i in range(len(clusters)):
	print clusters[i];
	print "\n"

# print arr1
# arr2=[]
# for i in range(3000):
# 	arr2.append(i);
print metrics.adjusted_rand_score(y, belongsto)
print metrics.normalized_mutual_info_score(y, belongsto)
print metrics.adjusted_mutual_info_score(y, belongsto)

# X_embedded = TSNE(n_components=2,random_state=0).fit_transform(x)	

# plt.scatter(X_embedded[:,0],X_embedded[:,1], c= belongsto)

# plt.show()
