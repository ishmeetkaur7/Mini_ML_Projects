import os
import os.path
import argparse
import h5py
from sklearn import svm
import numpy as np 
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC

filename= "data_4.h5"

def dividedata(X,Y):
	test_X = []
	test_Y = []
	train_X = []
	train_Y = []	
	temp_X = X
	temp_Y = np.array(Y)
	y_values = []
	i=0
	while (i<len(temp_Y)):
		y_values.append(temp_Y[i])
		i+=1
	y_values = np.array(y_values)
	temp_Y = y_values
	temp_X = np.split(X,5)
	temp_Y = np.split(temp_Y,5)
	temp_X =  (temp_X)
	temp_Y =  (temp_Y)

	for i in range(5):
		test_X.append( (temp_X[i]))
		test_Y.append( (temp_Y[i]))

	i=0
	while i<5:
		temp1 = []
		temp2 = []
		j=0
		while j<5: 
			if(i!=j):
				temp1 = temp1 + list(test_X[j])
				temp2 = temp2 + list(test_Y[j])
			j=j+1
		train_X.append(temp1)
		train_Y.append(temp2)
		i=i+1


	return train_X,train_Y,test_X,test_Y



#load data
file= h5py.File(filename,'r');
X = file.get('x')
Y= file.get('y')

#divide data into test and train.

train_X,train_Y,test_X,test_Y= dividedata(X,Y)

labels= np.max(Y)+1
foldsize= len(X)/5
C= [0.001, 0.1, 1.0, 10, 1000]

for c in C:
	correct = 0
	wrong = correct

	folditer=0
	while folditer < (len(train_X)):

		Ans = [[] for i in range(foldsize)]

		labeliter=0
		while labeliter < labels:
			trainXnew = train_X[folditer]
			trainYnew = np.array(train_Y[folditer])
			testXnew = test_X[folditer]
			testYnew = np.array(test_Y[folditer])
			value1 = len(trainYnew)
			i=0
			while i < (value1):
				if(trainYnew[i]==labeliter):
					trainYnew[i]=1
				else:
					trainYnew[i]=-1
				i+=1

			clf = svm.SVC(kernel='linear', C=c)
			vauetoprint=0
			clf.fit(trainXnew,trainYnew)

			support_vectors = clf.support_vectors_
			vauetoprint=0
			dual_coef = clf.dual_coef_
			vauetoprint=1
			intercept = clf.intercept_


			value2=len(testXnew)
			counter=0
			while counter <(value2):
				xnew = testXnew[counter]
				ynew = testYnew[counter]
				ans = 0
				temp=0
				while temp < (len(dual_coef[0])):
					value = dual_coef[0][temp] 
					#value= value * np.dot(support_vectors[temp],xnew)
					value= value * np.exp(-0.7*np.linalg.norm(support_vectors[temp]-xnew)**2)
					ans += value
					temp+=1
				ans += intercept
				Ans[counter].append(ans)
				counter+=1
			labeliter+=1


		predicted = []
		i=0
		while i <(len(Ans)):
			n = np.argmax(Ans[i])
			predicted.append(n)
			i+=1

		i=0
		while i< (len(predicted)):
			if(predicted[i] == testYnew[i]):
				correct += 1
			else:
				wrong += 1
			i+=1
		folditer+=1

	print(float(correct)/(correct + wrong), c)
