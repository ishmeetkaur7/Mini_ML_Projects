import os
import os.path
import argparse
import h5py
from sklearn import svm
import numpy as np 
from sklearn.manifold import TSNE

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
x = file.get('x')
y= file.get('y')

#divide data into test and train.
X= np.array(x)
Y=np.array(y)
train_X,train_Y,test_X,test_Y= dividedata(x,y)

labels= np.max(y)+1
foldsize= len(x)/5
C= [0.001, 0.1, 1.0, 10, 1000]

for c in C:
	correct = 0
	wrong = correct

	finalAns = np.zeros((len(X),labels))

	folditer=0
	while folditer < (len(train_X)):
		Ans = [ [] for i in range(foldsize)]

		labeliter1=0
		
		while labeliter1 < (labels):
			labeliter2=labeliter1+1
			while labeliter2 < (labels):
				
				trainXnew = train_X[folditer]
				trainYnew = np.array(train_Y[folditer])
				testXnew = test_X[folditer]
				testYnew = np.array(test_Y[folditer])

				trainXnew2 = []
				trainYnew2 = []

				i=0
				valuetoprint =0
				while i < (len(trainYnew)):
					if(trainYnew[i]==labeliter1):
						trainXnew2.append(trainXnew[i])
						valuetoprint=1
						trainYnew2.append(valuetoprint)
					elif(trainYnew[i]==labeliter2):
						trainXnew2.append(trainXnew[i])
						valuetoprint=-1
						trainYnew2.append(valuetoprint)
					i+=1

				clf = svm.SVC(kernel = 'linear', C=c)
				valuetoprint=0
				clf.fit(trainXnew2,trainYnew2)


				support_vectors_ = clf.support_vectors_
				valuetoprint=1
				dual_coef_ = clf.dual_coef_
				valuetoprint=-1
				intercept_ = clf.intercept_

				counter=0
				while counter < len(testXnew):
					xnew = testXnew[counter]
					ynew = testYnew[counter]
					ans = 0
					temp=0
					while temp < (len(dual_coef_[0])):
						value = dual_coef_[0][temp]
						value= value * np.exp(-0.7*np.linalg.norm(support_vectors_[temp]-xnew)**2)
						ans += value
						temp+=1
					ans += intercept_
					position=folditer*foldsize + counter
					if(ans > 0):
						finalAns[position][labeliter1] += 1
					else:
						finalAns[position][labeliter2] += 1
					counter+=1
				labeliter2+=1
			labeliter1+=1
		folditer+=1
	predicted = []
	i=0
	while i < (len(finalAns)):
		n = np.argmax(finalAns[i])
		predicted.append(n)
		i+=1

	correct = 0
	wrong = 0
	i=0
	while i < (len(predicted)):
		if(predicted[i] == Y[i]):
			correct += 1
		else:
			wrong += 1
		i+=1

	print(float(correct)/(correct + wrong), c)
