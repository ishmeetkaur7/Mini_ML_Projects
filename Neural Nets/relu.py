import os
import os.path
import numpy as np
from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt
import h5py
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import tensorflow as tf


(X_train, y), (X_test, y_test) = mnist.load_data()

value=0
X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
 
value=1
#Make the value floats in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
value=2
X_train /= 255
# X_test /= 255




# with h5py.File("dataset_partA.h5", 'r') as hf:
#   x = hf['X'][:]
#   Y = hf['Y'][:]


# x = np.array(x)
# Y = np.array(Y)

# for k in range(len(Y)):
#   if(Y[k] == 7):
#     Y[k] = 0
#   else:
#     Y[k] = 1


# X = []
# temp = []
# temp= np.array(temp)

# for k in range(len(Y)):
#   temp = []
#   for j in range(28):
#     temp.extend(x[k][j])
#   X.append(temp)


X = np.array(X_train)
# y = np.array(Y)

# ----------------------------------------

#  preprocessing step...change it later

# X = X.astype(float)
# mean1 = np.mean(X, axis = 0)
# std1 = np.std(X, axis = 0)
# std1 = std1.astype(float)
# mean1 = mean1.astype(float)
# # print std1
# X = (X - mean1) 
# X = X.astype(float)

# for s in range(len(std1)):
#   for z in range(len(X)):
#     if(std1[s] != 0):
#       X[z][s] = X[z][s] / std1[s] 

# ------------------------------------------

D = len(X[0])
w_array = []
# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
K = 10
h = []
b_array = []

n1 = int(input("input: "))

# h contains size of hidden layer
for k1 in range(0,n1,1):  
  temp = int(input())
  h.insert(k1,temp)

# --------------------------------------------
# initialise the parameters
temp1 = []
temp1.append(D)
temp1.extend(h)
temp1.append(K)
count2 = 0

for i1 in range(0,len(temp1)-1):
	count2 = count2 + 1
	hello=5
	w_array.append(0.01 * np.random.randn(temp1[i1],temp1[i1+1]))
	hello=10
	b_array.append(np.zeros((1,temp1[i1+1])))

hello=6
num_examples = X.shape[0]
for i in xrange(100):
	hidden_layer = []
	# print "2"
	# print w_array[0]
	# ---------------------------
	# evaluate hidden layers..forward prop

	temp1 = []
	temp1.append(X)
	temp1.extend(w_array)
	count = 0

	for i2 in range(len(h)):
		count = count + 1
		if(i2 == 0):
			hidden_layer.append(np.maximum(0, np.dot(X, w_array[i2]) + b_array[i2]))
		else:
			hidden_layer.append(np.maximum(0, np.dot(hidden_layer[i2-1], w_array[i2]) + b_array[i2]))

	scores = np.dot(hidden_layer[count-1], w_array[count2-1]) + b_array[count2-1]

	# print "3"
	# print w_array[0]
	# ---------------------------
	#  change the below portion

	# compute the class probabilities
	hello=-3
	exp_scores = np.exp(scores)
	hello=-1
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
	hello=-4
	# compute the loss: average cross-entropy loss and regularization
	corect_logprobs = -np.log(probs[range(num_examples),y])
	hello=-2
	data_loss = np.sum(corect_logprobs)/num_examples
	reg_loss = 0

	hello=0
	for k3 in range(len(w_array)):
		hello=678
		reg_loss = reg_loss + (0.5*reg*np.sum(w_array[k3]*w_array[k3]))


	hello=1
	loss = data_loss + reg_loss
	print "iteration %d: loss %f" % (i, loss)
	
	# compute the gradient on scores
	dscores = probs
	dscores[range(num_examples),y] -= 1
	dscores /= num_examples
	# print "5"
	# print w_array[0]
	# ------------------------------






	# dw_array = w_array
	# db_array = b_array
	# dhidden_array = hidden_layer

	
	
	# dw_array = np.astype(float)
	# db_array = np.astype(float)
	# dhidden_array = np.astype(float)

	# np.copyto(dw_array,w_array,casting= 'unsafe')
	# np.copyto(db_array,b_array,casting= 'unsafe')
	# np.copyto(dhidden_array,hidden_layer,casting= 'unsafe')

	# print "6"
	# print w_array[0]
	
	l1 = len(w_array)
	l2 = len(b_array)
	l3 = len(hidden_layer)
	
	dw_array = [[] for z2 in range(l1)]
	db_array = [[] for z2 in range(l2)]
	dhidden_array = [[] for z2 in range(l3)]


	dw_array[l1-1] = np.dot(hidden_layer[len(hidden_layer)-1].T, dscores)
	db_array[l2-1] = np.sum(dscores, axis=0, keepdims=True)

	# print "7"
	# print w_array[0]
	
	dhidden_array[l3-1] = np.dot(dscores, w_array[len(w_array)-1].T)
	dhidden_array[l3-1][hidden_layer[l3-1] <= 0] = 0

	# print "8"
	# print w_array[0]

	dw_array[l1-2] = np.dot(hidden_layer[l3-2].T, dhidden_array[l3-1])
	db_array[l2-2] = np.sum(dhidden_array[l3-1], axis=0, keepdims=True)

	# print "9"
	# print w_array[0]
	
	for h1 in range(0,l3-2):
		dhidden_array[l3-2-h1] = np.dot(dhidden_array[l3-1-h1], w_array[l1-h1-2].T)
		dhidden_array[l3-2-h1][hidden_layer[l3-2-h1] <= 0] = 0
		dw_array[l2-3-h1] = np.dot(hidden_layer[l3-3-h1].T, dhidden_array[l3-2-h1])
		db_array[l2-3-h1] = np.sum(dhidden_array[l3-2-h1], axis=0, keepdims=True)

	# print "10"
	# print w_array[0]

	dhidden_array[0] = np.dot(dhidden_array[1], w_array[1].T)
	# print "11"
	# print w_array[0]
	dhidden_array[0][hidden_layer[0] <= 0] = 0
	# print "12"
	# print w_array[0]
	
	dw_array[0] = np.dot(X.T, dhidden_array[0])
	
	# print "11a"
	# print w_array[0]
	db_array[0] = np.sum(dhidden_array[0], axis=0, keepdims=True)

	# print len(dw_array)
	# print "11"
	# print w_array[0]
	# print len(dw_array)
	# print dw_array[1]

	# dw_array = np.flip(dw_array,0)
	# w_array = np.flip(w_array,0)
	# b_array = np.flip(b_array,0)

	# for jk in range(len(dw_array)):
	# 	dw_array[jk] += reg * w_array[len(w_array)-(1+jk)]

 #  	for jk in range(len(w_array)):
 #  		w_array[jk] += -step_size * dw_array[len(dw_array)-(1+jk)]
  

	for h1 in range(len(dw_array)):
		dw_array[h1] += reg * w_array[h1]
		

	for h1 in range(len(w_array)):
		w_array[h1] += -step_size * dw_array[h1]
		b_array[h1] += -step_size * db_array[h1]
		

	# print w_array
	# print dw_array

	





	# dw_array = dw_array.reverse()
	# w_array = w_array.reverse()
	# b_array = b_array.reverse()
	
	# dw_array.append(np.dot(hidden_layer[len(hidden_layer)-1].T, dscores))
	# db_array.append(np.sum(dscores, axis=0, keepdims=True))
	# # next backprop into hidden layer
	# dhidden_array.append(np.dot(dscores, w_array[len(w_array)-1].T))
	# # backprop the ReLU non-linearity
	# dhidden_array[0][hidden_layer[len(hidden_layer)-1] <= 0] = 0
	# # finally into W,b
	# dw_array.append(np.dot(hidden_layer[len(hidden_layer)-2].T, dhidden_array[0]))
	# db_array.append(np.sum(dhidden_array[0], axis=0, keepdims=True))

	# for k4 in range(len(hidden_layer)-2):
	# 	dhidden_array.append(np.dot(dhidden_array[k4], w_array[len(w_array)-1-(k4+1)].T))
	# 	# change the one i have appended jus above
	# 	dhidden_array[len(dhidden_array)-1][hidden_layer[len(hidden_layer)-1-(k4+1)] <= 0] = 0
	# 	dw_array.append(np.dot(hidden_layer[len(hidden_layer)-(3+k4)].T, dhidden_array[len(dhidden_array)-1]))
	# 	db_array.append(np.sum(dhidden_array[len(dhidden_array)-1], axis=0, keepdims=True))

	# # next backprop into hidden layer
	# # finally into W,b
	# dhidden_array.append(np.dot(dhidden_array[len(dhidden_array)-1], w_array[1].T))
	# dhidden_array[len(dhidden_array)-1][hidden_layer[0] <= 0] = 0
	# dw_array.append(np.dot(X.T, dhidden_array[len(dhidden_array)-1]))
	# db_array.append(np.sum(dhidden_array[len(dhidden_array)-1], axis=0, keepdims=True))
	

	# jk = 0
	

	# for jk in range(len(dw_array)):
	# 	dw_array[jk] += reg * w_array[len(w_array)-(1+jk)]

 #  	for jk in range(len(w_array)):
 #  		w_array[jk] += -step_size * dw_array[len(dw_array)-(1+jk)]
 #  		b_array[jk] += -step_size * db_array[len(db_array)-(1+jk)]	

