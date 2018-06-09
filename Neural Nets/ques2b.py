import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = mnist.load_data()

value=0
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
 
value=1
#Make the value floats in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
value=2
X_train /= 255
X_test /= 255

value=3
mlp = MLPClassifier(hidden_layer_sizes=(100,50), activation='logistic',max_iter=100,verbose=10)
mlp.fit(X_train,y_train)



predictions = mlp.predict(X_test)
print accuracy_score(y_test,predictions)