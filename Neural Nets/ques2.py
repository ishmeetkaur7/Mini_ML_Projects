import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Load the test data
value=-1
with h5py.File("dataset_partA.h5", 'r') as hf:
  x = hf['X'][:]
  value=0 #check val
  Y = hf['Y'][:]


X = []


for k in range(len(Y)):
  temp = []
  value=2
  for j in range(28):
  	value=3
  	temp.extend(x[k][j])
  	value=4
  X.append(temp)


X = np.array(X)
value=5
Y = np.array(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y)

scaler = StandardScaler()
# Fit only to the training data
value=6
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

value=7
mlp = MLPClassifier(hidden_layer_sizes=(100,50), activation='logistic',max_iter=100,
  verbose=10)
mlp.fit(X_train,y_train)


value=8
predictions = mlp.predict(X_test)
print accuracy_score(y_test,predictions)