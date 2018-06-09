import os
import os.path
import argparse
import h5py
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plot
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()


# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

X,Y = load_h5py(args.train_data)
Y_new=[]
Y_predicted=[]
for i in range(len(X)):
	for j in range(len(Y[0])):
		if(Y[i][j]==1):
			Y_new.append(j)
#Y_new= Y_new.reshape(-1,1)
# Preprocess data and split it
#k=5

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []

size=len(X)/5 

for i in range(size):
	x1.append(X[i])  
	y1.append(Y_new[i])
for i in range(size, size*2):
	x2.append(X[i]) 
	y2.append(Y_new[i])
for i in range(size*2, size*3):
	x3.append(X[i]) 
	y3.append(Y_new[i])
for i in range(size*3, size*4):
	x4.append(X[i]) 
	y4.append(Y_new[i])
for i in range(size*4, size*5):
	x5.append(X[i]) 
	y5.append(Y_new[i])

count1=0 


# Train the models

if args.model_name == 'GaussianNB':
	print("1")
	clf = GaussianNB();
	#case1 : test data: x1. 
	train_x= x2+x3+x4+x5
	train_y= y2+y3+y4+y5
	clf.fit(train_x,train_y)
	#predict for x1.
	Y_predicted= clf.predict(x1)
	#compare

	print(accuracy_score(Y_predicted,y1)) 
	#case1 : test data: x1. 
	train_x= x1+x3+x4+x5
	train_y= y1+y3+y4+y5
	clf.fit(train_x,train_y)
	#predict for x1.
	Y_predicted= clf.predict(x2)
	#compare
	print(accuracy_score(Y_predicted,y2));
	#case1 : test data: x1. 
	train_x= x2+x1+x4+x5
	train_y= y2+y1+y4+y5
	clf.fit(train_x,train_y)
	#predict for x1.
	Y_predicted= clf.predict(x3)
	#compare
	print(accuracy_score(Y_predicted,y3))
	#case1 : test data: x1. 
	train_x= x2+x3+x1+x5
	train_y= y2+y3+y1+y5
	clf.fit(train_x,train_y)
	#predict for x1.
	Y_predicted= clf.predict(x4)
	#compare
	print(accuracy_score(Y_predicted,y4))
	#case1 : test data: x1. 
	train_x= x2+x3+x4+x1
	train_y= y2+y3+y4+y1
	clf.fit(train_x,train_y)
	#predict for x1.
	Y_predicted= clf.predict(x5)
	#compare
	print(accuracy_score(Y_predicted,y5))
	#save model
	joblib.dump(clf,args.weights_path+ "GausseanNBBestModel.pkl")
elif args.model_name == 'LogisticRegression':
	print("2")
	array2=[]
	array1=[]
	array0=[]
	penalty= ['l1','l2']
	c=[1.0,12.0,50.0]
	max_iter=[100,200]
	verbose=[0]
	for i in range(len(penalty)):
		for j in range(len(c)):
			for k in range(len(max_iter)):
				for l in range(len(verbose)):
					clf= linear_model.LogisticRegression(penalty=penalty[i], C= c[j], max_iter= max_iter[k], verbose = verbose[l])
					#case 1. 
					train_x= x2+x3+x4+x5
					train_y= y2+y3+y4+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x1)
					a1=accuracy_score(Y_predicted,y1)
					print (a1)
					#case 2. 
					train_x= x1+x3+x4+x5
					train_y= y1+y3+y4+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x2)
					a2=accuracy_score(Y_predicted,y2)
					print(a2)
					#case 3. 
					train_x= x2+x1+x4+x5
					train_y= y2+y1+y4+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x3)
					a3=accuracy_score(Y_predicted,y3)
					print(a3)
					#case 4. 
					train_x= x2+x3+x1+x5
					train_y= y2+y3+y1+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x4)
					a4=accuracy_score(Y_predicted,y4)
					print(a4)
					#case 5. 
					train_x= x2+x3+x4+x1
					train_y= y2+y3+y4+y1
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x5)
					a5=accuracy_score(Y_predicted,y5)
					print(a5)
					mean= (a1+a2+a3+a4+a5)/5
					array2.append(mean*100);
					array1.append([penalty[i], c[j], max_iter[k], verbose[l]])
					print(array2)
					print(array1)
	x = len(array1)
	for m in range(x):
		array0.append(m)
	plot.bar(array0,array2);
	plot.xticks(array0, array1, rotation = 'vertical')
	#plot.show();
	#plot.savefig(args.plots_save_dir + "accuracyParametersLRpartB.png")
	maximum= max(array2);
	print(maximum)
	for i in range(x):
		print(array2[i])
		if(array2[i]==maximum):
			attr1=array1[i][0]
			attr2=array1[i][1]
			attr3=array1[i][2]
			attr4=array1[i][3]
			model = linear_model.LogisticRegression(penalty=attr1, C= attr2, max_iter= attr3, verbose = attr4)
			joblib.dump(model,args.weights_path+ "LRBestModel.pkl")
			break;
		print("here")
elif args.model_name == 'DecisionTreeClassifier':
	print("3")
	array2=[]
	array1=[]
	array0=[]
	max_depth=[1,20,3000]
	min_samples_split=[20,4,6]
	min_samples_leaf=[1,200,30]
	max_features=[1,2]
	for i in range(len(max_depth)):
		for j in range(len(min_samples_split)):
			for k in range(len(min_samples_leaf)):
				for l in range(len(max_features)):
					clf= DecisionTreeClassifier(max_depth=max_depth[i], min_samples_split= min_samples_split[j], min_samples_leaf= min_samples_leaf[k], max_features = max_features[l])
					#case 1. 
					train_x= x2+x3+x4+x5
					train_y= y2+y3+y4+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x1)
					a1=accuracy_score(Y_predicted,y1)
					print (a1)
					#case 2. 
					train_x= x1+x3+x4+x5
					train_y= y1+y3+y4+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x2)
					a2=accuracy_score(Y_predicted,y2)
					print(a2)
					#case 3. 
					train_x= x2+x1+x4+x5
					train_y= y2+y1+y4+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x3)
					a3=accuracy_score(Y_predicted,y3)
					print(a3)
					#case 4. 
					train_x= x2+x3+x1+x5
					train_y= y2+y3+y1+y5
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x4)
					a4=accuracy_score(Y_predicted,y4)
					print(a4)
					#case 5. 
					train_x= x2+x3+x4+x1
					train_y= y2+y3+y4+y1
					clf.fit(train_x, train_y)
					Y_predicted= clf.predict(x5)
					a5=accuracy_score(Y_predicted,y5)
					print(a5)
					mean= (a1+a2+a3+a4+a5)/5
					array2.append(mean*100);
					array1.append([max_depth[i], min_samples_split[j], min_samples_leaf[k], max_features[l]])
					print(array2)
					print(array1)
	x = len(array1)
	for m in range(x):
		array0.append(m)
	plot.bar(array0,array2);
	plot.xticks(array0, array1, rotation = 'vertical')
	maximum= max(array2);
	print(maximum)
	for i in range(x):
		print(array2[i])
		if(array2[i]==maximum):
			attr1=array1[i][0]
			attr2=array1[i][1]
			attr3=array1[i][2]
			attr4=array1[i][3]
			model = DecisionTreeClassifier(max_depth=attr1, min_samples_split= attr2, min_samples_leaf= attr3, max_features = attr4)
			joblib.dump(model,args.weights_path+ "DTBestModel.pkl")
			break;
	#plot.show();
	#plot.savefig(args.plots_save_dir + "accuracyParametersDTpartC.png")

	# model = DecisionTreeClassifier(  ...  )

	# save the best model and print the results
else:
	raise Exception("Invald Model name")
