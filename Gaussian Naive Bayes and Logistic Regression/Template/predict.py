import os
import os.path
import argparse
import h5py
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()
args.test_data="Data/part_C_train.h5"

# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

x,y=load_h5py(args.test_data)

if args.model_name == 'GaussianNB':
	clf = joblib.load(args.weights_path)
	prediction = clf.predict(x)
	print(prediction);
	f=open("test.txt","a+")
	f.write(prediction);
	f.close();
	#testing the prediction on the entire data.
elif args.model_name == 'LogisticRegression':
	clf = joblib.load(args.weights_path)
	prediction = clf.predict(args.test_data)
	print(prediction);
	f=open("test.txt","a+")
	f.write(prediction);
	f.close();
elif args.model_name == 'DecisionTreeClassifier':
	# load the model

	# model = DecisionTreeClassifier(  ...  )

	# save the predictions in a text file with the predicted clasdIDs , one in a new line 
	clf = joblib.load(args.weights_path)
	prediction = clf.predict(args.test_data)
	print(prediction);
	f=open("test.txt","a+")
	f.write(prediction);
	f.close();
else:
	raise Exception("Invald Model name")
