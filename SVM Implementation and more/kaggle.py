import json 
import numpy
import numpy as np 
from sklearn import svm
import csv
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


data = json.load(open('train.json'))
x1=[]
y1=[]
x_array=[]
y_array=[]
x_test=[]
y_test=[]
testvar=0

for d in data:
    string=""
    value= len(d['X'])
    for i in range(value):
        if(i!=value-1):
            string = string+ str(d['X'][i])
            string= string  + " "
            testvar=1
        else:
            string = string + str(d['X'][i])
    x1.append(string)
    y1.append(d['Y'])

x=numpy.array(x1)
y = numpy.array(y1)
print len(x)

vectorizer = TfidfVectorizer(norm='l2',min_df=1, use_idf=True, smooth_idf=True, sublinear_tf=True,
                        ngram_range=(0,3),token_pattern=r"\b\w+\b",max_df=1.0,strip_accents='unicode',
                        max_features=None, analyzer="word", binary=True, decode_error='strict',stop_words='english'
                        )


xx = vectorizer.fit_transform(x)
testvar=2
model= svm.LinearSVC(random_state=0,C=0.5)
testvar=3
model = model.fit(xx, y)
testvar=4
res = model.predict(xx)



while i<len(res):
    if(y[i]!=res[i]):
        print "error"
    else:
        x_array.append(x[i])
        y_array.append(y[i])
    i+=1
xx = vectorizer.fit_transform(x_array)
testvar=0
model= svm.LinearSVC(random_state=0,C=0.5)
testvar=1
model = model.fit(xx, y_array)
testvar=2



data = json.load(open('test.json'))


for d in data:
    string=""
    value= len(d['X'])
    for i in range(value):
        if(i!=value-1):
            string = string+ str(d['X'][i])
            string= string  + " "
            testvar=1
        else:
            string = string + str(d['X'][i])
    x_test.append(string)

x=numpy.array(x_test)
dr = vectorizer.transform(x_test)

y_test=model.predict(dr)

with open('hope3.csv', 'wb') as file:
    file.write("Id,Expected"+"\n")
    testvar=5
    for row in range(0,472427):
        p =  str(row+1)+ "," + str(y_test[row])
        print p
        if(row!=472426):
            file.write(p + "\n")
        else:
            file.write(p)