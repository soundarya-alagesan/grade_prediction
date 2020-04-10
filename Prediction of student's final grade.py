#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from pandas import DataFrame
import numpy as np

filename = "student-mat.csv"
data = pd.read_csv(filename)
data.shape
df = pd.DataFrame(data)

#dropped the columns based on the feature importance calculated below
x = df.drop(['school', 'sex', 'famsize','Pstatus','Medu','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','higher','internet','romantic'], axis=1)

x.shape
x.dtypes

#seperating the feature's which is of object data type to encode into numerical feature
obj_data = x.select_dtypes(include=['object']).copy()
obj_data.head()

obj_data.shape

#encoding the categorical feature into numerical feature
from sklearn.preprocessing import LabelEncoder
encoded_data = obj_data.apply(LabelEncoder().fit_transform)
encoded_data.shape


#droping the feature's in order to join the encoded feature's to the actual dataset
y = df.drop(['school', 'sex', 'famsize','Pstatus','Medu','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','higher','internet','romantic','activities','address','nursery'], axis=1)


y.shape


#based on the user defined condition the 'G3'(output) feature has been labeled as 0(fail) and 1(pass)
y['G3'] = (y['G3'] > 10).astype(int)


#here the encoded feature is joint to the actual dataset
y=y.join(encoded_data)


#rearranged the column name's and these are the features that are considered based on the feature importance calculation
y = y[['age', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','address', 'activities', 'nursery', 'G1',
       'G2', 'G3',]]


y.columns.values
y.dtypes

#input features
X=y.iloc[:,0:16]

#actual output to be considered
Y=y['G3']

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,Y)
print 'Feature importance based on extra tree classifier :'
print(model.feature_importances_)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
#Here I have used Decision Tree Classifier,SVM and Logistic Regression

clf = DecisionTreeClassifier()
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
clf=clf.fit(X,Y)

print 'Based on decision tree classifier the feature importance are : '
print(clf.feature_importances_)

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
y_pred = clf_gini.predict(X_test)
print 'Based on gini criterion the prediction result in desicion tree classifier : '
print (y_pred)
#from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred_en = clf_entropy.predict(X_test)
print 'Based on entropy criterion the prediction result in desicion tree classifier : '
print (y_pred_en)
X_train.shape
X_test.shape
print 'Accuracy rate for desicion tree classifier '
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

from sklearn import svm
#Creating a svm Classifier
clf = svm.SVC(kernel='linear',degree=3,gamma='auto',C=1.0) # Linear Kernel(RBF kernel and polynomial gives lower accuracy rate when compared with linear kernel)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print 'Accuracy,Precision and recall rate for SVM : '
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000).fit(X, Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=100) # 70% training and 30% test



#clf.predict(X_test)
clf.predict_proba(X_test) 
print 'Accuracy rate for logistic regression : '
print(clf.score(X,Y))


#So when comparing the three above mentioned classifier's based on the accuracy rate decisiontreeclassifier has performed well

