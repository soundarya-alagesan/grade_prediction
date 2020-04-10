#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The goal here is to predict the final grades of each student based on the features(input's) given. So, initially I read 
#the dataset by using pandas and dropped the unimportant features based on the feature importance calculation. Then I coverted 
#the categorical dataset to numerical, based on label encoding. And here I used three classifiers primiraly,based on the dataset
#that are decisiontree,support vector machine and logistic regression.The accuracy rate for each of the classifier is 
#calculated individually. And finally based on the accuracy rate calculated 'decisiontree classifier' has higher accuracy rate
#then the rest of the two in predicting the student's final grade.


# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from pandas import DataFrame
import numpy as np


# In[3]:


filename = "student-mat.csv"
data = pd.read_csv(filename)


# In[4]:


data.shape


# In[5]:


df = pd.DataFrame(data)


# In[6]:


#dropped the columns based on the feature importance calculated below
x = df.drop(['school', 'sex', 'famsize','Pstatus','Medu','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','higher','internet','romantic'], axis=1)


# In[7]:


x.shape


# In[8]:


x.dtypes


# In[9]:


#seperating the feature's which is of object data type to encode into numerical feature
obj_data = x.select_dtypes(include=['object']).copy()
obj_data.head()


# In[10]:


obj_data.shape


# In[11]:


#encoding the categorical feature into numerical feature
from sklearn.preprocessing import LabelEncoder
encoded_data = obj_data.apply(LabelEncoder().fit_transform)


# In[12]:


encoded_data.shape


# In[13]:


#droping the feature's in order to join the encoded feature's to the actual dataset
y = df.drop(['school', 'sex', 'famsize','Pstatus','Medu','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','higher','internet','romantic','activities','address','nursery'], axis=1)


# In[14]:


y.shape


# In[15]:


#based on the user defined condition the 'G3'(output) feature has been labeled as 0(fail) and 1(pass)
y['G3'] = (y['G3'] > 10).astype(int)


# In[16]:


#here the encoded feature is joint to the actual dataset
y=y.join(encoded_data)


# In[17]:


#rearranged the column name's and these are the features that are considered based on the feature importance calculation
y = y[['age', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','address', 'activities', 'nursery', 'G1',
       'G2', 'G3',]]


# In[18]:


y.columns.values


# In[19]:


y.dtypes


# In[20]:


#input features
X=y.iloc[:,0:16]


# In[21]:


#actual output to be considered
Y=y['G3']


# In[22]:


from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,Y)
print 'Feature importance based on extra tree classifier :'
print(model.feature_importances_)


# In[23]:


#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[24]:


#Here I have used Decision Tree Classifier,SVM and Logistic Regression


# In[25]:


clf = DecisionTreeClassifier()
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[26]:


clf=clf.fit(X,Y)


# In[27]:


print 'Based on decision tree classifier the feature importance are : '
print(clf.feature_importances_)


# In[28]:


#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[29]:


y_pred = clf_gini.predict(X_test)
print 'Based on gini criterion the prediction result in desicion tree classifier : '
print (y_pred)


# In[30]:


#from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[31]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# In[32]:


y_pred_en = clf_entropy.predict(X_test)
print 'Based on entropy criterion the prediction result in desicion tree classifier : '
print (y_pred_en)


# In[33]:


X_train.shape


# In[34]:


X_test.shape


# In[35]:


#from sklearn import metrics
print 'Accuracy rate for desicion tree classifier '
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[36]:


#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[37]:


from sklearn import svm

#Creating a svm Classifier
clf = svm.SVC(kernel='linear',degree=3,gamma='auto',C=1.0) # Linear Kernel(RBF kernel and polynomial gives lower accuracy rate when compared with linear kernel)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[38]:


#from sklearn import metrics
print 'Accuracy,Precision and recall rate for SVM : '
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[39]:


print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[40]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
#X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000).fit(X, Y)


# In[41]:


#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=100) # 70% training and 30% test


# In[42]:


#clf.predict(X_test)
clf.predict_proba(X_test) 
print 'Accuracy rate for logistic regression : '
print(clf.score(X,Y))


# In[43]:


#So when comparing the three above mentioned classifier's based on the accuracy rate decisiontreeclassifier has performed well

