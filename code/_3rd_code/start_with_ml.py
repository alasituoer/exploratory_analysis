#coding:utf-8
import numpy as np
import urllib

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
#print dataset.shape
# 8个特征和1个类别
X = dataset[:, 0:7]
y = dataset[:, 8]
#print X.shape
#print list(y).count(0)


# 特征预处理
from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)
standardized_X = preprocessing.scale(X)
print normalized_X
print standardized_X


#2 feather importance ranking 特征重要性评级
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
#print model.feature_importances_


#3 Recursive Feature Elimination 循环特征消减
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
#print rfe.support_, rfe.ranking_


#4 逻辑回归
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
#print model
expected  = y
predicted = model.predict(X)
#print metrics.classification_report(expected, predicted)
#print metrics.confusion_matrix(expected, predicted)


#5 朴素贝叶斯
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#model.fit(X, y)
model.fit(X, y)
#print model
expected = y
predicted = model.predict(X)
#print metrics.classification_report(expected, predicted)
#print metrics.confusion_matrix(expected, predicted)


#6 KNN
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(normalized_X, y)
expected = y
predicted = model.predict(normalized_X)
#print metrics.classification_report(expected, predicted)
#print metrics.confusion_matrix(expected, predicted)


#7 决策树
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X, y)
#print model
expected = y
predicted = model.predict(X)
#print metrics.classification_report(expected, predicted)
#print metrics.confusion_matrix(expected, predicted)


# 支持向量机
from sklearn import metrics
from sklearn.svm import SVC
model = SVC()
model.fit(X, y)
expected = y
predicted = model.predict(X)
#print metrics.classification_report(expected, predicted)
#print metrics.confusion_matrix(expected, predicted)







