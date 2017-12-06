#coding:utf-8

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

X, y = make_classification(
	n_samples=1000, n_features=32, n_informative=4,
	n_redundant=2, n_repeated=0, n_classes=8,
	n_clusters_per_class=1, random_state=0)
print X, y

svc = SVC(kernel='linear')
rfecv = RFECV(estimator=svc, step=1,\
		cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X, y)
print rfecv.n_features_
print rfecv.support_
print rfecv.ranking_
print rfecv.grid_scores_
print rfecv.estimator

#plt.figure()
#plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
#plt.show()

