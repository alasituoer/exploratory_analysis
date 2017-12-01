#coding:utf-8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
#from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def clusterSelectedIndex(df_selected_index):
    df_selected_index = df_selected_index.dropna(axis=0, how='any')
    #df_selected_index.fillna(df_selected_index.mean(), inplace=True)
    #print df_selected_index.describe()
    X = df_selected_index.values
    scaled_x = Normalizer().fit_transform(X)
    #print scaled_x.shape

    df_raw = pd.DataFrame(X, columns=[x+'_raw' for x in df_selected_index.columns])
    df_scaled = pd.DataFrame(scaled_x, columns=df_selected_index.columns)
    df_one_var = pd.concat([df_raw['ovd_daynum_raw'], df_scaled['ovd_daynum']], axis=1)
    #print df_one_var.describe()    
    #print df_one_var[df_one_var['ovd_daynum_raw']==58]
    
    k = 4
    iteration = 500
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    model.fit(scaled_x)
    print pd.Series(model.labels_).value_counts()
    print pd.DataFrame(model.cluster_centers_)


def selectOrderInfo(df_order_info):
    #print df_order_info.describe()
    y = df_order_info['ovd_daynum'].values
    y = y.reshape(len(y),1)
    del df_order_info['ovd_daynum']
    X = df_order_info.values
    scaled_x = Normalizer().fit_transform(X)
    scaled_y = Normalizer().fit_transform(y)
    
    iter_num = 20
    top_num = 20
    clf = ExtraTreesClassifier()
    list_index = []
    for i in range(iter_num):
	clf.fit(scaled_x, scaled_y.ravel())
	#print clf.feature_importances_
	data = zip(df_order_info.columns, clf.feature_importances_)
	#print data
	top_index = sorted(data, key=lambda t: (-t[1]))[:top_num]
	#print top_index, '\n'
	list_index.append(top_index)
    #print list_index

    dict_index = {}
    list_name = list({x[0] for l in list_index for x in l})
    for n in list_name:
	s = 0.0
	it = 0
	for l in list_index:
	    #print n, dict(l)[n]
	    s += dict(l)[n]
	    it += 1
	#print n, s, s*1.0/it
	dict_index[n] = s*1.0/it

    list_selected_index = [x for x in dict_index.iteritems()\
	    if x[1] > max(dict_index.values())/2]
    list_selected_index = sorted(list_selected_index,\
	    key=lambda x: x[1], reverse=True) 
    #print list_selected_index
    print [x[0] for x in list_selected_index]


def selectCustApp3rd(df_cust_app_3rd):
#    print len(df_cust_app_3rd.columns)
    y = df_cust_app_3rd['ovd_daynum'].values
    y = y.reshape(len(y),1)
    del df_cust_app_3rd['ovd_daynum']
    X = df_cust_app_3rd.values

    scaled_x = Normalizer().fit_transform(X)
    scaled_y = Normalizer().fit_transform(y)
    
    iter_num = 20
    top_num = 20
    clf = ExtraTreesClassifier()
    list_index = []
    for i in range(iter_num):
	clf.fit(scaled_x, scaled_y.ravel())
	#print clf.feature_importances_
	data = zip(df_cust_app_3rd.columns, clf.feature_importances_)
	#print data
	top_index = sorted(data, key=lambda t: (-t[1]))[:top_num]
	#print top_index, '\n'
	list_index.append(top_index)
    #print list_index
    # 获取iter_num组特征名的并集
    dict_index = {}
    list_name = list({x[0] for l in list_index for x in l})
    for n in list_name:
	s = 0.0
	it = 0
	for l in list_index:
	    #print n, dict(l)[n]
	    s += dict(l)[n]
	    it += 1
	#print n, s, s*1.0/it
	dict_index[n] = s*1.0/it

    list_selected_index = [x for x in dict_index.iteritems()\
	    if x[1] > max(dict_index.values())/2]
    list_selected_index = sorted(list_selected_index,\
	    key=lambda x: x[1], reverse=True) 
    print list_selected_index
    print [x[0] for x in list_selected_index]



def selectTelDetailInfo(df_tel_detail_info):
    y = df_tel_detail_info['ovd_daynum'].values
    y = y.reshape(len(y),1)
    del df_tel_detail_info['ovd_daynum']
    X = df_tel_detail_info.values
    #print X.shape, '\n', X, '\n'
    #print y.shape, '\n', y, '\n'

    # 删除达不到最低方差标准的特征(方差会受异常值的较大影响)
    #X_new = VarianceThreshold(threshold=10**5).fit_transform(X) 
    #print X_new.shape, '\n', X_new, '\n'

    # 无量纲化
    #scaled_x = MinMaxScaler().fit_transform(X)
    #scaled_y = MinMaxScaler().fit_transform(y)
    scaled_x = Normalizer().fit_transform(X)
    scaled_y = Normalizer().fit_transform(y)
    #print scaled_x.shape, '\n', scaled_x, '\n'
    #print scaled_y.shape, '\n', scaled_y, '\n'
    #print pd.DataFrame(scaled_x).describe()
    #print pd.DataFrame(scaled_y).describe()
    
    clf = ExtraTreesClassifier()
    # 重复多次得到特征的频数靠前的
    # 作为从121个特征中筛选得到的
    list_index = []
    iter_num = 20
    top_num = 20
    for i in range(iter_num):
	# 特征选择(树算法计算特征信息量)
	clf.fit(scaled_x, scaled_y.ravel())
	#print len(df_tel_detail_info.columns)
	#print clf.feature_importances_.shape
	# 返回排名靠前的特征(序号及特征名)
	data = zip(clf.feature_importances_, df_tel_detail_info.columns)
	top_index = sorted(data, key=lambda t: (-t[0]))[:top_num]
	list_index.extend([x[1] for x in top_index])
    dict_index_appear = dict((a, list_index.count(a)) for a in list_index)
    dict_index_appear = sorted(dict_index_appear.iteritems(),\
	    key=lambda t:t[1], reverse=True)
    print [x for x in dict_index_appear if x[1] > iter_num/2]







def pcaTelDetailInfo(df_tel_detail_info):
    # 特征选择完成后进行降维处理
    df_tel_detail_info.fillna(df_tel_detail_info.mean(), inplace=True)
    array_tel_detail_info = df_tel_detail_info.values
    array_std_detail_info = Normalizer().fit_transform(array_tel_detail_info)
    #print array_tel_detail_info.shape
    #print array_tel_detail_info
    print array_std_detail_info.shape
    #print array_std_detail_info

    pca = PCA(n_components='mle')
    #pca = PCA(n_components=5)
    print pca.fit_transform(array_std_detail_info)
    #print pca.components_
#    print 'explained_variance_ratio_', pca.explained_variance_ratio_#.sum()
    #print 'explained_variance_', pca.explained_variance_
    print 'n_components', pca.n_components_


def tryScalerData(filename):

#    print df.head()
#    print df.describe()

    df_test = df#[qualitative_variables_list]
    # 处理定性特征缺失值为'NA', 单独为一类
    # 对定性特征标准化
    for i in qualitative_variables_list:
	integer_encoded = LabelEncoder().fit_transform(df_test[i])
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	#print integer_encoded
	# 需不需要对定性变量哑编码(减少信息损失)
	#onehot_encoder = OneHotEncoder(sparse=False)
	#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	#print onehot_encoded
	df_test[i] = integer_encoded
    #print df_test
    
    df_test.dropna(axis=0, how='any', inplace=True)
    #print df_test

    #print df_test.as_matrix()
    print df_test.values
    #mat_test = StandardScaler().fit_transform(df_test.values)
    mat_test = Normalizer().fit_transform(df_test.values)
    print mat_test.shape
    print sum(mat_test.T)
    

