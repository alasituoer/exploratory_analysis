#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
#from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def clusterSelectedIndex(df_selected_index):
#    print df_selected_index.describe()
    y = df_selected_index['ovd_daynum'][:200].values
    y = y.reshape(len(y),1)
    del df_selected_index['ovd_daynum']

    X = df_selected_index[:200].values
    scaled_x = Normalizer().fit_transform(X)
    #print scaled_x.shape
    X_tsne = TSNE(learning_rate=100).fit_transform(X)
    print X_tsne

    #plt.figure(figsize=(10, 5))
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    #plt.show()

"""
    k = 10
    iteration = 500
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    model.fit(scaled_x)
    print pd.Series(model.labels_).value_counts()
    print pd.DataFrame(model.cluster_centers_)
"""



def ivSelectedIndex(path_file, filename1, filename2):
    import woe.config as config
    import woe.feature_process as fp
    import woe.eval as eval

    data_path = path_file + filename1
    config_path = path_file + filename2
    cfg = config.config()
    cfg.load_file(config_path, data_path)
    
    # target ? 求IV值要先定义好坏样本, 由好坏比例求WOE
    #print type(cfg.dataset_train)
    print cfg.dataset_train.head()
    #print type(cfg.bin_var_list)
    
    """
    for var in cfg.bin_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 0
    
    # change feature dtypes
    fp.change_feature_dtype(cfg.dataset_train, cfg.variable_type)
    
    #print cfg.variable_type
    #print cfg.dataset_train.head()
    
    rst = []
    
    # process woe transformation of continuous variables
    for var in cfg.bin_var_list[:3]:
        rst.append(fp.proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))
    
    # process woe transformation of discrete variables
    for var in cfg.discrete_var_list:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
        rst.append(fp.proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))
    
    feature_detail = eval.eval_feature_detail(rst,'output_feature_detail.csv')
    """



# 订单信息
def selectOrderInfo(df_order_info):
    #print df_order_info.describe()

    #df_corr = df_order_info.corr()
    #df_corr_t1 = df_corr[df_corr>=0.8].replace(1.0, np.nan)
    #df_corr_t1.dropna(axis=0, how='all', inplace=True)
    #df_corr_t1.dropna(axis=1, how='all', inplace=True)
    #print df_corr_t1
    
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


# 客户基本信息 APP信息 第三方信息
def selectCustApp3rd(df_cust_app_3rd):
    #print df_cust_app_3rd.describe()

    #df_corr = df_cust_app_3rd.corr()
    #df_corr_t1 = df_corr[df_corr>0.8].replace(1.0, np.nan)
    #df_corr_t1.dropna(axis=0, how='all', inplace=True)
    #df_corr_t1.dropna(axis=1, how='all', inplace=True)
    #print df_corr_t1

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



# 电话详单
def selectTelDetailInfo(df_tel_detail_info):
    #print df_tel_detail_info.describe()

    # 筛选高度共线性变量
    #df_corr = df_tel_detail_info.corr()
    #df_corr_t1 = df_corr[df_corr>=0.8].replace(1.0, np.nan)
    #df_corr_t1.dropna(axis=0, how='all', inplace=True)
    #df_corr_t1.dropna(axis=1, how='all', inplace=True)
    #print df_corr_t1

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
    

