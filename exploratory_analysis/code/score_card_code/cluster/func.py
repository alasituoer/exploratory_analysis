#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC


def clusterSelectedIndex(df_selected_index):
    #print df_selected_index.describe()

    # 因后面的递归特征消除法筛选的代删除特征
#    df_selected_index = df_selected_index.drop([], axis=1)

    # 对各部分合并的总指标检查共线性
    df_corr = df_selected_index.corr()
    df_corr_t1 = df_corr[df_corr>=0.8].replace(1.0, np.nan)
    df_corr_t1.dropna(axis=0, how='all', inplace=True)
    df_corr_t1.dropna(axis=1, how='all', inplace=True)
    if len(df_corr_t1)>0:
	print df_corr_t1, '\n'

    datasets = df_selected_index.values
    labels_list = df_selected_index.columns.tolist()
    maxabs_scaled = MaxAbsScaler().fit_transform(datasets)
    #print datasets.shape, '\n', datasets, '\n'
    #print maxabs_scaled.shape, '\n', maxabs_scaled, '\n'

    maxabs_scaled = maxabs_scaled[:2000, :]
    k = 5
    iteration = 500
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    y_pred = model.fit_predict(maxabs_scaled)
    print pd.Series(model.labels_).value_counts()
    df_cluster_centers = pd.DataFrame(model.cluster_centers_, columns=labels_list)
    df_cluster_centers.to_csv('cluster_' + str(k) + '_centers_.csv')
    print df_cluster_centers
    #print y_pred

    # 递归特征消除法 选择特征变量
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1,\
		cv=StratifiedKFold(2), scoring="accuracy")
    rfecv.fit(maxabs_scaled, y_pred)
    print rfecv.n_features_
    print rfecv.support_
    print sorted(zip(rfecv.ranking_, df_selected_index.columns))

    X_tsne = TSNE(learning_rate=100).fit_transform(maxabs_scaled)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:,0], X_tsne[:, 1], c=y_pred)
    plt.show()


#feature seleceting 电话详单, 客户信息+p2p+第三方, 订单信息
def featureSelecting(df_index_tobe_selected, *path_to_write):
    #print df_index_tobe_selected.describe()

    """
    datasets = df_index_tobe_selected.values
    labels_list = df_index_tobe_selected.columns.tolist()

    #1 Normalizer()
    #normal_scaler = Normalizer().fit_transform(datasets)
    #print normal_scaler[:, 0]

    #2 MaxAbsScaler [采用]
    maxabs_scaler = MaxAbsScaler().fit_transform(datasets)
    #print datasets.shape, '\n', datasets, '\n'
    #print maxabs_scaler.shape, '\n', maxabs_scaler, '\n'
    y = maxabs_scaler[:, 0]
    X = maxabs_scaler[:, 1:]
    #print X.shape, '\n', X, '\n'
    #print y.shape, '\n', y, '\n'

    # RandomizedLasso [采用]
    rlasso = RandomizedLasso()
    rlasso.fit(X, y)
    rank_list = sorted(zip(labels_list[1:], 
	    map(lambda x: round(x, 4), rlasso.scores_)),
	    key=lambda x: x[1], reverse=True)

    #1 RandomFroestRegressor
    #rf = RandomForestRegressor()
    #rf.fit(X, y)
    #print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_),
	#    labels_list[1:]), reverse=True)

    #2 target should be types or muticlasses [removed]
    #clf = ExtraTreesClassifier()
    #clf.fit(X, y)
    #print sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_),
	#    labels_list[1:]), reverse=True)

    #print rank_list
    df_score_rank = pd.DataFrame(rank_list, columns=['label', 'score_rank',])
    # save the result
    print df_score_rank[df_score_rank['score_rank']>=0.8]['label'].values.tolist()
    print '\n\n'
    #df_score_rank.to_csv(filename + '.csv', index=False)
    """

def sepCorrFeatures(df_index_tobe_selected):
    """输入一个DataFrame或ndarray, 筛选高度共线性变量,
	返回删除共线性特征后的DataFrame"""
    #DataFrame.corr(method='pearson', min_periods=1)
    #method: ['pearson', 'kendall', 'spearman',]

    all_features_list = df_index_tobe_selected.columns.tolist()
#    print "去除共线性前的特征数: ", len(all_features_list)
    # 截取correlation coefficient > corr_coef_ 部分, 返回截取后的DataFrame
    corr_coef_ = 0.8

    # 计算特征间相关系数矩阵, 作为循环判断的起始条件
    df_corr = df_index_tobe_selected.corr()
    df_corr_t1 = df_corr[df_corr >= corr_coef_].replace(1.0, np.nan)
    df_corr_t1.dropna(axis=0, how='all', inplace=True)
    df_corr_t1.dropna(axis=1, how='all', inplace=True)
    feature_tobe_removed_list = []
    while len(df_corr_t1)>0:
        max_series = df_corr_t1.max()
        corr_feature = max_series[max_series==max_series.max()].index
#        print '相关系数矩阵中最大值及对应特征: ', corr_feature, max_series.max()
        corr1_left_feature_list =\
	    df_corr_t1.ix[corr_feature[0]].drop(corr_feature[1]).dropna().values.tolist()
        corr2_left_feature_list =\
	    df_corr_t1.ix[corr_feature[1]].drop(corr_feature[0]).dropna().values.tolist()
#        print corr1_left_feature_list
#        print corr2_left_feature_list

	try:
	    max_corr1_left_feature = max(corr1_left_feature_list)
	except:
	    max_corr1_left_feature = 0
	try:
	    max_corr2_left_feature = max(corr2_left_feature_list)
	except:
	    max_corr2_left_feature = 0
#        print corr_feature[0], '的剩余交叉特征最大相关系数: ', max_corr1_left_feature
#        print corr_feature[1], '的剩余交叉特征最大相关系数: ', max_corr2_left_feature

        #删除与其他特征最大相关系数较小的特征
        if max_corr1_left_feature >= max_corr2_left_feature:
#	    print '拟删除特征: ', corr_feature[1]
	    # 从大于指定阈值的待删减系数矩阵去除已确定的删减特征
	    df_corr_t1.drop(corr_feature[1], axis=0, inplace=True)
	    df_corr_t1.drop(corr_feature[1], axis=1, inplace=True)
	    df_corr_t1.dropna(axis=0, how='all', inplace=True)
	    df_corr_t1.dropna(axis=1, how='all', inplace=True)
	    feature_tobe_removed_list.append(corr_feature[1])
        else:
#	    print '拟删除特征: ', corr_feature[0]
	    #df_index_tobe_selected.drop(corr_feature[0], axis=1, inplace=True)
	    df_corr_t1.drop(corr_feature[0], axis=0, inplace=True)
	    df_corr_t1.drop(corr_feature[0], axis=1, inplace=True)
	    df_corr_t1.dropna(axis=0, how='all', inplace=True)
	    df_corr_t1.dropna(axis=1, how='all', inplace=True)
	    feature_tobe_removed_list.append(corr_feature[0])

#	print '新特征间相关系数矩阵特征数: ', len(df_corr_t1), '\n'
#	print df_corr_t1

#    print feature_tobe_removed_list
    return df_index_tobe_selected[[f for f in all_features_list\
	    if f not in feature_tobe_removed_list]]


def onehotDiscreteFeatures(df):
    """将定性变量重编码为哑变量, 返回转换后的新DataFrame, 同时返回各变量的重编码规则
	以列表的形式返回 [dict_all_cates_label_encoder, df_onehot_encoded]"""
    #print df.describe().ix['count']
    #count_ss = df.describe().ix['count']
    #print count_ss
    #print count_ss.describe()

    df_init = pd.DataFrame(np.zeros(len(df)), columns=['init'])
    # 以字典的格式存放所有列的重编码规则
    dict_all_cates_label_encoder = {}
    for c in df.columns:
	tobe_encoded = df[c].values
#	print tobe_encoded, '\n'
	label_encoder = LabelEncoder()
	label_encoder.fit(tobe_encoded)
	#print 'label, tobe_encoded'

	# 临时存放某一列的重编码规则
	dict_one_cate_label_encoder = {}
	for ix, it in enumerate(label_encoder.classes_):
	    dict_one_cate_label_encoder[ix] = it
	#print dict_one_cate_label_encoder
	dict_all_cates_label_encoder[c] = dict_one_cate_label_encoder

	label_encoded = label_encoder.transform(tobe_encoded)
	label_encoded = np.array(label_encoded).reshape(len(label_encoded), 1)
	#print label_encoded
	onehot_encoded = OneHotEncoder().fit_transform(label_encoded).toarray()
	columns = [c + '_mat_' + str(i) for i in range(len(label_encoder.classes_))]
#	print len(onehot_encoded), '\n', onehot_encoded, '\n'
	df_init = pd.concat([df_init,
		pd.DataFrame(onehot_encoded, columns=columns)], axis=1)
    df_onehot_encoded = df_init.drop(columns=['init'], axis=1)
#    print dict_all_cates_label_encoder['industry'][0]
#    print dict_all_cates_label_encoder
#    print df_onehot_encoded.head()

    return [dict_all_cates_label_encoder, df_onehot_encoded]






# PCA 降维
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


