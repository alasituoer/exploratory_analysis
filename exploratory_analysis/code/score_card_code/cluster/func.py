#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, permutations

#from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
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
    columns_list = df_selected_index.columns.tolist()
    maxabs_scaled = MaxAbsScaler().fit_transform(datasets)
    #print datasets.shape, '\n', datasets, '\n'
    #print maxabs_scaled.shape, '\n', maxabs_scaled, '\n'

    maxabs_scaled = maxabs_scaled[:2000, :]
    k = 5
    iteration = 500
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    y_pred = model.fit_predict(maxabs_scaled)
    print pd.Series(model.labels_).value_counts()
    df_cluster_centers = pd.DataFrame(model.cluster_centers_, columns=columns_list)
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

def sampleClustering(df_features):
    print "原始数据: ", df_features.shape
    #print df_features.describe().ix['count']
    
    #print list(combinations(df_features.columns, 5))
    
    #df_features = df_features.sample(5, axis=1)[:5000]
    df_features = df_features[:5000]
    datasets = StandardScaler().fit_transform(df_features.values)
    print "随机抽样: ", datasets.shape
    print "",
    #datasets = PCA(n_components=0.9).fit(datasets.T).components_.T
    #print "PCA降维后: ", datasets.shape

    k = 4
    iteration = 500
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    y_pred = model.fit_predict(datasets)
    #print "聚类中心:\n", model.cluster_centers_.shape, '\n', model.cluster_centers_
    #print "样本标签:\n", model.labels_.shape, '\n', model.labels_

    tsne_reduced = TSNE().fit_transform(datasets)
    fig, ax = plt.subplots()
    x,y = tsne_reduced[:, 0], tsne_reduced[:, 1]
    ax.scatter(x, y, c=y_pred)
    #ax.grid(True)
    plt.show()

    """
    tsne_reduced = TSNE(n_components=3).fit_transform(datasets)
    #print tsne_reduced
    fig = plt.figure()
    ax = Axes3D(fig)
    x,y,z = tsne_reduced[:, 0], tsne_reduced[:, 1], tsne_reduced[:, 2]
    ax.scatter(x,y,z, c=y_pred)
    plt.show()
    """


#continuous features seleceting 
def continuousFeaturesSelecting(df_continuous_removed_corr, *path_to_write):
    """对所有变量作标准化处理, 然后分为X自变量(除ovd_daynum)和y因变量(逾期天数),
	    采用随机Lasso选择对y回归贡献得分大于0.8的X,
	    返回y和X组成的特征选择后的数据"""
    #print df_continuous_removed_corr.describe()

    datasets = df_continuous_removed_corr.values
    columns_list = df_continuous_removed_corr.columns.tolist()
    #print columns_list

    #1 Normalizer() 归一化
    #normal_scaler = Normalizer().fit_transform(datasets)
    #print normal_scaler[:, 0]

    #2 StandardScaler() 标准化
    std_scaled = StandardScaler().fit_transform(datasets)
    y = std_scaled[:, 0]
    X = std_scaled[:, 1:]
#    print datasets
#    print X.shape, '\n', X, '\n'
#    print y.shape, '\n', y, '\n'

    # RandomizedLasso 
    rlasso = RandomizedLasso()
    rlasso.fit(X, y)
    # 给ndarray格式特征匹配上特征名(此处不包括y逾期天数)
    list_features_rank = sorted(zip(columns_list[1:], 
	    map(lambda x: round(x, 4), rlasso.scores_)),
	    key=lambda x: x[1], reverse=True)
    df_features_rank = pd.DataFrame(list_features_rank,
	    columns=['features_label', 'features_rank',])

    # 如果选择了存数路径, 则将连续型特征及其排名得分存入指定文件中
    if path_to_write:
	df_features_rank.to_csv(path_to_write[0], index=False)

    # 截取得分在1.0及以上的特征X
    list_columns_features_selected =\
	    df_features_rank[df_features_rank[
	    'features_rank']>=1.0]['features_label'].values.tolist()
    # 再添加上y(逾期天数), 构成进行样本聚类的所有特征
    list_columns_features_selected.insert(0, 'ovd_daynum')
#    print list_features_rank
#    print df_features_rank
#    print list_columns_features_selected
    # 在输入DataFrame上删减后的DataFrame
    df_features_selected = df_continuous_removed_corr[list_columns_features_selected]

    # 对删减后的DataFrame作标准化后再返回, 方便直接进行样本聚类
    #print df_features_selected.values
    #print StandardScaler().fit_transform(df_features_selected.values)
    df_continuous_features_selected =\
	    pd.DataFrame(StandardScaler().fit_transform(df_features_selected.values),
	    index=df_features_selected.index, columns=df_features_selected.columns)
    #print df_features_selected.head()
    #print df_continuous_features_selected.head()
    # 按行展示均值和方差
    #print df_continuous_features_selected.values.mean(axis=0)
    #print df_continuous_features_selected.values.var(axis=0)

    return df_continuous_features_selected


def sepCorrFeatures(df_tobe_removed_corr):
    """输入一个DataFrame或ndarray, 筛选高度共线性变量,
	返回删除共线性特征后的DataFrame"""
    #DataFrame.corr(method='pearson', min_periods=1)
    #method: ['pearson', 'kendall', 'spearman',]

    all_features_list = df_tobe_removed_corr.columns.tolist()
#    print "去除共线性前的特征数: ", len(all_features_list)
    # 截取correlation coefficient > corr_coef_ 部分, 返回截取后的DataFrame
    corr_coef_ = 0.8

    # 计算特征间相关系数矩阵, 作为循环判断的起始条件
    df_corr = df_tobe_removed_corr.corr()
    #df_corr_t1 = df_corr[df_corr >= corr_coef_].replace(1.0, np.nan)
    # 认为相关系数大于0.8或者小于-0.8的为重度相关
    df_corr_t1 = df_corr[(df_corr >= corr_coef_) |
	    (df_corr <= -corr_coef_)].replace(1.0, np.nan)
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
	    #df_tobe_removed_corr.drop(corr_feature[0], axis=1, inplace=True)
	    df_corr_t1.drop(corr_feature[0], axis=0, inplace=True)
	    df_corr_t1.drop(corr_feature[0], axis=1, inplace=True)
	    df_corr_t1.dropna(axis=0, how='all', inplace=True)
	    df_corr_t1.dropna(axis=1, how='all', inplace=True)
	    feature_tobe_removed_list.append(corr_feature[0])

#	print '新特征间相关系数矩阵特征数: ', len(df_corr_t1), '\n'
#	print df_corr_t1
#    print feature_tobe_removed_list
    return df_tobe_removed_corr[[f for f in all_features_list\
	    if f not in feature_tobe_removed_list]]


def onehotDiscreteFeatures(df):
    """将定性变量重编码为哑变量, 返回转换后的新DataFrame, 同时返回各变量的重编码规则
	以列表的形式返回 [dict_all_cates_label_encoder, df_onehot_encoded]"""
    print df.describe().ix['count']
    #count_ss = df.describe().ix['count']
    #print count_ss
    #print count_ss.describe()

    df_init = pd.DataFrame(np.zeros(len(df)), columns=['init'], index=df.index)
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
	# 按照order_id进行列合并(axis=1)
	df_init = pd.concat([df_init,
		pd.DataFrame(onehot_encoded, columns=columns, index=df.index)], axis=1)
    df_onehot_encoded = df_init.drop(columns=['init'], axis=1)
#    print dict_all_cates_label_encoder['industry'][0]
#    print dict_all_cates_label_encoder
#    print df_onehot_encoded.head()

    return [dict_all_cates_label_encoder, df_onehot_encoded]

def discreteFeaturesSelecting(df):
    """对输入离散特征做选择, 返回拟选择特征名列表
	(对定性变量与逾期天数做方差分析, 即Anova F-Value检验)"""
    #print df.describe()
    return ['payment_amount', 'reloan', 'gender',]



















