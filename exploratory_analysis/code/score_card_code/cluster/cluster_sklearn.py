#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from data import removed_features
from data import discrete_features
from data import features_selected

from func import discreteFeaturesSelecting
from func import onehotDiscreteFeatures
from func import sepCorrFeatures
from func import continuousFeaturesSelecting
from func import sampleClustering



if __name__ == "__main__":
    working_space = "/mnt/hgfs/windows_desktop/exploratory_analysis/" +\
		      "data/dataset_score_card/cleaned_full_data/test/"
    #filename = "coll_dataset2_test.txt"
    filename = "coll_dataset2.txt"

    df = pd.read_csv(working_space + filename, index_col=0)
    selected_features = [ft for ft in df.columns if ft not in removed_features]
    df_selected = df[selected_features]
    # 检查并去掉缺失值太多的指标
    #count_ss = df_selected.describe().ix['count']
    #print count_ss.describe()
    #print count_ss[count_ss == 0]


    """
    #对离散型变量利用方差分析做特征选择, 返回拟选择离散特征名列表
    df_discrete = df_selected[['ovd_daynum'] + discrete_features]
    #print df_discrete.head()
    list_discrete_features_selected = discreteFeaturesSelecting(df_discrete)


    # 对筛选后的离散变量哑编码
    df_discrete = df_selected[list_discrete_features_selected]
    # 将缺失值填充为正无穷, 作为一类
    # 虽然LabelEncoder可以自动将缺失值划为一类
    # 但是无法将多个缺失值划为一类, 原因是np.nan != np.nan)
    df_discrete = df_discrete.fillna(float('inf'))
    onehotDiscreteFeatures(df_discrete)

    # 返回哑编码后的离散型指标l[1], 同时返回每个指标的构建关系l[0]
    label_encoder_encoded_data_list = onehotDiscreteFeatures(df_discrete)
    df_discrete_features_selected = label_encoder_encoded_data_list[1]
    # 看是否需要输出哑编码的编码规则
    dict_discrete_features_onehot_encoded = label_encoder_encoded_data_list[0]
    print "离散型变量数: \t", df_discrete.shape
    print "重编码后的离散变量数: \t", df_discrete_features_selected.shape
    """

    """
    # 去除连续型变量的共线性, 返回去共线性后的DataFrame
    continuous_features = [ft for ft in df_selected.columns\
	    if ft not in discrete_features]
    df_continuous = df_selected[continuous_features]
#    print df_continuous
    #print df_continuous.describe().ix['count']
    #print df_continuous.describe().ix['count'].describe()
    df_continuous = df_continuous.fillna(df_continuous.mean())
    #print df_continuous.describe().ix['count'].describe()
    df_continuous_removed_corr = sepCorrFeatures(df_continuous)
    print "连续型变量数: \t", df_continuous.shape
    print "去共线性后的连续变量数: \t", df_continuous_removed_corr.shape, '\n'

    # 对上述去除共线性的连续型变量做特征选择, 返回选择后的数据集
    path_to_write = working_space + "results/features_selected.csv"
    # 同时选择是否输出特征选择的依据(加上第二个存数路径参数),即各变量的得分
    df_continuous_features_selected =\
	    continuousFeaturesSelecting(df_continuous_removed_corr)
	    #continuousFeaturesSelecting(df_continuous_removed_corr, path_to_write)
    #print df_continuous_features_selected.head()
    print "特征选择后的连续变量数: \t", df_continuous_features_selected.shape, '\n'
    """

    # 样本聚类()
    df_features = df_selected[features_selected].dropna(how='any')
    sampleClustering(df_features)


