#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import removed_features
from data import discrete_features

from data import tel_detail_info_index_list
from data import cust_app_3rd_index_list
from data import order_info_index_list

from data import selected_tel_detail_info_index_list
from data import selected_cust_app_3rd_index_list
from data import selected_order_info_index_list

from func import onehotDiscreteFeatures
from func import sepCorrFeatures
from func import featureSelecting
from func import clusterSelectedIndex
#from func import rfeSelectedIndex



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

    # 统计离散型变量的特征
    df_discrete = df_selected[discrete_features]
    # 将缺失值填充为正无穷, 作为一类
    # 虽然LabelEncoder可以自动将缺失值划为一类
    # 但是无法将多个缺失值划为一类, 原因是np.nan != np.nan)
    df_discrete = df_discrete.fillna(float('inf'))
    # 返回哑编码后的离散型指标, 同时返回每个指标的构建关系
#    label_encoder_encoded_data_list = onehotDiscreteFeatures(df_discrete)
#    print label_encoder_encoded_data_list[1]
    

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
#    print df_continuous_removed_corr

    # 对上述去除共线性的连续型变量作特征选择
#    path_to_write = working_space + "/results/df_features_selected.csv"
    df_features_selected =\
	    featureSelecting(df_continuous_removed_corr)
    print df_features_selected



    # 对样本聚类分析
    """
    selected_index_list = selected_tel_detail_info_index_list +\
	    selected_cust_app_3rd_index_list + \
	    selected_order_info_index_list
    selected_index_list = [x for x in selected_index_list\
	    if x not in corr_all_index_list]
    df_selected_index = df[['ovd_daynum'] + selected_index_list]
    df_selected_index = df_selected_index.dropna(axis=0, how='any')
    df_selected_index.applymap(lambda x: float(x))
    #print len(df_selected_index)
    clusterSelectedIndex(df_selected_index) 

    # RFECV select features, 事先已知各样对应的分类结果
    #rfeSelectedIndex(df_selected_index)
    """

