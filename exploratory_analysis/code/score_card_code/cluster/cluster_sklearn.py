#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import removed_list
from data import tel_detail_info_index_list
from data import corr_tel_detail_info_index_list
from data import cust_app_3rd_index_list
from data import corr_cust_app_3rd_index_list
from data import order_info_index_list
from data import corr_order_info_index_list
from data import corr_all_index_list

from data import selected_tel_detail_info_index_list
from data import selected_cust_app_3rd_index_list
from data import selected_order_info_index_list

from func import sepCorrFeatureList
from func import featureSelecting
from func import clusterSelectedIndex
#from func import rfeSelectedIndex



if __name__ == "__main__":
    working_space = "/mnt/hgfs/windows_desktop/exploratory_analysis/" +\
		      "data/dataset_score_card/cleaned_full_data/test/"
    #filename = "coll_dataset2_test.txt"
    filename = "coll_dataset2.txt"

    df = pd.read_csv(working_space + filename, index_col=0)
    list_df_columns = list(df.columns)
    # 去除认定不需要的变量
    for i in removed_list:
	list_df_columns.remove(i)
    df = df[list_df_columns]
    #print df.head()

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

    """
    #2 客户基本信息+app+3rd
    cust_app_3rd_index_list = [x for x in cust_app_3rd_index_list\
	    if x not in corr_cust_app_3rd_index_list]
    df_cust_app_3rd = df[['ovd_daynum'] + cust_app_3rd_index_list]
    df_cust_app_3rd = df_cust_app_3rd.dropna(axis=0, how='any')
    #print df_cust_app_3rd.describe().ix['count']
    featureSelecting(df_cust_app_3rd, 'cust_app_3rd')
    """

    #1 对电话详单部分筛选变量
    #print len(tel_detail_info_index_list)
    #print corr_tel_detail_info_index_list
#    tel_detail_info_index_list = [x for x in tel_detail_info_index_list\
#	    if x not in corr_tel_detail_info_index_list]
    df_tel_detail_info = df[['ovd_daynum'] + tel_detail_info_index_list]
    df_tel_detail_info = df_tel_detail_info.dropna(axis=0, how='any')
    df_tel_detail_info.applymap(lambda x: float(x))
    sepCorrFeatureList(df_tel_detail_info)
#    featureSelecting(df_tel_detail_info, 'tel_detail_info')







