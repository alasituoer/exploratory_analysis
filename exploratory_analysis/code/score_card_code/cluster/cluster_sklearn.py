#coding:utf-8
import pandas as pd

from data import removed_list
from data import tel_detail_info_index_list
from data import cust_app_3rd_index_list
from data import order_info_index_list
from data import selected_tel_detail_info_index_list
from data import selected_cust_app_3rd_index_list
from data import selected_order_info_index_list

#from func import pcaTelDetailInfo
from func import selectTelDetailInfo
from func import selectCustApp3rd
from func import selectOrderInfo
from func import clusterSelectedIndex



if __name__ == "__main__":
    working_space = "/mnt/hgfs/windows_desktop/exploratory_analysis/" +\
		      "data/dataset_score_card/cleaned_full_data/test/"
    #filename = "coll_dataset2_test.txt"
    filename = "coll_dataset2.txt"

    # DataFrame预处理
    df = pd.read_csv(working_space + filename, index_col=0)
    list_df_columns = list(df.columns)
    for i in removed_list:
	list_df_columns.remove(i)
    df = df[list_df_columns]
    
    # 
    selected_index_list = selected_tel_detail_info_index_list +\
	    selected_cust_app_3rd_index_list + \
	    selected_order_info_index_list
    #print selected_index_list
    df_selected_index = df[['ovd_daynum'] + selected_index_list]
    clusterSelectedIndex(df_selected_index) 




    #3 订单信息
    #df_order_info = df[['ovd_daynum'] + order_info_index_list]
    #print df_order_info.describe()
    #selectOrderInfo(df_order_info)


    #2 客户基本信息+app+3rd
    #df_cust_app_3rd = df[['ovd_daynum'] + cust_app_3rd_index_list]
    #print df_cust_app_3rd.describe()
    #df_cust_app_3rd = df_cust_app_3rd.fillna(0)
    #selectCustApp3rd(df_cust_app_3rd)


    #1 对电话详单部分筛选变量
    #df_tel_detail_info = df[['ovd_daynum'] + tel_detail_info_index_list]
    #df_tel_detail_info = df_tel_detail_info.dropna(axis=0, how='any')
    #df_tel_detail_info.applymap(lambda x: float(x))
    #print df_tel_detail_info.describe()
    #print df_tel_detail_info.head()
    #selectTelDetailInfo(df_tel_detail_info)

    # 对电话详单样本数据做主成份分析
    #df_tel_detail_info = df[tel_detail_info_index_list]
    #pcaTelDetailInfo(df_tel_detail_info)











