#coding:utf-8
import pandas as pd

from data import removed_list
from data import qualitative_variables_list
from data import tel_detail_info_index_list

from func import pcaTelDetailInfo
from func import selectTelDetailInfo



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

    # 对电话详单部分筛选变量
    df_tel_detail_info = df[['ovd_daynum'] + tel_detail_info_index_list]
    df_tel_detail_info = df_tel_detail_info.dropna(axis=0, how='any')
    df_tel_detail_info.applymap(lambda x: float(x))
    #print df_tel_detail_info.describe()
    #print df_tel_detail_info.head()
    selectTelDetailInfo(df_tel_detail_info)

    # 对电话详单样本数据做主成份分析
    #df_tel_detail_info = df[tel_detail_info_index_list]
    #pcaTelDetailInfo(df_tel_detail_info)











