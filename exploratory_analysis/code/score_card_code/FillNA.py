#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *

def descFillNA(path_file, filename):
    df_splited_detail_info = pd.read_csv(path_file + filename)
    del df_splited_detail_info['UPDATE_DT']
#    print df_splited_detail_info.head(n=10)
#    print df_splited_detail_info.describe()

    # 缺失值分析及处理
    print df_splited_detail_info[df_splited_detail_info.isnull()]
#    df_splited_detail_info.fillna(method="ffill", inplace=True)

    # 调用另一函数处理异常值, 返回处理好的DataFrame
#    df_all_new = dealWithOutliers(df_splited_detail_info)
#    print df_all_new.describe()
    #df_all_new.to_csv(path_file + "cleaned_detail_info.csv", index=False)


if __name__ == "__main__":
    working_space = "C:/Users/Administrator/Desktop/exploratory_analysis/"
    path_file = working_space + "data/dataset_score_card/"

    filename = "splited_detail_info.csv"
    df = pd.read_csv(path_file + filename)
    print df.head()
    print df.describe()


    #filename_order_info = "cash_order_info.txt"
    #filename_cust_base_info = "customer_base_info.txt"
    #getMobileF0omOrderid(path_file, filename_order_info, filename_cust_base_info)

    #filename_needed_detail_info = "needed_detail_info.csv"
    #splitMixIndexDetailInfo(path_file, filename_needed_detail_info)

    #filename_splited_detail_info = "splited_detail_info.csv"
    #descFillNA(path_file, filename_splited_detail_info)








