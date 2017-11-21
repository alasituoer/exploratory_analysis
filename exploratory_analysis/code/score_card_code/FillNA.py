#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getMobileFromOrderid(path_file, filename1, filename2):
    df_order_info = pd.read_csv(path_file + filename1)
    df_needed_order_info = df_order_info[['order_id', 'cust_id', 'ovd_daynum',]]

    df_cust_base_info = pd.read_csv(path_file + filename2)
    df_needed_cust_base_info = df_cust_base_info[['id', 'mobile',]]

    df_merged = pd.merge(df_needed_order_info, df_needed_cust_base_info,\
            left_on="cust_id", right_on="id", how="left")
    print df_merged.head()
    filename_to_write = "needed_order_id_with_mobile.csv" 
    #df_merged.to_csv(path_file + filename_to_write, index=False)


def splitMixIndexDetailInfo(path_file, filename):
    df_needed_detail_info = pd.read_csv(path_file + filename)
    df_all_cat = df_needed_detail_info.drop_duplicates(['order_id', 'id', 'mobile',], keep='last')
    #print df_all_cat.head(n=6)

    list_mix_index = ['C6_VALUE', 'C7_VALUE', 'C8_VALUE',]
    for v in list_mix_index:
        df_split_cat = df_all_cat[v].str.split(',', expand=True)
        df_split_cat.columns = [v + "_" +\
                str(i+1) for i in df_split_cat.columns]
        df_split_cat = df_split_cat.replace("", np.nan).replace("NA",\
                np.nan).fillna(np.nan).applymap(lambda x: float(x))
        df_all_cat = pd.concat([df_all_cat, df_split_cat], axis=1)
    df_all_cat.drop(list_mix_index, axis=1, inplace=True)    
    print df_all_cat.head(n=10)
    #df_all_cat.to_csv(path_file + "splited_detail_info.csv", index=False)

def descFillNA(path_file, filename):
    df_splited_detail_info = pd.read_csv(path_file + filename)
    #print df_splited_detail_info.head()

    for v in df_splited_detail_info.columns[4:5]:
        df_one_cat = df_splited_detail_info[[v]]
        print df_one_cat.describe()
        NA_ALL = df_one_cat
        #df_one_cat.boxplot()
        #plt.show()






if __name__ == "__main__":
    working_space = "C:/Users/Administrator/Desktop/exploratory_analysis/"
    path_file = working_space + "data/dataset_score_card/"

    filename_order_info = "cash_order_info.txt"
    filename_cust_base_info = "customer_base_info.txt"
    #getMobileFromOrderid(path_file, filename_order_info, filename_cust_base_info)

    filename_needed_detail_info = "needed_detail_info.csv"
    #splitMixIndexDetailInfo(path_file, filename_needed_detail_info)

    filename_splited_detail_info = "splited_detail_info.csv"
    descFillNA(path_file, filename_splited_detail_info)









