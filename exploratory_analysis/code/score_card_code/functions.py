#coding:utf-8
import numpy as np
import pandas as pd

def getMobileFromOrderid(path_file, filename1, filename2):
    df_order_info = pd.read_csv(path_file + filename1)
    df_needed_order_info = df_order_info[[
        'order_id', 'cust_id', 'ovd_daynum',]]

    df_cust_base_info = pd.read_csv(path_file + filename2)
    df_needed_cust_base_info = df_cust_base_info[['id', 'mobile',]]

    df_merged = pd.merge(df_needed_order_info, df_needed_cust_base_info,\
            left_on="cust_id", right_on="id", how="left")
    print df_merged.head()
    filename_to_write = "needed_order_id_with_mobile.csv"
    #df_merged.to_csv(path_file + filename_to_write, index=False)

def splitMixIndexDetailInfo(path_file, filename):
    df_needed_detail_info = pd.read_csv(path_file + filename)
    #print len(df_needed_detail_info)
    df_all_cat = df_needed_detail_info.drop_duplicates(
            ['order_id', 'id', 'mobile',], keep='last')
    #print len(df_needed_detail_info)
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
    #print df_all_cat.head(n=10)
    df_all_cat.to_csv(path_file + "splited_detail_info.csv", index=False)

def dealWithOutliers(df_splited_detail_info):
    list_df_one_cat = []
    for v in df_splited_detail_info.columns[4:]:
        df_one_cat = df_splited_detail_info[[v]]
        all_counts = len(df_one_cat)

        # 异常值分析及处理
        df_desc_one_cat = df_one_cat.describe()
        Q1 = df_desc_one_cat[v].ix['25%']
        Q3 = df_desc_one_cat[v].ix['75%']
        IQR = Q3 - Q1
        df_Q3_mild_outliers = df_one_cat[df_one_cat[v] > Q3+1.5*IQR]
        df_Q1_mild_outliers = df_one_cat[df_one_cat[v] < Q1-1.5*IQR]
        Q3_mild_outliers_counts = len(df_Q3_mild_outliers)
        Q1_mild_outliers_counts = len(df_Q1_mild_outliers)
        all_mild_outliers_counts =\
                Q1_mild_outliers_counts + Q3_mild_outliers_counts
        Q3_mild_outliers_proportion =\
                Q3_mild_outliers_counts*1.0/all_counts
        Q1_mild_outliers_proportion =\
                Q1_mild_outliers_counts*1.0/all_counts
        all_mild_outliers_proportion =\
                Q1_mild_outliers_proportion + Q3_mild_outliers_proportion
#        print "Q3 mild outliers counts and proportion: ", 
#        print Q3_mild_outliers_counts, Q3_mild_outliers_proportion
#        print "Q1 mild outliers counts and proportion: ", 
#        print Q1_mild_outliers_counts, Q1_mild_outliers_proportion
#        print "all mild outliers counts and proportion: ",
#        print all_mild_outliers_counts, all_mild_outliers_proportion


        df_Q3_extreme_outliers = df_one_cat[df_one_cat[v] > Q3+3*IQR]
        df_Q1_extreme_outliers = df_one_cat[df_one_cat[v] < Q1-3*IQR]
        Q3_extreme_outliers_counts = len(df_Q3_extreme_outliers)
        Q1_extreme_outliers_counts = len(df_Q1_extreme_outliers)
        all_extreme_outliers_counts =\
                Q1_extreme_outliers_counts + Q3_extreme_outliers_counts
        Q3_extreme_outliers_proportion =\
                Q3_extreme_outliers_counts*1.0/all_counts
        Q1_extreme_outliers_proportion =\
                Q1_extreme_outliers_counts*1.0/all_counts
        all_extreme_outliers_proportion =\
                Q1_extreme_outliers_proportion +\
                Q3_extreme_outliers_proportion
#        print "Q3 extreme outliers counts and proportion: ", 
#        print Q3_extreme_outliers_counts, Q3_extreme_outliers_proportion
#        print "Q1 extreme outliers counts and proportion: ", 
#        print Q1_extreme_outliers_counts, Q1_extreme_outliers_proportion
#        print "all extreme outliers counts and proportion: ",
#        print all_extreme_outliers_counts, all_extreme_outliers_proportion

def printDescOfDataFrame(df_sample):
    pass


if __name__ == "__main__":
    working_space = "C:/Users/Administrator/Desktop/exploratory_analysis/"
    path_file = working_space + "data/dataset_score_card/"
    filename = "splited_detail_info.csv"
    df = pd.read_csv(path_file + filename)
    print len(df)
    print df.describe('id')


