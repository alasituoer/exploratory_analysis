#coding:utf-8
import matplotlib.pyplot as plt
from functions import *


if __name__ == "__main__":
    working_space = "C:/Users/Administrator/Desktop/exploratory_analysis/"
    path_file = working_space + "data/dataset_score_card/"

    testSklearn()

    #filename = "BC.csv"
    #df = pd.read_csv(path_file + filename)
    #print df.head()

    #filename = "splited_detail_info.csv"
    #df = pd.read_csv(path_file + filename)
    #print df.head()
    #print df.describe()

    #filename_order_info = "cash_order_info.txt"
    #filename_cust_base_info = "customer_base_info.txt"
    #getMobileFromOrderid(path_file,\
    #        filename_order_info, filename_cust_base_info)

    # 对去重后的电话详单填充缺失值
    #filename_needed_detail_info = "needed_detail_info.csv"
    #splitMixIndexDetailInfo(path_file, filename_needed_detail_info)

    # 将填充缺失值后的电话详单的混合列分列
    #filename_splited_detail_info = "splited_detail_info.csv"
    #descFillNA(path_file, filename_splited_detail_info)
    








