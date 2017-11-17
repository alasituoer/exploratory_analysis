#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

def selectedIndexTelDetailInfo(path_file, filename):
    """拆分混合字段为多列, 删除掉拆分过的字段, 导出新的文件"""
    df_all_cat = pd.read_table(path_file + filename)
    filename = "sample_detail_info.csv"
    list_mix_index = ['C6_VALUE', 'C7_VALUE', 'C8_VALUE', 'Ui_VALUE',
            'D1_VALUE', 'D2_VALUE', 'D3_VALUE',
            'D9_VALUE', 'D10_VALUE', 'D11_VALUE',]
    for v in list_mix_index:
        df_split_cat = df_all_cat[v].str.split(',', expand=True)
        df_split_cat.columns = [v + "_" +\
                str(i+1) for i in df_split_cat.columns]
        #print df_split_cat
        df_split_cat = df_split_cat.replace("", np.nan).replace("NA",\
                np.nan).fillna(np.nan).applymap(lambda x: float(x))
        df_all_cat = pd.concat([df_all_cat, df_split_cat], axis=1)

    df_all_cat.drop(list_mix_index, axis=1, inplace=True)    
    df_all_cat.to_csv(path_file +\
            "telephone_detail_info_reviewed.csv", index=False)


def computeDescTelDetailInfo(path_file, filename):
    df = pd.read_csv(path_file + filename)

    series_desc_list = []
    for v in df.columns[4:]:
        #df[v].describe()
        series_desc_list.append(df[[v]].describe())

    # 将所有字段的描述性统计值横向拼接后输出
    df_desc_all = pd.concat(series_desc_list, axis=1)
    df_desc_all.to_csv(path_file + "desc_del_detail_info.csv")


def saveBoxplotFigTelDetailInfo(path_file, filename):
    df = pd.read_csv(path_file + filename)
    # 挨个字段画出并存储箱线图
    for v in df.columns[4:]:
        df[[v]].boxplot()
        plt.savefig(path_file + "NA_boxplot/" + v + ".png")
        plt.show()


if __name__ == "__main__":
    print "hello alas"
    working_space = "C:/Users/cherish/Desktop/exploratory_analysis/"
    path_file = working_space + "data/_3rd_data/"

#    filename1 = "telephone_detail_info.txt"
#    selectedIndexTelDetailInfo(path_file, filename1)

    filename2 = "telephone_detail_info_reviewed.csv"
    computeDescTelDetailInfo(path_file, filename2)


    filename3 = "telephone_detail_info_reviewed.csv"
    saveBoxplotFigTelDetailInfo(path_file, filename3)


"""
            # 去掉异常值, 防止箱线图受极度异常值的影响
            series_desc = df_one_cat.describe()
            #print series_desc
            Q1 = series_desc.ix['25%']
            Q3 = series_desc.ix['75%']
            IQR = Q3 - Q1
            df_one_cat = df_one_cat[df_one_cat > (Q1 - IQR*5)]
            df_one_cat = df_one_cat[df_one_cat < (Q3 + IQR*5)]
"""






