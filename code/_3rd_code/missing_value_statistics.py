#coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

if __name__ == "__main__":
    print "hello alas"
    working_space = "C:/Users/cherish/Desktop/exploratory_analysis/"
    path_file = working_space + "data/_3rd_data/"
    filename = "telephone_detail_info.txt"
    df_all_cat = pd.read_table(path_file + filename)
    #filename = "sample_detail_info.csv"
    #df_all_cat = pd.read_csv(path_file + filename)
    #print df_all_cat

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
#    path_to_write = path_file + "counts_NA.csv"
#    df_all_cat.count().to_csv(path_to_write)

    #print df_all_cat
    path_to_save = path_file + "NA_boxplot/"
    print df_all_cat.columns[4:]
    print len(df_all_cat.columns[4:])

    series_list = []
    for v in df_all_cat.columns[4:]:
        try:
            df_one_cat = df_all_cat[[v]].dropna(axis=0, how='any')
            #print df_one_cat
            series_list.append(df_one_cat.describe())
        except Exception, e:
            print e
            continue
    df_desc_all = pd.concat(series_list, axis=1)
    #print df_desc_all
    
    df_desc_all.to_csv(path_file + "NA_Desc_ALL.csv")


"""
            df_one_cat.boxplot()
            plt.savefig(path_to_save + v + '.png')
            #plt.show()
            print v
            time.sleep(3)
        except Exception, e:
            print e
            continue
"""

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






