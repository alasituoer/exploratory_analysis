#coding:utf-8
import numpy as np
import pandas as pd

def t_cluster(path_file, filename):
    df = pd.read_csv(path_file + filename)
    print df.head(n=10)
    print "J"

def testSklearn():
    # imput sample data
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data[:5]
    np.random.seed(0)
    #data = np.random.randint(30, 100, 20).reshape(4,5)*1.0
    #data = iris.target
    #print data.reshape((-1, 1))


    from sklearn.preprocessing import Imputer
    #imputer = Imputer()
    #print imputer.fit_transform(np.vstack(((np.array([np.nan]*4)), data)))
    #imputer.fit_transform()


    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    """
    data = np.array(['chrome', 'firefox', 'ie'])
    print data
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    print integer_encoded
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print onehot_encoded
    inverted = label_encoder.inverse_transform([
        np.argmax(onehot_encoded[0,:])])
    print inverted
    """

    from sklearn.preprocessing import StandardScaler
    #standard_data = StandardScaler().fit_transform(data)
    #print standard_data
    #print(StandardScaler().fit_transform(standard_data))

    from sklearn.preprocessing import MinMaxScaler
    #minmax_scaler_data = MinMaxScaler().fit_transform(data)
    #print minmax_scaler_data

    from sklearn.preprocessing import Normalizer
    #normalizer_data = Normalizer().fit_transform(data)
    #print normalizer_data

    from sklearn.preprocessing import Binarizer
    #binarizer_data = Binarizer(threshold=2).fit_transform(data)
    #print binarizer_data



def descFillNA(path_file, filename):
    df_splited_detail_info = pd.read_csv(path_file + filename)
    #del df_splited_detail_info['UPDATE_DT']
    #print df_splited_detail_info.head(n=10)

    # 缺失值统计及处理
    #print df_splited_detail_info.describe().ix['count']*1.0/44163
    #print 1-df_splited_detail_info.describe().ix['count']*1.0/44163
    print df_splited_detail_info.describe()
    #print df_splited_detail_info[df_splited_detail_info.columns[4:]].mean()
    df_splited_detail_info.fillna(df_splited_detail_info.mean(),\
            inplace=True)
    print df_splited_detail_info.describe()
    df_splited_detail_info.to_csv(path_file +\
            "detail_info_filledna.csv", index=False)

    # 调用另一函数处理异常值, 返回处理好的DataFrame
    #dealWithOutliers(df_splited_detail_info)
    #df_all_new = dealWithOutliers(df_splited_detail_info)
    #print df_all_new.describe()
    #df_all_new.to_csv(path_file + "cleaned_detail_info.csv", index=False)


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
            ['phone_no',], keep='last')
    print len(df_needed_detail_info)
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

def explainOutliers(df_to_do_outliers):
    print df_to_do_outliers.describe()

    list_df_one_cat = []
    for v in df_to_do_outliers.columns[3:]:
        df_one_cat = df_to_do_outliers[[v]]
        all_counts = len(df_one_cat)
#        print v

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
#        print '\n'

        # 在已知extreme outliers proportion均小于0.05的条件下
        if Q3_mild_outliers_proportion > 0.05:
            df_one_cat[df_one_cat[v] > Q3+3*IQR] = Q3+3*IQR
#            df_one_cat[df_one_cat[v] > Q3+1.5*IQR and\
#                    df_one_cat[v] < Q3+3*IQR] = Q3+1.5*IQR
        else:
            df_one_cat[df_one_cat[v] > Q3+1.5*IQR] = Q3+1.5*IQR
        if Q1_mild_outliers_proportion > 0.05:
            df_one_cat[df_one_cat[v] < Q1-3*IQR] = Q1-3*IQR
#            df_one_cat[df_one_cat[v] > Q1-3*IQR and\
#                    df_one_cat[v] < Q1-1.5*IQR] = Q1-1.5*IQR
        else:
            df_one_cat[df_one_cat[v] < Q1-1.5*IQR] = Q1-1.5*IQR

        list_df_one_cat.append(df_one_cat)
    df_value_new = pd.concat(list_df_one_cat, axis=1)
    print df_value_new.describe()

def dealWithOutliers(df_splited_detail_info):
    df_splited_detail_info.index = df_splited_detail_info['order_id']
    del df_splited_detail_info['order_id']
    #print df_splited_detail_info.head()

    explainOutliers(df_splited_detail_info)



def printDescOfDataFrame(df_sample):
    pass


if __name__ == "__main__":
    working_space = "C:/Users/Administrator/Desktop/exploratory_analysis/"
    path_file = working_space + "data/dataset_score_card/"
    filename = "splited_detail_info.csv"
    df = pd.read_csv(path_file + filename)
    print len(df)
    print df.describe('id')


