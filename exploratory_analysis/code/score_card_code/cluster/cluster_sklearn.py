#coding:utf-8
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

#from data import removed_list
from data import removed_list
from data import qualitative_variables_list
#from data import fillna_list

if __name__ == "__main__":
    working_space = "/mnt/hgfs/windows_desktop/exploratory_analysis/" +\
		      "data/dataset_score_card/cleaned_full_data/test/"
    filename = "coll_dataset2_test.txt"
    df = pd.read_csv(working_space + filename, index_col=0)
    list_df_columns = list(df.columns)
    for i in removed_list:
	list_df_columns.remove(i)
    df = df[list_df_columns]
#    print df.head()
#    print df.describe()

    df_test = df#[qualitative_variables_list]
    # 处理定性特征缺失值为'NA', 单独为一类
    # 对定性特征标准化
    for i in qualitative_variables_list:
	integer_encoded = LabelEncoder().fit_transform(df_test[i])
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	#print integer_encoded
	# 需不需要对定性变量哑编码(减少信息损失)
	#onehot_encoder = OneHotEncoder(sparse=False)
	#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	#print onehot_encoded
	df_test[i] = integer_encoded
    #print df_test
    
    df_test.dropna(axis=0, how='any', inplace=True)
    #print df_test

    #print df_test.as_matrix()
    print df_test.values
    #mat_test = StandardScaler().fit_transform(df_test.values)
    mat_test = Normalizer().fit_transform(df_test.values)
    print mat_test.shape
    print sum(mat_test.T)
    







