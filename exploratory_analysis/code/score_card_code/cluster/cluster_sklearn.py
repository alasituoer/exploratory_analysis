#coding:utf-8
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier


#from data import removed_list
from data import removed_list
from data import qualitative_variables_list
from data import tel_detail_info_index_list

def pcaTelDetailInfo(df_tel_detail_info):
    df_tel_detail_info.fillna(df_tel_detail_info.mean(), inplace=True)
    array_tel_detail_info = df_tel_detail_info.values
    array_std_detail_info = Normalizer().fit_transform(array_tel_detail_info)
    #print array_tel_detail_info.shape
    #print array_tel_detail_info
    print array_std_detail_info.shape
    #print array_std_detail_info

    pca = PCA(n_components='mle')
    #pca = PCA(n_components=5)
    print pca.fit_transform(array_std_detail_info)
    #print pca.components_
#    print 'explained_variance_ratio_', pca.explained_variance_ratio_#.sum()
    #print 'explained_variance_', pca.explained_variance_
    print 'n_components', pca.n_components_

def selectTelDetailInfo(df_tel_detail_info):
    df_tel_detail_info.fillna(df_tel_detail_info.mean(), inplace=True)
    y = df_tel_detail_info['ovd_daynum'].values
    y = y.reshape(len(y),1)
    del df_tel_detail_info['ovd_daynum']
    X = df_tel_detail_info.values
    stded_x = Normalizer().fit_transform(X)
    stded_y = Normalizer().fit_transform(y)

    #print stded_x.shape
    #print stded_x
    #print stded_y.shape
    #print stded_y
    #x_new = SelectKBest().fit_transform(stded_x, stded_y)
    #print x_new.shape
    #print x_new

    clf = ExtraTreesClassifier()
    clf.fit(stded_x, stded_y.ravel())
    print clf.feature_importances_
#    print clf.decision_path(stded_x)



if __name__ == "__main__":
    working_space = "/mnt/hgfs/windows_desktop/exploratory_analysis/" +\
		      "data/dataset_score_card/cleaned_full_data/test/"
    filename = "coll_dataset2_test.txt"
    #filename = "coll_dataset2.txt"

    df = pd.read_csv(working_space + filename, index_col=0)
    list_df_columns = list(df.columns)
    for i in removed_list:
	list_df_columns.remove(i)
    df = df[list_df_columns]

    df_tel_detail_info = df[['ovd_daynum'] + tel_detail_info_index_list]
    selectTelDetailInfo(df_tel_detail_info)

    #df_tel_detail_info = df[tel_detail_info_index_list]
    #pcaTelDetailInfo(df_tel_detail_info)





def tryScalerData(filename):

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
    







