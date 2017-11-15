#coding:utf-8
import pandas as pd


def getNeededIndexValueFromOrderInfo(path_file, filename):
    """从cash_order_info中取出所有需要的字段数据"""
    df_order_info = pd.read_csv(\
            path_file + filename, low_memory=False)
    #print df_order_info['status'].unique() #[0,1,2,3,4,6,8]
    df_order_info = df_order_info[['order_id', 'cust_id', 'status',]]
    # 0-待提交 1-审核中 8-订单已取消
    # 2-审核成功 3-审核失败 4-放款成功 5-放款失败
    # 6-还款成功 7-还款失败
    # 筛选出订单经过审核的部分(不管是否通过)
    df_order_reviewed = df_order_info[df_order_info['status']!=0]
    df_order_reviewed = df_order_reviewed[df_order_reviewed['status']!=1]
    df_order_reviewed = df_order_reviewed[df_order_reviewed['status']!=8]
    #print df_order_reviewed.head()
    #print df_order_reviewed['status'].unique() #[2,3,4,6]

    path_to_write = path_file +\
            "lt_order_id_and_cust_id_reviewed.csv"
    df_order_reviewed.to_csv(path_to_write, index=False)

def getPhoneNumberFromCustomerBaseInfo(path_file, filename):
    """从customer_base_info中获取(cust_)id与mobile的对照表"""
    df_cust_base_info = pd.read_csv(path_file + filename)
    df_cust_base_info = df_cust_base_info[['id', 'mobile',]]
    #print df_cust_base_info.head()

    path_to_write = path_file +\
            "lt_cust_id_and_phone_number_all.csv"
    df_cust_base_info.to_csv(path_to_write, index=False)

def matchOrderIdAndPhoneNumber(\
        path_file1, file1name, path_file2, file2name):
    """将order_info_reviewed中的订单号都匹配上手机号
        后倒入数据库匹配上电话详单的有限指标(B C D U)数据"""
    df_order_id = pd.read_csv(path_file1 + file1name)
    df_phone_number = pd.read_csv(path_file2 + file2name)

    df_merged = pd.merge(df_order_id, df_phone_number,\
            left_on='cust_id', right_on='id', how='left')
    #print df_merged[:20]

    path_to_write = path_file1 + "lt_order_id_and_phone_number.csv"
    df_merged.to_csv(path_to_write, index=False)



if __name__ == "__main__":
    print 'hello alas'
    working_space = "C:/Users/cherish/Desktop/exploratory_analysis/"

    # 将order_info_reviewed中的订单号都匹配上手机号
    # 后倒入数据库匹配上电话详单的有限指标(B C D U)数据
    # 有order_id 与 cust_id 对照关系的文件
#    path_file_order_id = working_space + "data/_3rd_data/"
#    filename_order_id = "lt_order_id_and_cust_id_reviewed.csv"
    # 有cust_id 与 mobile 对应关系的文件
#    path_file_phone_number = working_space + "data/_3rd_data/"
#    filename_phone_number = "lt_cust_id_and_phone_number_all.csv"
#    matchOrderIdAndPhoneNumber(\
#            path_file_order_id, filename_order_id,\
#            path_file_phone_number, filename_phone_number)



    #从customer_base_info中获取(cust_)id与mobile的对照表
    #path_file_cust_base_info = working_space + "data/_3rd_data/"
#    filename_cust_base_info = "customer_base_info_20171110.txt"
#    getPhoneNumberFromCustomerBaseInfo(\
#            path_file_cust_base_info, filename_cust_base_info)


    # 从cash_order_info中取出所有需要的字段数据
    #path_file_order_info = working_space + "data/_3rd_data/"
#    filename_order_info = "cash_order_info_20171110.txt"
#    getNeededIndexValueFromOrderInfo(\
#            path_file_order_info,filename_order_info)
    


