#coding:utf-8
import pandas as pd

def writeAsCsvWithOrderSuccess(path_file, status_value, path_to_write):
    """将状态为status_value的订单保存为一个CSV文件"""
    df_cash_order_info = pd.read_csv(path_filename, low_memory=False)
    df_order_success = df_cash_order_info[df_cash_order_info['status']==status_value]
    df_order_success.to_csv(path_to_write, sep=',', index=False)


#list_cust_id = list(df_cash_order_info['cust_id'].unique())
#print type(list_cust_id[:10][0])

#df_cash_order_info_one_cust = df_cash_order_info[df_cash_order_info['cust_id']==3]
#print df_cash_order_info_one_cust[['audit_time', 'payment_time', 
#    'repayment_time', 'status', 'reject_reason',]]

#print df_order_success



if __name__ == "__main__":
    working_space = "C:/Users/cherish/Desktop/exploratory_analysis/"
    filename = "data/_3rd_data/cash_order_info.txt"
    path_sourcefile = working_space + filename
    df_cash_order_info = pd.read_csv(path_sourcefile, low_memory=False)
    print 'h'
    print len(df_cash_order_info['repayment_type'])
    for i in df_cash_order_info['repayment_type'].unique():
        print len(df_cash_order_info[df_cash_order_info["repayment_type"]==i])

"""
    for status_value in [0, 1, 2, 3,]:
        path_to_write = working_space + "results/cash_order_info_status_" + \
                str(status_value) + "_success.csv"
        writeAsCsvWithOrderSuccess(working_space + path_filename, status_value, path_to_write)
"""

    #path_filename = working_space +"results/cash_order_info_success.csv"
    #df_order_success = pd.read_csv(path_filename)
    #print df_order_success.head()
    #print df_order_success["status"].unique()









