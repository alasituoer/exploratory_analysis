#coding:utf-8
import pandas as pd
import time, datetime
import numpy as np

def getCustomerIdAndMobilePhone(path_file, filename):
    """get cust_id and mobile_phone from the table customer_base_info"""
    df_full = pd.read_csv(path_file + filename)
    df_needed = df_full[['id', 'mobile',]]
    #print df_needed.head()

    path_to_write = path_file + "cust_id_mobile_phone_from_" + filename[:-4] + ".csv"
    df_needed.to_csv(path_to_write, index=False)


def computeDelayDaysOfRepayPlan(path_file, filename):
    """compute delay days of the orders having the plan of repaymemt"""
    print 'alas'
    df_repay_plan = pd.read_csv(path_file + filename, low_memory=False)

    #以统计日期的后一天填充缺失的finish_date
    statistical_date = '2017/11/8'
    unix_statistical_date = time.mktime(time.strptime(statistical_date, '%Y/%m/%d'))
    fillna_date = '/'.join([str(x) for x in\
            time.localtime(unix_statistical_date + 60*60*24)[:3]])
    df_repay_plan['finish_date'].fillna(fillna_date, inplace=True)

    # 截取日期的年月日, 不要时分秒
    df_repay_plan['finish_date'] =\
            df_repay_plan['finish_date'].apply(lambda x: x.split(' ')[0])
    # 提取需要的字段计算逾期天数
    df_needed = df_repay_plan[['order_id', 'repay_corpfine',
                                'repay_date', 'finish_date', 'status',]]

    #print df_needed.head(), len(df_needed)
    # 统计日之后的日期逻辑上为缺失值, 即替换为统计值后一天的日期值
    df_2 = df_needed[df_needed['finish_date'].apply(lambda t:\
            datetime.datetime.strptime(t, "%Y/%m/%d")) <= \
            datetime.datetime.strptime('2017/11/8', "%Y/%m/%d")]
    #print df_2.head(), len(df_2)

    df_needed['finish_date'] =\
            df_needed['finish_date'].replace('2017/11/10', '2017/11/9')

    #df_3 = df_needed[df_needed['finish_date'].apply(lambda t:\
    #        datetime.datetime.strptime(t, "%Y/%m/%d")) == \
    #        datetime.datetime.strptime('2017/11/10', "%Y/%m/%d")]
    #print len(df_3)

    #compute the delay days
    series_repay_date = df_needed['repay_date'].apply(lambda t:\
            datetime.datetime.strptime(t, "%Y/%m/%d"))
    series_finish_date = df_needed['finish_date'].apply(lambda t:\
            datetime.datetime.strptime(t, "%Y/%m/%d"))
    series_date_delta = series_finish_date - series_repay_date
    df_needed['delay_days'] = series_date_delta.apply(lambda d: d.days)
    #print df_needed.head()
    #print len(df_needed)

    path_to_write = path_file +\
            "delay_days_order_id_from_" + filename[:-4] + '.csv'
    df_delay_days = df_needed[df_needed['delay_days']>=0]
    df_delay_days.to_csv(path_to_write, index=False)

def getMobileByCustidByOrderid(path_file_order_id, filename_order_id,\
        path_file_cust_id, filename_cust_id,\
        path_file_mobile, filename_mobile):
    """get customer id from the file cash_order_info_20171110.txt by order id"""
    print 'hello'
    df_order_id = pd.read_csv(path_file_order_id + filename_order_id)
    df_cust_id = pd.read_csv(path_file_cust_id + filename_cust_id, low_memory=False)
    df_mobile = pd.read_csv(path_file_mobile + filename_mobile)
    print 'needed_order_id:', len(df_order_id)
    print 'all_cust_id', len(df_cust_id[['order_id', 'cust_id',]])
    print 'all_mobile:', len(df_mobile)

    path_to_write =  path_file_order_id + "cust_id_order_id.csv"
    with open(path_to_write, 'w') as f1:
        for i in df_order_id['order_id']:
            # 写入订单号
            strings_to_write = i + ',' + str(df_order_id[df_order_id[
                    'order_id']==i]['delay_days'].values[0]) + ','
            # 写入客户编号
            try:
                cust_id = df_cust_id[df_cust_id['order_id']==i
                    ]['cust_id'].values[0]
                strings_to_write += str(cust_id)
            except Exception, e:
                strings_to_write += ''
            strings_to_write += ','
            # 写入客户手机号
            try:
                mobile = df_mobile[df_mobile['id']==cust_id
                    ]['mobile'].values[0]
                strings_to_write += str(mobile)
            except Exception, e:
                strings_to_write += ''
            strings_to_write += '\n'
            f1.write(strings_to_write)

def getTelephoneDetailInfoHavingRepayPlan(path_file, filename):
    df_telephone_detail_info = pd.read_csv(path_file + filename)
    df_needed = df_telephone_detail_info[
            ['B1_VALUE', 'B3_VALUE', 'B4_VALUE', 'C1_VALUE', 'C2_VALUE', 'C3_VaLUE',
            'C4_VALUE', 'C5_VALUE', 'C6_VALUE', 'C7_VALUE', 'C8_VALUE', 'C9_VALUE', 
            'C10_VALUE', 'C11_VALUE', 'D1_VALUE', 'D2_VALUE', 'D3_VALUE', 'D4_VALUE',
            'D5_VALUE', 'D6_VALUE', 'D7_VALUE', 'D8_VALUE', 'D9_VALUE', 'D10_VALUE',
            'D11_VALUE', 'D12_VALUE', 'D13_VALUE', 'D14_VALUE', 'Ui_VALUE',]]
    print df_needed.head()


if __name__ == "__main__":
    working_space = "C:/Users/cherish/Desktop/exploratory_analysis/"

    #通过mobile 在risk_ctrl_identificaiton_20171110.txt中
    #得到电话详单部分的指定指标数据
    path_file_risk_ctrl_identification =\
            working_space + "data/_2rd_data/"
    filename_risk_ctrl_identification = "risk_ctrl_identification_20171109.txt"
    getTelephoneDetailInfoHavingRepayPlan(path_file_risk_ctrl_identification,\
            filename_risk_ctrl_identification)







    #通过order_id 在cash_order_info_20171110.txt中得到cust_id
    #通过cust_id 在customer_base_info_20171110.txt中得到mobile
"""
    path_file_order_id = working_space + "data/_2rd_data/"
    filename_order_id = "delay_days_order_id_from_credit_repay_plan_20171110.csv"
    path_file_cust_id = working_space + "data/_2rd_data/"
    filename_cust_id = "cash_order_info_20171110.txt"
    path_file_mobile = working_space + "data/_2rd_data/"
    filename_mobile = "customer_base_info_20171110.txt"
    getMobileByCustidByOrderid(path_file_order_id, filename_order_id,\
            path_file_cust_id, filename_cust_id,\
            path_file_mobile, filename_mobile)
"""



#    compute delay days of the orders having the plan of repaymemt
#    and save the result as a csv file
    #path_file_credit_repay_plan = working_space + "data/_2rd_data/"
    #filename_credit_repay_plan = "credit_repay_plan_20171110.txt"
    #computeDelayDaysOfRepayPlan(path_file_credit_repay_plan,\
    #                            filename_credit_repay_plan)

#    get cust_id and mobile_phone from the table customer_base_info
#    and save as a csv file
    #path_file_customer_base_info = working_space + "data/_2rd_data/"
    #filename_customer_base_info = "customer_base_info_20171110.txt"
    #getCustomerIdAndMobilePhone(path_file_customer_base_info,\
    #                            filename_customer_base_info)

    





