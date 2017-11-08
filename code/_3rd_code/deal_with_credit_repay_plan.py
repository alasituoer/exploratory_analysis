#coding:utf-8
import pandas as pd
import time
import datetime

if __name__ == "__main__":
    working_space = "C:/Users/cherish/Desktop/exploratory_analysis/"
    filename = "credit_repay_plan.txt"
    path_source_file = working_space + "data/_3rd_data/" + filename
    df_repay_plan = pd.read_csv(path_source_file, low_memory=False)
    # 填充finish_date列的缺失, 用程序运行的当天时间+1作为缺失值
    # 注意实际运行时该值从数据库中提取可能会发生变化
    localtime_str = '/'.join([str(x) for x in time.localtime(time.time() + 60*60*24)[:3]])
    df_repay_plan['finish_date'].fillna(localtime_str, inplace=True)
    # 并去掉长格式时间中时分秒的部分, 便于后面做减法得到天数
    df_repay_plan['finish_date'] = df_repay_plan['finish_date'].apply(lambda x: x.split(' ')[0])
    #print df_repay_plan[:20]

    print 'hello'
    # 仅截取想要的字段
    df_repay_plan = df_repay_plan[['order_id', 'repay_corpfine',\
            'repay_date', 'finish_date', 'status',]]
    #df_status_0 = df_repay_plan[df_repay_plan['status']==0]['finish_date'].unique()
    #print df_status_0

    # 计算单子的逾期天数
    #print df_repay_plan[:20]
    series_repay_date = df_repay_plan['repay_date'].apply(lambda t: datetime.datetime.strptime(t, '%Y/%m/%d'))
    series_finish_date = df_repay_plan['finish_date'].apply(lambda t: datetime.datetime.strptime(t, '%Y/%m/%d'))
    series_delta = series_finish_date - series_repay_date
    #print series_delta.apply(lambda d: d.days)[:20]

    # 新建一列追加到df_repay_plan最后一列
    df_repay_plan['ACTUAL_DELAY_DAYS'] = series_delta.apply(lambda d: d.days)
    #print df_repay_plan.head()

    # 输出样本数据中订单的还款逾期情况数据
    path_to_write = working_space + "results/repay_plan_delay.csv"
    df_repay_plan[['order_id', 'ACTUAL_DELAY_DAYS',]].to_csv(path_to_write, index=False)

























