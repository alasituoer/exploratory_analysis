#coding:utf-8
import pandas as pd

if __name__ == "__main__":
    working_space = "C:/Users/cherish/Desktop/exploratory_analysis/"
    filename1 = "repay_plan_delay.csv"
    path_repay_plan_delay = working_space + 'results/' + filename1

    print 'hello'
    df_delay = pd.read_csv(path_repay_plan_delay)
    #print df_delay

    #print len(df_delay[df_delay['ACTUAL_DELAY_DAYS']>0])
    #print len(df_delay[df_delay['ACTUAL_DELAY_DAYS']<=0])

    # 根据order_id提取cash_order_info中的订单详情数据
    filename2 = "cash_order_info.txt"
    path_cash_order_info = working_space + 'data/_3rd_data/' + filename2
    df_cash_order_info = pd.read_csv(path_cash_order_info, low_memory=False)
    #print df_cash_order_info.head()

    path_to_write = working_space + "results/order_info_delay.csv"
    with open(path_to_write, 'w') as f1:
        for i in df_delay['order_id'][:3]:
            list_to_strings = [str(x) for x in\
                    df_cash_order_info[df_cash_order_info[
                        'order_id']==i].values[0]]
            strings_to_write = ','.join(list_to_strings)
            strings_to_write += '\n'
            f1.write(strings_to_write)









