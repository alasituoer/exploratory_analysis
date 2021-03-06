#coding:utf-8
# 不需要的指标
removed_features =\
	[#时间点类型数据以及明显不用加入的指标
	'cust_id', 'phone_no', 'id_no', 'name',
	'payment_date', 'apply_time', 'audit_time', 'payment_time',
	'repayment_time', 'reject_reason', 'not_get_fetch_detail',
	'repayment_type', 'pay_channel', 'created_by', 'created_time',
	'last_modified_by', 'last_modified_time', 'upd_dt', 'order_source',
	'st_date', 'min_date', 'now_city_code', 'work_area',
	'is_sms', 'is_contacts', 'refuseMessage', 'max_ovd',
	# 因缺失数量太多(且填充其他数值具有误导性)而去掉的指标
	'id_type', 'reason_code', 'institu_amt', 'bnk_amt', 'cnss_amt', 
	'p2p_amt', 'query_amt', 'query_amt_m3', 'query_amt_m6',
	# 离散型变量分类不准确导致过多
	'H1',
	# 完全重复特征, 名称不同
	'live_city.work_city', 'live_city.mobile_city', 'live_city.ec2_city', 
	'now_city.live_city', 'now_city.id_city', 'now_city.ec1_city', 
	'now_city.bk_mob_city', 'work_city.now_city', 'work_city.mobile_city', 
	'work_city.ec2_city', 'id_city.live_city', 'id_city.work_city', 
	'id_city.ec1_city', 'id_city.bk_mob_city', 'mobile_city.now_city', 
	'mobile_city.id_city', 'mobile_city.ec2_city', 'ec1_city.live_city', 
	'ec1_city.work_city', 'ec1_city.mobile_city', 'ec1_city.bk_mob_city', 
	'ec2_city.now_city', 'ec2_city.id_city', 'ec2_city.ec1_city', 
	'bk_mob_city.live_city', 'bk_mob_city.work_city', 'bk_mob_city.mobile_city', 
	'bk_mob_city.ec2_city', 'live_area.work_area', 'now_area.live_area', 
	'now_area.id_area', 'work_area.now_area', 'id_area.live_area', 
	'id_area.work_area',]


	

# 离散型特征
discrete_features =\
	['amount', 'term', 'rate', 'service_fee', 'payment_amount',
	'status', 'reloan', 'source', 'channel', 'id_province',
	'now_province', 'work_province', 'gender', 'industry', 
	'realname_isme', 'address_contact', 'coll_result',

	'live_city.now_city', 'live_city.id_city', 'live_city.ec1_city',
	'live_city.bk_mob_city','now_city.work_city', 'now_city.mobile_city',
	'now_city.ec2_city', 'work_city.live_city', 'work_city.id_city',
	'work_city.ec1_city', 'work_city.bk_mob_city', 'id_city.now_city',
	'id_city.mobile_city', 'id_city.ec2_city', 'mobile_city.live_city',
	'mobile_city.work_city', 'mobile_city.ec1_city', 'mobile_city.bk_mob_city',
	'ec1_city.now_city', 'ec1_city.id_city', 'ec1_city.ec2_city',
	'ec2_city.live_city', 'ec2_city.work_city', 'ec2_city.mobile_city',
	'ec2_city.bk_mob_city', 'bk_mob_city.now_city', 'bk_mob_city.id_city',
	'bk_mob_city.ec1_city', 'live_area.now_area', 'live_area.id_area',
	'now_area.work_area', 'work_area.live_area', 'work_area.id_area',
	'id_area.now_area',]

    
# 客户地址信息
cust_address_list =\
        ['live_city.now_city', 'live_city.work_city', 'live_city.id_city',
        'live_city.mobile_city', 'live_city.ec1_city', 'live_city.ec2_city',
        'live_city.bk_mob_city',
        'now_city.live_city', 'now_city.work_city', 'now_city.id_city',
        'now_city.mobile_city', 'now_city.ec1_city', 'now_city.ec2_city',
        'now_city.bk_mob_city',
        'work_city.live_city', 'work_city.now_city', 'work_city.id_city',
        'work_city.mobile_city', 'work_city.ec1_city', 'work_city.ec2_city',
        'work_city.bk_mob_city',
        'id_city.live_city', 'id_city.now_city', 'id_city.work_city',
        'id_city.mobile_city', 'id_city.ec1_city', 'id_city.ec2_city', 
        'id_city.bk_mob_city',
        'mobile_city.live_city', 'mobile_city.now_city', 'mobile_city.work_city',
        'mobile_city.id_city', 'mobile_city.ec1_city', 'mobile_city.ec2_city',
        'mobile_city.bk_mob_city',
        'ec1_city.live_city', 'ec1_city.now_city', 'ec1_city.work_city',
        'ec1_city.id_city', 'ec1_city.mobile_city', 'ec1_city.ec2_city',
        'ec1_city.bk_mob_city',
        'ec2_city.live_city', 'ec2_city.now_city', 'ec2_city.work_city',
        'ec2_city.id_city', 'ec2_city.mobile_city', 'ec2_city.ec1_city',
        'ec2_city.bk_mob_city',
        'bk_mob_city.live_city', 'bk_mob_city.now_city', 'bk_mob_city.work_city',
        'bk_mob_city.id_city', 'bk_mob_city.mobile_city', 'bk_mob_city.ec1_city',
        'bk_mob_city.ec2_city',
        'live_area.now_area', 'live_area.work_area', 'live_area.id_area',
        'now_area.live_area', 'now_area.work_area', 'now_area.id_area',
        'work_area.live_area', 'work_area.now_area', 'work_area.id_area',
        'id_area.live_area', 'id_area.now_area', 'id_area.work_area',]



# 客户基本信息+app+第三方有效指标
cust_app_3rd_index_list =\
	['face_compare', 'zhima_score', 'age', #'institu_amt', 'bnk_amt', 
	'cnss_amt', 'p2p_amt', 'query_amt', 'query_amt_m3', 'query_amt_m6', 
	'dialing_count', 'called_count', 'contact_count', 'black_circle_count', 
	'e5_value_count', 'calllog_trad_count', 'collection_count', 
	'public_organ_count', 'dialing_one', 'address_black_count', 
	'address_collection_count', 'sms_fail_count', 'phoneToptenCount', 
	'address_count', 'apply_fail_count2', 'trade_register_count', 
	'last_usetime_cha',]


# 电话详单有效指标
tel_detail_info_index_list = \
	['I11_1', 'I11_2', 'I11_3', 'I11_4', 'I11_5',
	'C1', 'C2', 'C3', 'C4', 'C5',
	'C6_1', 'C6_2', 'C6_3', 'C6_4',
	'C7_1', 'C7_2', 'C7_3', 'C7_4',
	'C8_1', 'C8_2', 'C8_3',
	'C9', 'C10', 'C11',
	'U1', 'U2', 'U3', 'U4', 'U5',
	'D1_1', 'D1_2', 'D1_3', 'D1_4', 'D1_5',
	'D2_1', 'D2_2', 'D2_3', 'D2_4', 'D2_5',
	'D3_1', 'D3_2', 'D3_3', 'D3_4', 'D3_5',
	'D4', 'D5', 'D6', 'D7', 'D8',
	'D9_1', 'D9_2', 'D9_3', 'D9_4',
	'D10_1', 'D10_2', 'D10_3', 'D10_4',
	'D11_1', 'D11_2', 'D11_3',
	'D12', 'D13', 'D14',
	'E1_1', 'E1_2', 'E1_3', 'E1_4', 'E1_5',
	'E2_1', 'E2_2', 'E2_3', 'E2_4', 'E2_5',
	'E3_1', 'E3_2', 'E3_3', 'E3_4', 'E3_5',
	'E4', 'E5_1', 'E5_2', 'E5_3', 'E5_4', 'E5_5',
	'F1', 'F2', 'F3', 'F5',
	'F6_1', 'F6_2', 'F6_3', 'F6_4', 'F6_5',
	'F7', 'F8_1', 'F8_2', 'F8_3', 'F8_4',
	'F9_1', 'F9_2', 'F9_3', 'F9_4', 'F9_5',
	'G1', 'G2', 'H2', 'H3', 'H4',
	'I1_1', 'I1_2', 'I1_3', 'I1_4', 'I1_5',
	'I2_1', 'I2_2', 'I2_3', 'I2_4', 'I2_5',
	'I3', 'I4', 'I5',]

# 拟选择的特征
#features_selected =\
#	['ovd_daynum', 'F3', 'F5', 'G1', 'I4', 'zhima_score', 
#	'age', 'contact_count', 'collection_count', 'dialing_one',
#	'address_black_count', 'sms_fail_count',]
#features_selected =\
#	['ovd_daynum', 'F3', 'I4', 'age', 'collection_count', 'sms_fail_count',]
features_selected =\
	['F3', 'I4', 'age', 'contact_count',]
features_will_be_selected =\
	['ovd_daynum', 'I11_2', 'C8_2', 'D2_3', 'D2_5', 'D3_1', 'D3_5', 
	'D7', 'D8', 'D9_1', 'D9_2', 'D9_4', 'D10_2', 'E2_1', 'E3_1', 
	'E3_4', 'E3_5', 'E5_1', 'F3', 'F5', 'F6_1', 'F6_4', 'F7', 
	'F8_1', 'F8_2', 'F9_1', 'G1', 'H2', 'H3', 'I1_1', 'I1_3', 'I2_1', 
	'I2_2', 'I2_3', 'I3', 'I4', 
	'face_compare', 'zhima_score', 'age', 'contact_count', 
	'calllog_trad_count', 'collection_count', 'dialing_one', 
	'address_black_count', 'sms_fail_count', 'phoneToptenCount', 
	'address_count', 'apply_fail_count2', 'trade_register_count', 
	'last_usetime_cha',]
# random lasso
selected_tel_detail_info_index_list =\
	['E3_4', 'F5', 'I2_3', 'I2_4', 'E5_1', 'F6_4', 'E3_5', 
	'E3_1', 'C3', 'E2_1', 'F2', 'F3', 'F7', 'H3', 'I1_3', 
	'I1_1', 'I3', 'F9_5', 'F9_1', 'I4', 'D8', 'D9_4', 
	'D9_1', 'D7', 'I2_2', 'F9_4', 'D2_3', 'D3_1', 'D3_5', 
	'C8_3', 'F8_1', 'F8_2', 'G1', 
	'D9_2', 'I1_2', 'I1_4', 'F6_3', 'E5_3', 'I11_1', 'D1_1', 
	'C6_1', 'E3_2', 'D1_2', 'C8_2', 'I2_1', 'C1', 'D11_2', 
	'D2_5', 'E5_2', 'C4', 'D2_1']



