#%%
#coding:utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 参数
version = 'v1_0'
date = '20210415-20210421'
dataset = 'all'

# 路径
BASE_PATH = '/Users/sunhaowen/Documents/data/merge_relay_info'
FILE_NAME = 'merge_relay_info_flag_{}_95172.xlsx'.format(date)

BASE_PATH_pred = '/Users/sunhaowen/Documents/data/3.ivr_log/final_data/{}'.format(version)
FILE_NAME_pred = 'compute_fill_acc_rate_{}_{}.xlsx'.format(date, dataset)

OUT_BASE_PATH = '/Users/sunhaowen/Documents/data/3.ivr_log/final_data/{}/route_acc'.format(version)
OUT_FILE_NAME_data = 'route_acc_{}.xlsx'.format(date)
OUT_FILE_NAME_img = 'route_acc_{}.png'.format(date)

# merge_relay_info文件读入
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)
merge_relay_info = pd.read_excel(FILE_PATH)

# merge_relay_info保留语音机器人 和 所需要的列数据
merge_relay_info_bot = merge_relay_info.loc[merge_relay_info['按键类型'] == '语音机器人']
merge_relay_info_bot = merge_relay_info_bot.loc[:, ['session_id','predict_bu_id', '一级FAQID', '是否路由准确']]
merge_relay_info_bot = merge_relay_info_bot.set_index('session_id')

# 预测数据读入
FILE_PATH_pred = os.path.join(BASE_PATH_pred, FILE_NAME_pred)
data_frame_pred = pd.read_excel(FILE_PATH_pred)

data_frame_pred = data_frame_pred.loc[:, ['sessionid', '一级FAQID','pred']]
data_frame_pred['pred_label'], data_frame_pred['pred_val'] = data_frame_pred['pred'].str.split('_').str
data_frame_pred = data_frame_pred.loc[:, ['sessionid', '一级FAQID', 'pred_label', 'pred_val']]
label2buid_dict = {'0': 52, '1': 73, '2': 97, '3': 88, '4': 96, '5': 39, '6': 114, '7': 64, '8': 104, '9': 53, '10': 115, '11': 139, '12': None}
data_frame_pred['pred_label'].replace(label2buid_dict, inplace=True)
data_frame_pred = data_frame_pred.set_index('sessionid')
#%%
# 计算路由准确率
def compute_fill_acc(my_df):
    all_num_bot = len(my_df)
    route_correct_num_bot = 0
    for index, row in my_df.iterrows():
        if row['是否路由准确'] == 1:
            route_correct_num_bot += 1
    return route_correct_num_bot, all_num_bot, route_correct_num_bot/all_num_bot
#%%
# 计算原始路由准确率
_,_, ori_route_acc = compute_fill_acc(merge_relay_info_bot)
ori_route_acc
#%%
# 按阈值计算路由准确率
threshold_start = 0.0
threshold_end = 1.0
threshold_step_len = 0.01

fill_rate_x = []
fill_rate_y = []
threshold_tmp = threshold_start
while threshold_tmp <= threshold_end:
    pred_correct = 0
    pred_bigthanthreshold = 0

    tmp_data_frame = merge_relay_info_bot.copy(deep=True)
    for index, row in data_frame_pred.iterrows():
        if float(row['pred_val']) > threshold_tmp:
            tmp_data_frame.loc[index,'predict_bu_id'] = str(row['pred_label'])
            tmp_data_frame.loc[index,'是否路由准确'] = 1 if tmp_data_frame.loc[index,'predict_bu_id'] == str(tmp_data_frame.loc[index,'一级FAQID']) else 0
    tmp_fill_acc = compute_fill_acc(tmp_data_frame)[2]
    fill_rate_x.append(threshold_tmp)
    fill_rate_y.append(tmp_fill_acc-ori_route_acc)
    threshold_tmp = threshold_tmp + threshold_step_len


#%%
# 画图
fill_rate_y = [i*100 for i in fill_rate_y]
plt.plot(fill_rate_x, fill_rate_y)
plt.xlabel('阈值')
plt.ylabel('路由准确率影响(pp)')
# plt.show()
plt.savefig(os.path.join(OUT_BASE_PATH, OUT_FILE_NAME_img), bbox_inches='tight')

# %%
# 输出文件
res_df = pd.DataFrame({'threshold': fill_rate_x, 'sub': fill_rate_y})
res_df.to_excel(os.path.join(OUT_BASE_PATH, OUT_FILE_NAME_data), index=False)