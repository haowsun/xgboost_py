import os
import pandas as pd
import matplotlib.pyplot as plt

threshold_start = 0.0
threshold_end = 1.0
threshold_step_len = 0.01
cls_nums = 13

BASE_PATH = '/Users/sunhaowen/Documents/data/3.ivr_log'
FILE_NAME = 'relay_info_flag_20210408-20210414_95172.xlsx'

FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)
data_frame = pd.read_excel(FILE_PATH)
data_frame = data_frame.loc[:, ['session_id', 'predict_bu_id']]
data_frame = data_frame.set_index('session_id')


BASE_PATH_pred = '/Users/sunhaowen/Documents/data/3.ivr_log/final_data/v1_0'
FILE_NAME_pred = 'compute_fill_acc_rate_20210408-20210414_testttt.xlsx'

FILE_PATH_pred = os.path.join(BASE_PATH_pred, FILE_NAME_pred)
data_frame_pred = pd.read_excel(FILE_PATH_pred)

data_frame_pred = data_frame_pred.loc[:, ['sessionid', 'pred']]
data_frame_pred['pred_label'], data_frame_pred['pred_val'] = data_frame_pred['pred'].str.split('_').str
data_frame_pred = data_frame_pred.loc[:, ['sessionid', 'pred_label', 'pred_val']]
label2buid_dict = {'0': 52, '1': 73, '2': 97, '3': 88, '4': 96, '5': 39, '6': 114, '7': 64, '8': 104, '9': 53, '10': 115, '11': 139, '12': None}
data_frame_pred['pred_label'].replace(label2buid_dict, inplace=True)

data_frame_pred = data_frame_pred.set_index('sessionid')
# print(data_frame_pred['pred_label'])
# print(data_frame_pred)
ori_all_len = len(data_frame)

data_frame_fill = data_frame.loc[data_frame['predict_bu_id'] != '*']

ori_fill_len = len(data_frame_fill)

print(ori_all_len)
print(ori_fill_len)
print(ori_fill_len / ori_all_len)

threshold_tmp = threshold_start
fill_rate_x = []
fill_rate_y = []
while threshold_tmp <= threshold_end:
    pred_correct = 0
    pred_bigthanthreshold = 0

    tmp_data_frame = data_frame.copy(deep=True)
    # print(tmp_data_frame)
    for index, row in data_frame_pred.iterrows():
        # print(index)  # 输出每行的索引值
        # print(row['pred_label'], row['pred_val'])  # 输出每一行
        if float(row['pred_val']) > threshold_tmp:
            # print(tmp_data_frame[index,:])
            tmp_data_frame.loc[index,'predict_bu_id'] = row['pred_label']
    tmp_data_frame_fill = tmp_data_frame.loc[tmp_data_frame['predict_bu_id'] != '*']
    fill_rate_x.append(threshold_tmp)
    fill_rate_y.append(len(tmp_data_frame_fill)/ori_all_len- ori_fill_len / ori_all_len)
    threshold_tmp = threshold_tmp + threshold_step_len

# plt.plot(fill_rate_x, fill_rate_y, marker='o', linestyle='dashed')
fill_rate_y = ['{}pp'.format(str(i*100)) for i in fill_rate_y]
print(fill_rate_y)
# fill_rate_y = "{}pp".format(fill_rate_y)

plt.plot(fill_rate_x, fill_rate_y)
plt.xlabel('threshold')
plt.ylabel('fill_rate')
plt.show()
