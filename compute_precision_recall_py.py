
import os
import numpy as np
import pandas as pd


def compute_p_r(f_pred, threshold_start, threshold_end, threshold_step_len):
	pred_lines = f_pred.readlines()
	pred_lines = np.array(pred_lines)

	label_list = []
	preds_res_list = []
	for line in pred_lines:
		line_sp = line.split('\t')
		label_list.append(line_sp[0])
		pred = line_sp[1]
		pred = pred.split('_')
		preds_res_list.append([pred[0], float(pred[1])])

	all_num = len(label_list)

	threshold_tmp = threshold_start

	df = pd.DataFrame(columns=['threshold', 'P', 'R', 'F1'])

	while threshold_tmp <= threshold_end:
		pred_correct = 0
		pred_bigthanthreshold = 0

		# print(len(label_list))
		# print(len(preds_res_list))
		if len(label_list) == len(preds_res_list):
			for i in range(len(preds_res_list)):
				# print(preds_res_list[i][1], threshold_tmp)
				if preds_res_list[i][1] > threshold_tmp:
					pred_bigthanthreshold += 1
					# print(label_list[i], preds_res_list[i][0])
					if label_list[i] == preds_res_list[i][0]:
						pred_correct += 1
		Precision = pred_correct / pred_bigthanthreshold if pred_bigthanthreshold != 0 else 0
		Recall = pred_correct / all_num
		F1_Score = (2 * Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0
		data_row = {'threshold': threshold_tmp, 'P': Precision, 'R': Recall, 'F1': F1_Score}
		df = df.append(data_row, ignore_index=True)
		# print('t:', threshold_tmp)
		# print('Precision:', Precision)
		# print('Recall:', Recall)
		# print('F1:', F1_Score)
		threshold_tmp = threshold_tmp + threshold_step_len

	return df


def compute_p_r_one(f_pred, threshold):
	return compute_p_r(f_pred, threshold, threshold, 1)


if __name__ == '__main__':
	# 参数
	version = 'v1_0'
	date = '20210408-20210414'
	dataset = 'all'

	# 文件路径
	PRED_BASE_PATH = '/Users/sunhaowen/Documents/data/3.ivr_log/final_data/{}/pred_result'.format(version)
	OUT_BASE_PATH = '/Users/sunhaowen/Documents/data/3.ivr_log/final_data/{}/eval_result'.format(version)
	PRED_FILE_NAME = 'pred_{}_{}_py.txt'.format(date, dataset)
	OUT_FILE_NAME = 'eval_result_{}_{}_py.xlsx'.format(date, dataset)

	my_f_pred = open(os.path.join(PRED_BASE_PATH, PRED_FILE_NAME), 'r')

	# 修改函数参数
	res_df = compute_p_r(my_f_pred, threshold_start=0.0, threshold_end=1.0, threshold_step_len=0.01)
	# res_df = compute_p_r_one(my_f_pred, threshold=0.77)

	# 输出文件
	res_df.to_excel(os.path.join(OUT_BASE_PATH, OUT_FILE_NAME), index=False)
