# encoding: utf-8
import codecs
import os
from xlwt import *
import xlrd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 参数
version = 'v1_0'
date = '20210408-20210414'
dataset = 'all'

# 数据路径：模型，测试集
feature_dir = "/Users/sunhaowen/Documents/data/3.ivr_log/final_data/{}".format(version)
# 加载模型
bst = pickle.load(open(feature_dir + "/model.dat", "rb"))


def read_txt(file_name):
    with codecs.open(file_name, "r") as fr:
        data = fr.read()
    data = [line for line in data.split("\n") if line]
    return data


def write_txt(data, file_name):
    fw = open(file_name, "w+")
    for line in data:
        if isinstance(line, list):
            fw.write("\t".join([str(elem) for elem in line]) + "\n")
        else:
            fw.write(str(line) + "\n")
    fw.close()

# 打开测试文件
data_dir = feature_dir
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
file_name = 'acfeature_{}_'.format(date)

xg_test = xgb.DMatrix(data_dir + "/" + file_name + "{}.txt".format(dataset))
data = read_txt(data_dir + "/" + file_name + "{}.txt".format(dataset))

# 获取Y
test_Y = []
for line in data:
    line_split = line.split(" ")
    test_Y.append(int(line_split[0]))
test_Y = np.array(test_Y)

param = {}
param['num_class'] = 13

# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
time_begin = time.time()
yprob = bst.predict(xg_test).reshape(test_Y.shape[0], param['num_class'])
ylabel = np.argmax(yprob, axis=1)
time_end = time.time()
print('解析query用时: {}'.format(time_end - time_begin))
print('predicting, classification error=%f' % (
        sum(int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))

result = []
for cur_label, line in zip(test_Y, yprob):
    one_result = []
    sorted_result = sorted(enumerate(line), key=lambda x: x[1], reverse=True)
    for index in range(len(sorted_result)):
        # 输出top几
        if index >= 1:
            break
        one_result.append(str(int(sorted_result[index][0])) + "_" + str(float(sorted_result[index][1])))
    result.append(str(int(cur_label)) + "\t" + "\t".join(one_result))

# 输出文件
write_txt(result, feature_dir + "/pred_result/pred_{}_{}_py.txt".format(date, dataset))