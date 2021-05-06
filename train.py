#!/usr/bin/python

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
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

version = 'v1'
date = '20210304-20210406'


def read_xls(file_name):
    team_query_data = xlrd.open_workbook(file_name)
    table_data = team_query_data.sheet_by_index(0)
    return table_data


def write_xls(data, result_file):
    file = Workbook(encoding='utf-8')
    table = file.add_sheet("result")
    for index, each_data in enumerate(data):
        for i in range(len(each_data)):
            table.write(index, i, str(each_data[i]))
    file.save(result_file)


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


def train(feature_dir):
    data_dir = feature_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_name = 'acfeature_{}_'.format(date)
    xg_train = xgb.DMatrix(data_dir + "/" + file_name + "train.txt")
    xg_test = xgb.DMatrix(data_dir + "/" + file_name + "val.txt")
    data = read_txt(data_dir + "/" + file_name + "val.txt")

    test_Y = []
    for line in data:
        line_split = line.split(" ")
        test_Y.append(int(line_split[0]))
    test_Y = np.array(test_Y)

    param = {}
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.1  
    param['max_depth'] = 6  
    param['silent'] = 1  
    param['nthread'] = 4  
    param['num_class'] = 13
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 50

    # 训练：方法1
    # bst = xgb.train(param, xg_train, num_round, watchlist)
    # pred = bst.predict(xg_test)
    # print('predicting, classification error=%f' % (sum(int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))

    # 训练：方法2
    param['objective'] = 'multi:softprob'
    bst = xgb.train(param, xg_train, num_round, watchlist)

    # 模型保存与加载
    # 方式1
    pickle.dump(bst, open(feature_dir + "/model_1.dat", "wb"))
    # bst = pickle.load(open(feature_dir + "/model.dat", "rb"))
    # 方式2
    # bst.dump_model(feature_dir + "/model.txt")
    # bst.save_model(feature_dir + "/xgboost_python.model")
    # bst = xgb.Booster({'nthread': 4})
    # bst.load_model(feature_dir + "/xgboost_python.model")

    pdf = PdfPages(feature_dir + "/feature_importance.pdf")
    xgb.plot_importance(bst, max_num_features=10, importance_type="gain")
    pdf.savefig()
    plt.close()
    pdf.close()

    # Note: this convention has been changed since xgboost-unity
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    time_begin = time.time()
    yprob = bst.predict(xg_test).reshape(test_Y.shape[0], param['num_class'])
    ylabel = np.argmax(yprob, axis=1)
    time_end = time.time()
    print('解析query用时: {}'.format(time_end - time_begin))
    print('predicting, classification error=%f' % (sum(int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))

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
    write_txt(result, feature_dir + "/pred_{}_val_py.txt".format(date))


def main():
    feature_dir = "/Users/sunhaowen/Documents/data/3.ivr_log/final_data/{}".format(version)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    train(feature_dir)


if __name__ == "__main__":
    main()
