## 训练
#### 详细过程

从ivr_log文件中获取每个session对应的最后一个不为空的ivroriginalorderinfos列值

#### 输入数据
- acfeature_{date}_{train/val} label+feature

#### 输出数据
- model.dat: 保存模型
- pred_{}_val: 验证集的预测结果

#### 运行
```
python train.py
```

## 测试


## 模型评估
计算precision，Recall和F1

## 计算填槽准确率

## 计算路由准确率