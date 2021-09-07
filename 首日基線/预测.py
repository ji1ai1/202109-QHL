# python 3.9.2
# python套件 lightgbm 3.1.1
# python套件 numpy 1.20.1
# python套件 pandas 1.2.2
#
# 输入：
#  final_dataset_train.tsv
#  final_dataset_testA.tsv
#
# 输出：
# 	result.csv
#
# 0.76
#
import lightgbm
import numpy
import pandas
import random

def 统计特征(某表, 键, 统计字典, 前缀=""):
   if not isinstance(键, list):
      键 = [键]
   某统计表 = 某表.groupby(键).aggregate(统计字典)
   某统计表.columns = ["%s%s之%s%s" % (前缀, "".join(键), 栏名, 丑 if isinstance(丑, str) else 丑.__name__) for 栏名, 函式 in 统计字典.items() for 丑 in (函式 if isinstance(函式, list) else [函式])]
   return 某统计表
pandas.DataFrame.统计特征 = 统计特征

训练表 = pandas.read_csv("final_dataset_train.tsv", sep="\t")
训练表["id"] = range(-1, -len(训练表) -1, -1)
测试表 = pandas.read_csv("final_dataset_testA.tsv", sep="\t")
测试表["delta_g"] = -1

训练基础特征表 = 训练表.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
for 甲 in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
   训练基础特征表["%s长度" % 甲] = 训练基础特征表[甲].str.len()
   for 乙 in [chr(65 + 子) for 子 in range(26)]:
      训练基础特征表["%s_%s" % (甲, 乙)] = 训练基础特征表[甲].str.count(乙)
      for 丙 in [chr(65 + 子) for 子 in range(26)]:
         训练基础特征表["%s_%s" % (甲, 乙 + 丙)] = 训练基础特征表[甲].str.count(乙 + 丙)
         for 丁 in [chr(65 + 子) for 子 in range(26)]:
            训练基础特征表["%s_%s" % (甲, 乙 + 丙 + 丁)] = 训练基础特征表[甲].str.count(乙 + 丙 + 丁)
训练基础特征表 = 训练基础特征表.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)

测试基础特征表 = 测试表.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
for 甲 in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
   测试基础特征表["%s长度" % 甲] = 测试基础特征表[甲].str.len()
   for 乙 in [chr(65 + 子) for 子 in range(26)]:
      测试基础特征表["%s_%s" % (甲, 乙)] = 测试基础特征表[甲].str.count(乙)
      for 丙 in [chr(65 + 子) for 子 in range(26)]:
         测试基础特征表["%s_%s" % (甲, 乙 + 丙)] = 测试基础特征表[甲].str.count(乙 + 丙)
         for 丁 in [chr(65 + 子) for 子 in range(26)]:
            测试基础特征表["%s_%s" % (甲, 乙 + 丙 + 丁)] = 测试基础特征表[甲].str.count(乙 + 丙 + 丁)
测试基础特征表 = 测试基础特征表.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)



def 取得数据表(某表, 某基础特征表, 某特征表):
   某数据表 = 某表
   某数据表 = 某数据表.merge(某基础特征表, on="id", how="left")
   某数据表 = 某数据表.merge(某特征表.统计特征("antibody_seq_a", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antibody_seq_a", how="left")
   某数据表 = 某数据表.merge(某特征表.统计特征("antibody_seq_b", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antibody_seq_b", how="left")
   某数据表 = 某数据表.merge(某特征表.统计特征("antigen_seq", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antigen_seq", how="left")
   某数据表 = 某数据表.drop(["pdb", "antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)
   
   某数据表["标签"] = 某数据表.delta_g.rank()
   某数据表 = 某数据表.loc[:, ["id", "delta_g", "标签"] + [子 for 子 in 某数据表.columns if 子 not in ["id", "delta_g", "标签"]]]
   
   return 某数据表

折数 = 6
索引 = random.sample(range(len(训练表)), len(训练表))
训练数据表 = None
for 乙 in range(折数):
   乙标签表 = 训练表.iloc[[子 for 子 in range(len(索引)) if 子 % 折数 == 乙]].reset_index(drop=True)
   乙特征表 = 训练表.iloc[[子 for 子 in range(len(索引)) if 子 % 折数 != 乙]].reset_index(drop=True)
   
   乙数据表 = 取得数据表(乙标签表, 训练基础特征表, 乙特征表)
   训练数据表 = pandas.concat([训练数据表, 乙数据表], ignore_index=True)

轻模型 = lightgbm.train(train_set=lightgbm.Dataset(训练数据表.iloc[:, 3:], label=训练数据表.标签)
   , num_boost_round=2048, params={"objective": "regression", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)


测试数据表 = 取得数据表(测试表, 测试基础特征表, 训练表)

预测表 = 测试数据表.loc[:, ["id"]]
预测表["delta_g"] = 轻模型.predict(测试数据表.iloc[:, 3:])

预测表.to_csv("result.csv", index=False)
