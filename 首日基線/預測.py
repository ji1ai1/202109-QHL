# python 3.9.2
# python套件 lightgbm 3.1.1
# python套件 numpy 1.20.1
# python套件 pandas 1.2.2
#
# 輸入：
#	final_dataset_train.tsv
#	final_dataset_testA.tsv
#
# 輸出：
# 	result.csv
#
# 0.7852
#
import lightgbm
import numpy
import pandas
import random

def 統計特征(某表, 鍵, 統計字典, 前綴=""):
	if not isinstance(鍵, list):
		鍵 = [鍵]
	某統計表 = 某表.groupby(鍵).aggregate(統計字典)
	某統計表.columns = ["%s%s之%s%s" % (前綴, "".join(鍵), 欄名, 丑 if isinstance(丑, str) else 丑.__name__) for 欄名, 函式 in 統計字典.items() for 丑 in (函式 if isinstance(函式, list) else [函式])]
	return 某統計表
pandas.DataFrame.統計特征 = 統計特征

訓練表 = pandas.read_csv("final_dataset_train.tsv", sep="\t")
訓練表["id"] = range(-1, -len(訓練表) -1, -1)
測試表 = pandas.read_csv("final_dataset_testA.tsv", sep="\t")
測試表["delta_g"] = -1

訓練基礎特征表 = 訓練表.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
for 甲 in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
	訓練基礎特征表["%s長度" % 甲] = 訓練基礎特征表[甲].str.len()
	for 乙 in [chr(65 + 子) for 子 in range(26)]:
		訓練基礎特征表["%s_%s" % (甲, 乙)] = 訓練基礎特征表[甲].str.count(乙)
		for 丙 in [chr(65 + 子) for 子 in range(26)]:
			訓練基礎特征表["%s_%s" % (甲, 乙 + 丙)] = 訓練基礎特征表[甲].str.count(乙 + 丙)
			for 丁 in [chr(65 + 子) for 子 in range(26)]:
				訓練基礎特征表["%s_%s" % (甲, 乙 + 丙 + 丁)] = 訓練基礎特征表[甲].str.count(乙 + 丙 + 丁)
訓練基礎特征表 = 訓練基礎特征表.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)

測試基礎特征表 = 測試表.loc[:, ["id", "antibody_seq_a", "antibody_seq_b", "antigen_seq"]]
for 甲 in ["antibody_seq_a", "antibody_seq_b", "antigen_seq"]:
	測試基礎特征表["%s長度" % 甲] = 測試基礎特征表[甲].str.len()
	for 乙 in [chr(65 + 子) for 子 in range(26)]:
		測試基礎特征表["%s_%s" % (甲, 乙)] = 測試基礎特征表[甲].str.count(乙)
		for 丙 in [chr(65 + 子) for 子 in range(26)]:
			測試基礎特征表["%s_%s" % (甲, 乙 + 丙)] = 測試基礎特征表[甲].str.count(乙 + 丙)
			for 丁 in [chr(65 + 子) for 子 in range(26)]:
				測試基礎特征表["%s_%s" % (甲, 乙 + 丙 + 丁)] = 測試基礎特征表[甲].str.count(乙 + 丙 + 丁)
測試基礎特征表 = 測試基礎特征表.drop(["antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)



def 取得資料表(某表, 某基礎特征表, 某特征表):
	某資料表 = 某表
	某資料表 = 某資料表.merge(某基礎特征表, on="id", how="left")
	某資料表 = 某資料表.merge(某特征表.統計特征("antibody_seq_a", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antibody_seq_a", how="left")
	某資料表 = 某資料表.merge(某特征表.統計特征("antibody_seq_b", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antibody_seq_b", how="left")
	某資料表 = 某資料表.merge(某特征表.統計特征("antigen_seq", {"delta_g": ["mean", "median", "min", "max"]}).reset_index(), on="antigen_seq", how="left")
	某資料表 = 某資料表.drop(["pdb", "antibody_seq_a", "antibody_seq_b", "antigen_seq"], axis=1)
	
	某資料表["標籤"] = 某資料表.delta_g.rank()
	某資料表 = 某資料表.loc[:, ["id", "delta_g", "標籤"] + [子 for 子 in 某資料表.columns if 子 not in ["id", "delta_g", "標籤"]]]
	
	return 某資料表

折數 = 6
索引 = random.sample(range(len(訓練表)), len(訓練表))
訓練資料表 = None
for 乙 in range(折數):
	乙標籤表 = 訓練表.iloc[[子 for 子 in range(len(索引)) if 子 % 折數 == 乙]].reset_index(drop=True)
	乙特征表 = 訓練表.iloc[[子 for 子 in range(len(索引)) if 子 % 折數 != 乙]].reset_index(drop=True)
	
	乙資料表 = 取得資料表(乙標籤表, 訓練基礎特征表, 乙特征表)
	訓練資料表 = pandas.concat([訓練資料表, 乙資料表], ignore_index=True)

輕模型 = lightgbm.train(train_set=lightgbm.Dataset(訓練資料表.iloc[:, 3:], label=訓練資料表.標籤)
	, num_boost_round=2048, params={"objective": "regression", "learning_rate": 0.05, "max_depth": 6, "num_leaves": 32, "bagging_fraction": 0.7, "feature_fraction": 0.7, "num_threads": 64, "verbose": -1}
)


測試資料表 = 取得資料表(測試表, 測試基礎特征表, 訓練表)

預測表 = 測試資料表.loc[:, ["id"]]
預測表["delta_g"] = 輕模型.predict(測試資料表.iloc[:, 3:])

預測表.to_csv("result.csv", index=False)
