# 移除未使用的导入
# from xgboost import XGBRegressor as XGBR
from xgboost import XGBClassifier as XGBC
# 移除未使用的导入
# from sklearn.ensemble import RandomForestRegressor as RFR
# from sklearn.linear_model import LinearRegression as LinearR
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_squared_error as MSE
import pandas as pd
# 移除未使用的导入
# import numpy as np
# import matplotlib.pyplot as plt
# import torch

data=pd.read_csv(r"D:\pycharm2017\data\Data\por_dummies_.csv")
pd.set_option('display.max_row',None)   #让控制台显示的表格行的内容是完整的
pd.set_option('display.max_column',None)

#除去了G3
X=data.iloc[:,:41].values.astype(float)
y=data.iloc[:,41].values.astype(float) #atype实现变量类型转换
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=20)

# 假设是分类问题，修正 objective 参数
num_classes = len(set(y))  # 计算类别数
clf=XGBC(n_estimators=900
         ,learning_rate=0.1
         ,booster='gbtree'
         ,objective='multi:softmax'  # 修改为多分类 softmax
         ,num_class=num_classes  # 指定类别数
         ,gamma=6
         ,random_state=90).fit(X_train,y_train)
print(clf.predict(X))
print(clf.feature_importances_)
print(clf.score(X_test,y_test))
print(y.mean())
print(MSE(y_test,clf.predict(X_test)))

