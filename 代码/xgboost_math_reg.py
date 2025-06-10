#xgboost
#1.提升即成算法：重要参数n_estimators
# xgboost的基础是梯度提升算法，xgboost局势由梯度提升树发展而来的，xgboost中所有的树都是二叉的。
# 分类树：预测=叶子上少数服从多数；
# 回归树：预测=叶子上的平均值；
#2.有放回随机抽样：重要参数subsample
# 初次有放回抽样（随机）  第二次有放回抽样（加大前一棵树被预测错误的样本的权重）
#3.迭代决策树：重要参数eta

from xgboost import XGBRegressor as XGBR
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

data=pd.read_csv(r"D:\pycharm2017\data\Data\mat_dummies.csv")
pd.set_option('display.max_row',None)   #让控制台显示的表格行的内容是完整的
pd.set_option('display.max_column',None)

#除去了G3
X=data.iloc[:,:40].values.astype(float)
y=data.iloc[:,40].values.astype(float) #atype实现变量类型转换
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=20)


reg=XGBR(n_estimators=100,learning_rate=0.05,booster='dart',reg_lambda=1,gamma=6.5,colsample_bylevel=0.5,min_child_weight=0.6).fit(X_train,y_train)
# print(reg.predict(X_test))
print(reg.feature_importances_)
print(reg.score(X_test,y_test))
print(y.mean())
print(MSE(y_test,reg.predict(X_test)))




# axisx=np.linspace(0,1,20)
# rs=[]
# for i in axisx:
#     reg=XGBR(n_estimators=120,subsample=i,random_state=20)
#     rs.append(cross_val_score(reg,X_train,y_train,cv=10).mean())
# print(axisx[rs.index(max(rs))],max(rs))
# plt.figure(figsize=(20,5))
# plt.plot(axisx,rs,c='green',label="XGB")
# plt.legend()
# plt.show()





