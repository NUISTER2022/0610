#线性回归,岭回归,Lasso
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as MSE
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\pycharm2017\data\Data\por_dummies.csv")
pd.set_option('display.max_row',None)   #让控制台显示的表格行的内容是完整的
pd.set_option('display.max_column',None)

#除去了G3
X=data.iloc[:,:40].values.astype(float)
y=data.iloc[:,40].values.astype(float) #astype实现变量类型转换
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=20)

# 使用 StandardScaler 对特征数据进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建线性回归模型并训练，移除 normalize 参数
reg = LR().fit(X_train_scaled, y_train)
yhat = reg.predict(X_test_scaled)

print(cross_val_score(reg, scaler.transform(X), y, cv=5, scoring="neg_mean_squared_error").mean())  # 负均方误差
print(r2_score(y_test, yhat))   # 结果是0.7627930875675536，大概24%的信息没有捕捉到
print(reg.score(X_test_scaled, y_test))   # 结果是0.7627930875675536# plt.legend()
# plt.show()
