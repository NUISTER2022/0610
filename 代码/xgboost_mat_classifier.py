from xgboost import XGBRegressor as XGBR
from xgboost import XGBClassifier as XGBC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.model_selection import KFold, cross_val_score ,train_test_split
from  sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\pycharm2017\data\Data\mat_dummies_.csv")
pd.set_option('display.max_row',None)   #让控制台显示的表格行的内容是完整的
pd.set_option('display.max_column',None)

#除去了G3
X=data.iloc[:,:41].values.astype(float)
y=data.iloc[:,41].values.astype(float) #atype实现变量类型转换
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=20)


clf=XGBC(n_estimators=900
         ,learning_rate=0.1
         ,booster='dart'
         ,objective='softmax'
         ,gamma=7.5
         ,random_state=90).fit(X_train,y_train)
print(clf.predict(X))
print(clf.feature_importances_)
print(clf.score(X_test,y_test)) #0.810126582278481
# print(log_loss(y_test,clf.predict(X_test)))