#描述性统计
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"D:\pycharm2017\data\Data\mat_dummies.csv")
pd.set_option('display.max_row',None)
pd.set_option('display.max_column',None)

X=data.iloc[:,:41]
y=data.iloc[:,41]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=20)

print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
#print(y.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)

