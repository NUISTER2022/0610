#逻辑回归
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error,explained_variance_score
from pandas import read_csv
from pandas.plotting import scatter_matrix
from pandas import set_option
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.naive_bayes
from sklearn.neural_network import MLPClassifier

from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree,datasets
from sklearn.tree import DecisionTreeRegressor
import math
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix as CM,precision_score as P,recall_score as R

# 修改文件路径为实际路径
data = pd.read_csv(r"D:\pycharm2017\data\Data\mat_dummies.csv")
pd.set_option('display.max_row', None)   # 让控制台显示的表格行的内容是完整的
pd.set_option('display.max_column', None)

# print(data.shape[1])


X=data.iloc[:,:41].values.astype(float)
y=data.iloc[:,41].values.astype(float) #astype实现变量类型转换
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=20)

# Remove the 'multi_class' parameter
lr=LR(penalty="l2", solver="newton-cg", C=0.8, max_iter=1000)
lr=lr.fit(X,y)
# print(lr.coef_)   #coef_查看每个特征对应的参数
# print((lr.coef_!=0).sum(axis=1))    #l1正则化下的结果：[13 17 23 22 22]
# print(accuracy_score(lr.predict(X_train),y_train))
# print(accuracy_score(lr.predict(X_test),y_test))
# print(lr.score(X_train,y_train))


# l=[]
# ltest=[]
# Remove the 'multi_class' parameter
# for i in np.linspace(0.05,1,19):
#     lr = LR(penalty="l2", solver="newton-cg", C=i, max_iter=1000)
#     lr=lr.fit(X_train,y_train)
#     l.append(accuracy_score(lr.predict(X_train),y_train))
#     ltest.append(accuracy_score(lr.predict(X_test),y_test))
# graph=[l,ltest]
# color=["green","blue"]
# label=["lr","lr_test"]
# plt.figure(figsize=(6,6))
# for i in range(len(graph)):
#     plt.plot(np.linspace(0.05,1,19),graph[i],color[i],label=label[i])
# plt.legend(loc=4)
# plt.show()


# prob=lr.predict_proba(X)
# prob=pd.DataFrame(prob)
# print(prob)
#
# for i in range(0,len(prob)):
#     List = []
#     # List.append(prob.index[i])
#     List.append((prob.iloc[i].sort_values(ascending=False)).index[0])
#     # List.append((prob.iloc[i].sort_values(ascending=False))[0])
#     print(List)


data_preb=pd.read_excel(r"D:\pycharm2017\data\Data\预测值与真实值.xlsx")
# print(data_preb.head())
c=CM(data_preb.loc[:,"G3"],data_preb.loc[:,"pred"],labels=[0,1,2,3,4])
print(c)


