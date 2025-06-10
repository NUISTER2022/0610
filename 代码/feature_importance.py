#将特征的重要性进行排序

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree,datasets
from sklearn.tree import DecisionTreeRegressor
import math
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression

data=pd.read_csv(r"D:\pycharm2017\data\Data\combination_dummies.csv")

runs=20
num_folds=10
seed=0
num_tree=500
scoring='accuracy'
feature_names=data.columns
names=['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'school_MS',
       'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T', 'Mjob_health',
       'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_health',
       'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_home',
       'reason_other', 'reason_reputation', 'guardian_mother',
       'guardian_other', 'schoolsup_yes', 'famsup_yes', 'paid_yes',
       'activities_yes', 'nursery_yes', 'higher_yes', 'internet_yes',
       'romantic_yes', 'G1','G2']

X,y=data.iloc[:,:41].values.astype(float),data.iloc[:,41].values.astype(float)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=seed,stratify=y)
sc=StandardScaler()   #标准化
X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)

forest = RandomForestClassifier(n_estimators=50,random_state=1)
forest.fit(X_train_std, y_train)
print('train accuarcy:',forest.score(X_train_std,y_train))
print('test accuarcy:',forest.score(X_test_std,y_test))

importances = forest.feature_importances_     #输出一个数组，返回每个特征的重要性
# indices = np.argsort(importances) #这是升序
indices = np.argsort(importances)[::-1] #这是降序
# print(indices)  #结果[22 30 36 25 18 27 21 17 20 37 26 31 35 13 15 24 29 28 16 14 23 19 38 32 34 33  9  3  4  5  6  2  7  1 10 11  8  0 12 39 40]

results_x=[]
results_y=[]
for i in range(39):
    forest.fit(X_train_std[:,indices[:i+1]], y_train)
    print('选取特征重要性前',i+1)
    print('test accuarcy:',forest.score(X_test_std[:,indices[:i+1]],y_test))
    results_y.append(forest.score(X_test_std[:,indices[:i+1]],y_test))
    results_x.append(i+1)
plt.xlim(0,50)
plt.ylim(0,1)
plt.xlabel('number of the most important feature')
plt.ylabel('accuracy')
plt.plot(results_x,results_y,marker='*')
plt.show()