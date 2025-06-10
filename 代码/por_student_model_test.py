#将处理好的数据，建模测试

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
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree,datasets
from sklearn.tree import DecisionTreeRegressor
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


data=pd.read_csv(r"D:\pycharm2017\data\Data\por_dummies.csv")
pd.set_option('display.max_row',None)   #让控制台显示的表格行的内容是完整的
pd.set_option('display.max_column',None)
# print(data.head())

X=data.iloc[:,:41].values.astype(float)
y=data.iloc[:,41].values.astype(float) #astype实现变量类型转换
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

runs=20
seed=38
num_folds=10
scoring='accuracy'

#随机森林
# result_RF=[]
# pipe=Pipeline([('clf',RandomForestClassifier(random_state=420))])
# params= {                                             #定义搜索的参数范围
#     "clf__n_estimators": [100, 200, 500],           #数的棵数
#     "clf__criterion": ["gini", "entropy"],
# }
#
# grid=GridSearchCV(pipe,params,cv=num_folds)    #num_folds=10
# grid.fit(X_train,y_train)
# print('最佳模型:{}'.format(grid.best_params_))   #format用于填充
# print('RandomForest_Classifier模型的最佳得分:{:.3f}'.format(grid.best_score_))
# rf=grid.best_estimator_    #给出最高分数（或指定的最小损失）的估计器
# rf=rf.fit(X_train,y_train)
# print('RandomForest_Classifier最佳模型在测试集上的得分:{:.3f}'.format(rf.score(X_test,y_test)))

# rfc=RandomForestClassifier(n_estimators=90,criterion="gini",max_depth=6,min_samples_leaf=1,random_state=100)
# rfc.fit(X_train,y_train)
# print(cross_val_score(rfc,X,y,cv=10).mean())
# print(rfc.feature_importances_)

# tr=[]
# for i in range(90,150,5):
#     rfc=RandomForestClassifier(n_estimators=i,random_state=10)
#     rfc=rfc.fit(X_train,y_train)
#     score=cross_val_score(rfc,X,y,cv=10).mean()
#     tr.append(score)
# plt.plot(range(90,150,5),tr,color="black")
# plt.xticks(range(90,150,5))
# plt.show()

#重新编写的网格搜索
parameters={
    "n_estimators":[*range(90,150,5)]
    ,"criterion":("gini","entropy")
    ,"max_depth":[*range(1,7)]
    ,"min_samples_leaf":[*range(1,10,2)]
}
clf=RandomForestClassifier(random_state=10)
GS=GridSearchCV(clf,parameters,cv=10)
GS.fit(X_train,y_train)
print(GS.best_score_)
print(GS.best_params_)
# """
# 结果：
# 0.7343750000000001
# {'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 1, 'n_estimators': 145}
# """



#决策树
# clf=DecisionTreeClassifier(criterion="gini",max_depth=3,min_samples_leaf=1,splitter="best",random_state=1)
# clf=clf.fit(X_train,y_train)
# score=clf.score(X_test,y_test)
# print(cross_val_score(clf,X,y,cv=10).mean())

# tr=[]
# te=[]
# for i in range(10):
#     clf=DecisionTreeClassifier(criterion="entropy",random_state=1,max_depth=i+1)
#     clf=clf.fit(X_train,y_train)
#     score_tr=clf.score(X_train,y_train)
#     score_te=cross_val_score(clf,X,y,cv=10).mean()
#     tr.append(score_tr)
#     te.append(score_te)
# print(max(te))
# plt.plot(range(1,11),tr,color="red",label="train")
# plt.plot(range(1,11),te,color="blue",label="test")
# plt.xticks(range(1,11))
# plt.legend()
# plt.show()
# #结果：criterion="entropy",random_state=1,max_depth=4，测试集分数：0.76698
#
# gini_threholds=np.linspace(0,0.5,50)  #取0~0.5，50个随机数
# parameters={
#     "criterion":("gini","entropy")
#     ,"splitter":("best","random")
#     ,"max_depth":[*range(1,7)]
#     ,"min_samples_leaf":[*range(1,10,2)]
#     ,"min_impurity_decrease":[*np.linspace(0,0.5,50)]
# }
# clf=DecisionTreeClassifier(random_state=1)
# GS=GridSearchCV(clf,parameters,cv=10)
# GS.fit(X_train,y_train)
# print(GS.best_score_)
# print(GS.best_params_)
# names=['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
#        'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'school_MS',
#        'sex_M', 'address_U', 'famsize_LE3', 'Pstatus_T', 'Mjob_health',
#        'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_health',
#        'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_home',
#        'reason_other', 'reason_reputation', 'guardian_mother',
#        'guardian_other', 'schoolsup_yes', 'famsup_yes', 'paid_yes',
#        'activities_yes', 'nursery_yes', 'higher_yes', 'internet_yes',
#        'romantic_yes', 'G1','G2']
# dot_data = tree.export_graphviz(clf,feature_names=names, class_names=["A", "B", "C","D","E"], filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# graph.view()

# """
# 结果：
# 0.7682459677419355
# {'criterion': 'gini', 'max_depth': 4, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 9, 'splitter': 'best'}
# """



#SVM支持向量机
# runs=20
# scoring='accuracy'
# pipe_svc=Pipeline([('scl',StandardScaler()),('clf',SVC())])
# param_range=[0.0001,0.001,0.01,0.1,10.0,100.0]
# param_grid=[{'clf__C': param_range,
#             'clf__kernel': ['linear']},
#            {'clf__C': param_range,
#             'clf__gamma': param_range,
#             'clf__kernel': ['rbf']}]
# result_SVM=[]
# for i in range(runs):
#    grid=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,cv=num_folds,scoring=scoring)
#    grid=grid.fit(X_train,y_train)
#    print('最佳模型:{}'.format(grid.best_params_))
#    print('SVM_Classifier模型的最佳得分:{:.3f}'.format(grid.best_score_))
#    svm2=grid.best_estimator_
#    svm2.fit(X_train,y_train)
#    print('SVM最佳模型在测试集上的得分是:{:.3f}'.format(svm2.score(X_test,y_test)))
#    result_SVM.append(svm2.score(X_test,y_test))
# print('SVM_Classifier最佳模型在测试集上20epoch得分:{:.3f}'.format(np.mean(result_SVM)))



#多层感知机神经网络
# result_MLP=[]
# kfold=KFold(n_splits=num_folds)
# params=[{'clf':[MLPClassifier(solver='lbfgs',random_state=seed)],
#         'scaler':[StandardScaler(),None],
#         'clf__hidden_layer_sizes':[(50,),(100,),(100,100)]}]
# pipe=Pipeline([('scaler',StandardScaler()),('clf',MLPClassifier())])
# for i in range(runs):
#    grid=GridSearchCV(pipe,params,cv=num_folds)
#    grid.fit(X_train,y_train)
#    print('最佳模型:{}'.format(grid.best_params_))
#    print('MLP_Classifier模型的最佳得分:{:.3f}'.format(grid.best_score_))
#    nn=grid.best_estimator_
#    nn=nn.fit(X_train,y_train)
#    print('MLP_Classifier最佳模型在测试集上的得分:{:.3f}'.format(nn.score(X_test,y_test)))
#    result_MLP.append(nn.score(X_test,y_test))
# print('MLP_Classifier最佳模型在测试集上20epoch得分:{:.3f}'.format(np.mean(result_MLP)))



#逻辑回归
# lr=LR(penalty="l2",multi_class='multinomial',solver="newton-cg",C=0.77,max_iter=1000)
# lr=lr.fit(X,y)
# print(lr.coef_)   #coef_查看每个特征对应的参数
# print((lr.coef_!=0).sum(axis=1))    #l1正则化下的结果：[13 17 23 22 22]
# print(accuracy_score(lr.predict(X_train),y_train))
# print(accuracy_score(lr.predict(X_test),y_test))
# print(lr.score(X_train,y_train))


# l=[]
# ltest=[]
# for i in np.linspace(0.05,1,19):
#     lr = LR(penalty="l2", multi_class='multinomial',solver="newton-cg", C=i, max_iter=1000)
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


# data_preb=pd.read_excel(r"D:\pycharm2017\data\Data\正确率.xlsx")
# # print(data_preb.head())
# c=CM(data_preb.loc[:,"G3"],data_preb.loc[:,"pred"],labels=[0,1,2,3,4])
# print(c)