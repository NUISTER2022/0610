#特征的卡方值
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\pycharm2017\data\Data\por_dummies.csv")
X,y=data.iloc[:,:41].values.astype(float),data.iloc[:,41].values.astype(float)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10,stratify=y)

score=[]
for i in range(41,10,-4):
    X_fschi=SelectKBest(chi2,k=i).fit_transform(X,y)
    once=cross_val_score(RFC(n_estimators=10,random_state=10),X_fschi,y,cv=5).mean()
    score.append(once)
plt.plot(range(41,10,-4),score)
plt.show()
# #数学成绩：大概25个特征的时候表现最好，葡萄牙语成绩：17个

#求p值和卡方值
chivalue,pvalues_chi=chi2(X,y)
print(chivalue.tolist())#卡方
print(pvalues_chi.tolist())#p
# k=chivalue.shape[0] - (pvalues_chi>0.05).sum()
# print(k)
