#设定随机种子,以保证实验复现
#1
from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
#2
import torch
seed = 360
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
#3
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale #标准化
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(DEVICE)

# 读取数学数据
def get_mat_data():
    data = pd.read_csv('mat_dummies_.csv')
    y = data['G3']
    x = data.drop(['G3'], axis=1)
    #使用以上的特征需要将self.fc1 = nn.Linear(41, 200)中的41修改为34，结果精度更高

    x = scale(x) #标准化
    x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2)

    return x_train, x_test, y_train, y_test

#搭建模型
class MatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(41, 200)
        self.fc5 = nn.Linear(200, 50)
        #self.fc2 = nn.Linear(200,100)
        #self.fc3 = nn.Linear(100, 60)
        #self.fc4 = nn.Linear(60, 30)
        self.fc6 = nn.Linear(50, 20)
        self.fc7 = nn.Linear(20, 5)


        self.dropout1 = nn.Dropout(0.005) #防止或减轻过拟合而使用的函数，在不同的训练过程中随机扔掉一部分神经元。0.005是每个元素被保留下来的概率

    def forward(self, x):
        x = self.fc1(x) #经过第一层的传播 h1=wx+b
        x = F.relu(x) #激活函数，非线性的

        x = self.fc5(x)
        x = F.relu(x)
        '''
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        #x = self.dropout1(x)
        x = self.fc4(x)
        x = F.relu(x)
        '''
        x = self.fc6(x)
        x = F.relu(x)
        
        
        x = self.fc7(x)

        out = nn.Softmax(dim=1)(x)  #激活函数softmax，处理多分类

        return out


# 数学
x_train, x_test, y_train, y_test = get_mat_data()
model = MatNet().to(DEVICE)

#print(x_train.shape, y_train.shape)#(316, 41) (316,)

train_len, val_len = x_train.shape[0], x_test.shape[0]

optimizer = torch.optim.Adam(model.parameters(),lr=0.005) #优化器
loss = torch.nn.CrossEntropyLoss()   #交叉熵函数
#loss = torch.nn.NLLLoss()  

x_train, x_test, y_train, y_test = torch.tensor(x_train).to(DEVICE).float(), torch.tensor(x_test).to(
    DEVICE).float(), torch.tensor(
    y_train).to(DEVICE).long(), torch.tensor(y_test).to(DEVICE).long()

#训练
loss_h=[]
best_acc = 0
for i in range(300):#1000
    model.train()
    output = model(x_train)
    #print(output.size())#(316,5)
    #print(output.data)
    y_pred = torch.max(output.data, 1)#包含两个tensor,分别包含最大值、对应索引
    #print(output.data)
    #print(output.data)#[316, 5]
    #print(y_pred[0])
    #print(y_pred[0].indices)

    train_correct = (y_pred[1] == y_train).sum().to('cpu').item()   #Tensor中有一个独特的indices操作，按照所给的索引进行取数
    train_loss = loss(output, y_train)
    #print(train_loss)

    optimizer.zero_grad()  #清零梯度
    train_loss.backward() #梯度计算
    optimizer.step()   #更新梯度

    #验证模式
    model.eval()
    pred = model(x_test)
    #计算正确数
    y_pred = torch.max(pred.data, 1)
    val_loss = loss(pred, y_test)
    val_correct = (y_pred[1] == y_test).sum().to('cpu').item()

    # 修改这里，将损失值转换为 NumPy 数组
    loss_h.append(train_loss.detach().cpu().numpy())

    if best_acc < (val_correct / val_len):
        best_acc = val_correct / val_len

    print('epoch:{}, loss:{:.3f}, acc:{:.3f}, val_loss:{:.3f}, val_acc:{:.3f}'.format(i + 1, train_loss.item(),
                                                                                      train_correct / train_len,
                                                                                      val_loss.item(),
                                                                                      val_correct / val_len))
print('best val_acc:{:.3f}'.format(best_acc))
plt.figure(figsize=(6,6))
plt.plot(loss_h)
plt.show()
