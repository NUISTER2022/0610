from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
#2
import torch
seed = 380
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
#3
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# 读取葡萄牙数据
def get_por_data():
    data = pd.read_csv('por_dummies_.csv')
    y = data['G3']
    x = data.drop(['G3'], axis=1)

    x = scale(x)
    x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2, random_state=42,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


class PorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(41, 250)
        self.fc2 = nn.Linear(250, 90)
        self.fc3 = nn.Linear(90, 25)
        self.fc4 = nn.Linear(25, 5)

        #self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)
        #x = self.dropout1(x)

        x = self.fc4(x)

        out = nn.Softmax(dim=1)(x)

        return out


# 葡萄牙
x_train, x_test, y_train, y_test = get_por_data()
model = PorNet().to(DEVICE)

print(x_train.shape, y_train.shape)
train_len, val_len = x_train.shape[0], x_test.shape[0]



optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
loss = torch.nn.CrossEntropyLoss()

x_train, x_test, y_train, y_test = torch.tensor(x_train).to(DEVICE).float(), torch.tensor(x_test).to(
    DEVICE).float(), torch.tensor(
    y_train).to(DEVICE).long(), torch.tensor(y_test).to(DEVICE).long()

best_acc = 0
loss_h=[]
for i in range(200):
    model.train()
    output = model(x_train)

    y_pred = torch.max(output.data, 1)
    train_correct = (y_pred[1] == y_train).sum().to('cpu').item()
    train_loss = loss(output, y_train)

    optimizer.zero_grad()  #优化器清零
    train_loss.backward()  #反馈
    optimizer.step()       #优化

    model.eval()
    pred = model(x_test)
    y_pred = torch.max(pred.data, 1)
    val_loss = loss(pred, y_test)
    val_correct = (y_pred[1] == y_test).sum().to('cpu').item()  #预测对的个数

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