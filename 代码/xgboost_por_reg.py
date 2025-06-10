from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_squared_error as MSE
import pandas as pd

try:
    # 读取 CSV 文件
    data = pd.read_csv(r"D:\pycharm2017\data\Data\por_dummies.csv")
    # 修正参数名拼写错误
    pd.set_option('display.max_rows', None)   # 让控制台显示的表格行的内容是完整的
    pd.set_option('display.max_columns', None)

    # 检查数据列数是否足够
    if data.shape[1] < 41:
        raise IndexError("数据列数少于 41 列，无法按指定索引分割特征和目标变量")

    # 除去了 G3
    X = data.iloc[:, :40].apply(pd.to_numeric, errors='coerce').values
    y = data.iloc[:, 40].apply(pd.to_numeric, errors='coerce').values

    # 处理 NaN 值
    import numpy as np
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=20)

    reg = XGBR(n_estimators=120, learning_rate=0.1, booster='dart', gamma=4.4, max_depth=5, colsample_bylevel=0.8).fit(X_train, y_train)
    # print(reg.predict(X_test))
    print(reg.feature_importances_)
    print(reg.score(X_test, y_test))
    print(y.mean())
    print(MSE(y_test, reg.predict(X_test)))
except FileNotFoundError:
    print("指定的 CSV 文件未找到，请检查文件路径。")
except IndexError as e:
    print(e)
except Exception as e:
    print(f"发生未知错误: {e}")# print(reg.predict(X_test))
print(reg.feature_importances_)
print(reg.score(X_test,y_test))
print(y.mean())
print(MSE(y_test,reg.predict(X_test)))