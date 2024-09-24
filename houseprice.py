import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入波士顿房价数据集
data = pd.read_csv(r"D:\boston_house_prices.csv")

# 使用“RM”作为特征，使用“MEDV”作为目标
X = data['RM'].values  # 特征变量：平均房间数
y = data['MEDV'].values  # 目标变量：中位房价


# 线性回归的梯度下降实现
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m = len(y)  # 样本数量
    X_b = np.c_[np.ones((m, 1)), X]  # 添加偏置项（常数1）
    theta = np.random.randn(2)  # 随机初始化参数（权重）

    for epoch in range(epochs):
        # 计算预测值
        predictions = X_b.dot(theta)
        # 计算损失（均方误差）
        loss = predictions - y
        # 计算梯度
        gradients = 2 / m * X_b.T.dot(loss)
        # 更新参数
        theta -= lr * gradients

    return theta


# 执行梯度下降
theta = gradient_descent(X, y)

# 计算每个点的预测值
predictions = np.c_[np.ones((len(X), 1)), X].dot(theta)  # 计算预测值
losses = (predictions - y) ** 2  # 每个点的损失值（平方误差）

# 输出每个点的预测值和真实值
for i in range(len(y)):
    print(f"Point {i}: Real Value = {y[i]}, Predicted Value = {predictions[i]}, Loss = {losses[i]}")

# 画图
plt.scatter(X, y, color='blue', label='Data points')  # 绘制数据点
plt.plot(X, predictions, color='red', label='Regression line')  # 绘制回归线
plt.xlabel('Average Number of Rooms (RM)')  # X轴标签
plt.ylabel('Median House Price (MEDV)')  # Y轴标签
plt.title('Boston Housing Prices vs Average Rooms')  # 图表标题
plt.legend()  # 显示图例
plt.show()  # 显示图形
