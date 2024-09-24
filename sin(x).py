import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据
x_train = np.linspace(0, 2 * np.pi, 500)  # 在0到2π之间生成500个样本点
y_train = np.sin(x_train) + np.random.normal(0, 0.05, x_train.shape)  # 计算真实值并添加噪声

# 定义五次多项式模型函数
def model(x, params):
    # 返回五次多项式的计算结果
    return (params[0] * x**5 +
            params[1] * x**4 +
            params[2] * x**3 +
            params[3] * x**2 +
            params[4] * x +
            params[5])  # 包含常数项

# 定义损失函数
def loss(y_true, y_pred, params, lambda_reg=0.01):
    # 计算均方误差损失，并添加L2正则化项
    return np.mean((y_true - y_pred) ** 2) + lambda_reg * np.sum(params**2)

# 计算梯度函数
def gradients(x, y_true, params):
    y_pred = model(x, params)  # 计算当前模型的预测值
    dL_dparams = np.zeros_like(params)  # 初始化梯度数组
    for i in range(len(params)):
        params[i] += 1e-5  # 增加参数，计算数值梯度
        dL_dparams[i] = (loss(y_true, model(x, params), params) -
                         loss(y_true, model(x, params - 1e-5), params - 1e-5)) / 1e-5
        params[i] -= 1e-5  # 还原参数，确保参数不变
    return dL_dparams  # 返回计算出的梯度

# 初始化参数
params = np.random.rand(6) * 0.1  # 随机初始化6个模型参数，缩小范围
learning_rate = 0.001  # 设置学习率
iterations = 10000  # 设置迭代次数

# 训练模型
for _ in range(iterations):
    grads = gradients(x_train, y_train, params)  # 计算当前梯度
    params -= learning_rate * grads  # 更新参数

# 进行预测
y_pred = model(x_train, params)  # 计算最终预测值

# 计算最终损失
final_loss = loss(y_train, y_pred, params)  # 计算预测后的损失值

# 打印真实值、预测值和损失值
print("真实值:", y_train[:10])  # 打印前10个真实值
print("预测值:", y_pred[:10])    # 打印前10个预测值
print("最终损失:", final_loss)    # 打印最终损失值

# 可视化结果
plt.scatter(x_train, y_train, label='Training Data', color='red', alpha=0.5)  # 绘制训练数据点
plt.plot(x_train, y_pred, label='Fitted Model', color='blue')  # 绘制拟合模型曲线
plt.plot(x_train, np.sin(x_train), label='True sin(x)', color='green', linestyle='dashed')  # 绘制真实的sin(x)曲线
plt.legend()  # 显示图例
plt.title('Fitting sin(x) using 5th Degree Polynomial with Gradient Descent')  # 图表标题
plt.xlabel('x')  # x轴标签
plt.ylabel('y')  # y轴标签
plt.grid()  # 添加网格
plt.show()  # 显示图形
