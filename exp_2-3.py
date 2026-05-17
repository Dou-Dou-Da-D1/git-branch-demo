#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：任务3 学习率曲线（基于手写GD）
- 目的：让学生实现线性回归的梯度下降训练函数，并比较不同学习率下的收敛曲线
- 本模板仅保留数据读取、划分与可视化骨架；算法实现留白
- 允许使用 numpy / pandas / matplotlib，禁止使用 sklearn
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CSV = os.path.join(DATA_DIR, 'winequality-white.csv')
FIG_PATH = os.path.join(OUT_DIR, 'task3_lr_curves_template.png')
np.random.seed(42)


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio=0.2):
    """数据集划分：按比例随机拆分训练集/测试集"""
    n = X.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return X[idx[test_size:]], y[idx[test_size:]], X[idx[:test_size]], y[idx[:test_size]]


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ============================
# TODO: 算法实现区域（学生填写）
# 目标：实现梯度下降训练线性回归，返回 (w, b, hist)
# 建议：BGD 每轮使用全部样本；可扩展为小批量 SGD
# def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200):
#     # YOUR CODE HERE
#     # return w, b, history_mse_list
#     raise NotImplementedError
# ============================

def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200):
    """
    随机梯度下降（SGD）训练线性回归
    参数：
        X: 训练特征（n_samples×n_features）
        y: 训练标签（n_samples×1）
        lr: 学习率（控制参数更新步长）
        epochs: 迭代轮数（全量样本遍历次数）
    返回：
        w: 特征权重（n_features×1）
        b: 偏置项（标量）
        history_mse_list: 每轮迭代的训练MSE（记录收敛过程）
    """
    n_samples, n_features = X.shape
    # 1. 初始化参数：权重w全0（与特征数一致），偏置b=0（标量）
    w = np.zeros(n_features)
    b = 0.0
    history_mse_list = []  # 存储每轮训练的MSE，用于绘制收敛曲线

    for _ in range(epochs):
        # 2. SGD核心：随机选择单个样本计算梯度并更新参数（减少计算量，提升泛化性）
        random_idx = np.random.randint(0, n_samples)  # 随机选1个样本索引
        x_i = X[random_idx]  # 单个样本特征（1×n_features）
        y_i = y[random_idx]  # 单个样本标签（标量）

        # 3. 计算预测值与误差
        y_pred_i = np.dot(x_i, w) + b  # 单个样本的预测值
        error = y_pred_i - y_i         # 单个样本的预测误差（真实值-预测值）

        # 4. 计算梯度并更新参数（SGD梯度公式）
        dw = error * x_i  # 权重w的梯度（误差×样本特征）
        db = error        # 偏置b的梯度（直接等于误差）
        w -= lr * dw      # 权重更新：步长=学习率×梯度
        b -= lr * db      # 偏置更新：同上

        # 5. 计算当前轮的全量训练MSE（评估收敛进度）
        y_train_pred = np.dot(X, w) + b
        train_mse = mse(y, y_train_pred)
        history_mse_list.append(train_mse)

    return w, b, history_mse_list
# ============================


def main():

    os.makedirs(OUT_DIR, exist_ok=True)


    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)


    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    X_train_norm, X_test_norm = normalize(X_train, X_test)


    lrs = [0.001, 0.01, 0.05, 0.1]
    epochs = 300
    curves = {}


    print("开始对比不同学习率的模型训练...")
    for lr in lrs:
        w, b, train_mse_hist = gd_train(
            X=X_train_norm,
            y=y_train,
            lr=lr,
            epochs=epochs
        )
        y_train_pred = np.dot(X_train_norm, w) + b
        y_test_pred = np.dot(X_test_norm, w) + b
        final_train_mse = mse(y_train, y_train_pred)
        final_test_mse = mse(y_test, y_test_pred)
        curves[lr] = {
            'hist': train_mse_hist,
            'train_mse': final_train_mse,
            'test_mse': final_test_mse
        }

        print(f"学习率lr={lr:.3f} -> 训练MSE={final_train_mse:.4f}, 测试MSE={final_test_mse:.4f}")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 5))
    for lr, info in curves.items():

        plt.plot(
            range(1, epochs + 1),
            info['hist'],
            linewidth=1.5,
            label=f'lr={lr}'
        )

    plt.xlabel('Epoch ', fontsize=10)
    plt.ylabel('Train MSE ', fontsize=10)
    plt.title('MSE Convergence Curves Under Different Learning Rates', fontsize=12)
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f"\n收敛曲线已保存至：{FIG_PATH}")


if __name__ == '__main__':
    main()