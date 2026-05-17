#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：任务2 梯度下降（BGD/SGD）
- 目的：给学生手写实现线性回归的梯度下降训练（任选 BGD 或 SGD）
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
FIG_PATH = os.path.join(OUT_DIR, 'task2_mse_curve_template.png')
np.random.seed(42)

def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio=0.2):
    n = X.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return X[idx[test_size:]], y[idx[test_size:]], X[idx[:test_size]], y[idx[:test_size]]

def normalize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

# ============================
# 算法实现区域（SGD实现）
def gd_train(X: np.ndarray, y: np.ndarray, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    # 1. 初始化参数（权重w，偏置b）
    w = np.zeros(n_features)
    b = 0.0
    history_mse_list = []  # 记录每次迭代的训练MSE

    for epoch in range(epochs):
        # 2. SGD核心：随机选择单个样本更新参数
        idx = np.random.randint(0, n_samples)  # 随机选一个样本索引
        x_i = X[idx]
        y_i = y[idx]

        # 3. 计算预测值和误差
        y_pred_i = np.dot(x_i, w) + b
        error = y_pred_i - y_i

        # 4. 计算梯度并更新参数
        dw = error * x_i  # 单个样本的权重梯度
        db = error        # 单个样本的偏置梯度
        w -= lr * dw
        b -= lr * db

        # 5. 计算当前迭代的训练MSE（全量计算，记录收敛趋势）
        y_train_pred = np.dot(X, w) + b
        train_mse = np.mean((y - y_train_pred) ** 2)
        history_mse_list.append(train_mse)

    return w, b, history_mse_list
# ============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    X_train, X_test = normalize(X_train, X_test)

    w, b, hist = gd_train(X_train, y_train, lr=0.05, epochs=300)

    y_train_pred = X_train @ w + b
    y_test_pred = X_test @ w + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    print(f'Train MSE: {train_mse:.4f}')
    print(f'Test MSE:  {test_mse:.4f}')

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist) + 1), hist, label='Train MSE', color='#1f77b4')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('SGD Convergence Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('Saved figure:', FIG_PATH)

if __name__ == '__main__':
    main()