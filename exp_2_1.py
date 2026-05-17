#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：任务1 正规方程 Normal Equation
- 目的：给学生手写实现 θ = (X^T X)^{-1} X^T y 的位置
- 本模板仅保留数据读取、划分与可视化骨架；算法实现留白
- 允许使用 numpy / pandas / matplotlib，禁止使用 sklearn 等ML库
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
REG_CSV = os.path.join(DATA_DIR, 'dataset_regression.csv')
FIG_PATH = os.path.join(OUT_DIR, 'task1_fit_template.png')
np.random.seed(42)

def train_test_split_xy(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2):
    n = x.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return x[idx[test_size:]], y[idx[test_size:]], x[idx[:test_size]], y[idx[:test_size]]

# ============================
# 算法实现区域
# 目标：实现正规方程，返回 w, b 使 y ≈ w x + b（单特征）
# 核心：构造设计矩阵 Xb = [x, 1]，theta = (Xb^T Xb)^{-1} Xb^T y
def normal_equation_fit(x_train: np.ndarray, y_train: np.ndarray):
    # 1. 构造设计矩阵 Xb（n行2列：第一列x，第二列全1）
    n_samples = x_train.shape[0]
    Xb = np.hstack([x_train.reshape(-1, 1), np.ones((n_samples, 1))])
    # 2. 计算正规方程核心步骤
    Xb_T = Xb.T
    Xb_T_Xb = Xb_T @ Xb
    Xb_T_y = Xb_T @ y_train
    # 3. 求逆并得到参数 theta（[w, b]^T）
    theta = np.linalg.pinv(Xb_T_Xb) @ Xb_T_y
    # 4. 返回权重w和偏置b
    return theta[0], theta[1]
# ============================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(REG_CSV):
        print('Missing data file at:', REG_CSV)
        print('Please place dataset_regression.csv under data/.')
        return

    df = pd.read_csv(REG_CSV)
    num_df = df.select_dtypes(include=[np.number])
    x = num_df.iloc[:, 0].to_numpy().astype(float)
    y = num_df.iloc[:, 1].to_numpy().astype(float)

    x_train, y_train, x_test, y_test = train_test_split_xy(x, y, test_ratio=0.2)

    w, b = normal_equation_fit(x_train, y_train)

    y_train_pred = w * x_train + b
    y_test_pred = w * x_test + b
    train_mse = float(np.mean((y_train - y_train_pred) ** 2))
    test_mse = float(np.mean((y_test - y_test_pred) ** 2))
    print(f'Train MSE: {train_mse:.4f}\nTest MSE: {test_mse:.4f}')

    x_new = np.array([-12, -3, 0, 4.5, 11])
    y_new_pred = w * x_new + b
    print('\n5 new predictions:')
    for xi, yi in zip(x_new, y_new_pred):
        print(f'  x={xi:.2f} -> y_pred={yi:.3f}')

    plt.figure(figsize=(6, 4))
    plt.scatter(x_train, y_train, s=12, color='#1f77b4', label='Train data')
    xs = np.linspace(np.min(x_train), np.max(x_train), 200)
    ys = w * xs + b
    plt.plot(xs, ys, color='#d62728', label=f'Fit: y={w:.3f}x+{b:.3f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print('\nSaved figure:', FIG_PATH)

if __name__ == '__main__':
    main()