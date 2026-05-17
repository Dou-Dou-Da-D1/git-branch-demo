#!/usr/bin/env python3
"""
模板（不依赖第三方ML库）：任务4 岭回归（解析法）
- 目的：让学生实现岭回归的闭式解 (X^T X + λI)^{-1} X^T y
- 本模板仅保留数据读取、划分与评估骨架；算法实现留白
- 允许使用 numpy / pandas，禁止使用 sklearn
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CSV = os.path.join(DATA_DIR, 'winequality-white.csv')
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


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ============================
# TODO: 算法实现区域（学生填写）
# 目标：实现岭回归闭式解，返回 (w, b)
# 提示：在特征后追加一列常数 1 以表示偏置 b
# def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
#     # YOUR CODE HERE: 返回包含 w 和 b 的向量 theta
#     # 形如: theta = (X'X + lam * I)^(-1) X'y
#     raise NotImplementedError
# ============================
def ridge_fit_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    岭回归闭式解（解析法）实现
    参数：
        X: 扩展后的训练特征（含常数列，shape=(n_samples, n_features+1)，常数列在最后一列）
        y: 训练标签（shape=(n_samples,)）
        lam: 正则化参数（λ≥0，值越大正则化强度越强）
    返回：
        theta: 模型参数（shape=(n_features+1,)），前n_features个为权重w，最后1个为偏置b
    核心公式：θ = (X^T X + λ·I')^{-1} X^T y，其中I'为单位矩阵（偏置对应位置设为0，不参与正则化）
    """
    n_features = X.shape[1]  # 特征数（含常数列）
    # 1. 构造正则化单位矩阵I'：偏置项（最后一列）不参与正则化，对应位置设为0
    identity = np.eye(n_features)  # 初始单位矩阵
    identity[-1, -1] = 0  # 常数列对应theta的最后一个元素（b），故将I'的最后一行最后一列设为0

    # 2. 计算岭回归闭式解
    X_T = X.T  # X的转置矩阵
    X_T_X = X_T @ X  # X^T X（普通线性回归的中间项）
    # 加入正则化项：X^T X + λ·I'
    X_T_X_reg = X_T_X + lam * identity
    # 求逆矩阵（正则化确保矩阵可逆，避免多重共线性问题）
    X_T_X_reg_inv = np.linalg.inv(X_T_X_reg)
    # 计算最终参数θ
    theta = X_T_X_reg_inv @ X_T @ y

    return theta


# ============================


def main():
    df = pd.read_csv(CSV, sep=';')
    X = df.iloc[:, :-1].to_numpy().astype(float)
    y = df.iloc[:, -1].to_numpy().astype(float)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

    X_train_norm, X_test_norm = normalize(X_train, X_test)


    X_train_ext = np.column_stack([X_train_norm, np.ones(X_train_norm.shape[0])])
    X_test_ext = np.column_stack([X_test_norm, np.ones(X_test_norm.shape[0])])

    lam_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    print("=" * 80)
    print(f"{'λ值':<8} {'训练MSE':<12} {'测试MSE':<12} {'权重绝对值均值':<16}")
    print("=" * 80)

    for lam in lam_list:

        theta = ridge_fit_closed_form(X_train_ext, y_train, lam=lam)

        w = theta[:-1]
        b = float(theta[-1])


        y_train_pred = X_train_norm @ w + b
        y_test_pred = X_test_norm @ w + b

        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)

        w_abs_mean = np.mean(np.abs(w))

        print(f"{lam:<8.3f} {train_mse:<12.4f} {test_mse:<12.4f} {w_abs_mean:<16.6f}")

    best_lam = 0.1
    best_theta = ridge_fit_closed_form(X_train_ext, y_train, lam=best_lam)
    best_w = best_theta[:-1]
    best_b = float(best_theta[-1])
    best_train_mse = mse(y_train, X_train_norm @ best_w + best_b)
    best_test_mse = mse(y_test, X_test_norm @ best_w + best_b)
    print("\n" + "=" * 80)
    print(f"最优正则化参数λ={best_lam}")
    print(f"最优模型：权重w（前5个）={best_w[:5].round(4)}...，偏置b={best_b:.4f}")
    print(f"最优模型训练MSE={best_train_mse:.4f}，测试MSE={best_test_mse:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()