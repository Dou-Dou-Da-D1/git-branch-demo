#!/usr/bin/env python3
"""
拓展任务：多项式回归（基于poly_demo.csv）
- 功能：生成多项式特征、闭式解训练模型、多阶数对比、可视化拟合曲线
- 数据集：poly_demo.csv（单特征x+目标y）
- 核心：实现多项式特征生成与线性回归闭式解（pinv增稳）
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
CSV = os.path.join(DATA_DIR, 'poly_demo.csv')
FIG_PATH = os.path.join(OUT_DIR, 'task_ext_poly_fits_poly_demo.png')
np.random.seed(42)


def train_test_split_1d(x: np.ndarray, y: np.ndarray, test_ratio=0.2):
    n = x.shape[0]
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    return x[idx[test_size:]], y[idx[test_size:]], x[idx[:test_size]], y[idx[:test_size]]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# ============================
# 核心算法实现（多项式特征生成 + 闭式解训练）
def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """
    生成多项式特征：将1D特征x扩展为[1, x, x², ..., x^degree]
    参数：
        x: 1D输入特征（shape=(n_samples,)）
        degree: 多项式阶数（≥1）
    返回：
        X_poly: 多项式特征矩阵（shape=(n_samples, degree+1)），第一列为常数项1
    """
    # 处理x维度：从1D(n,)转为2D(n,1)，便于后续幂次计算
    x_2d = x.reshape(-1, 1)  # shape: (n_samples, 1)
    # 初始化特征矩阵：第一列为常数项1（对应偏置项）
    X_poly = np.ones((x_2d.shape[0], 1))

    # 循环生成x^1到x^degree的特征列，拼接至特征矩阵
    for d in range(1, degree + 1):
        x_power = x_2d ** d  # 计算x的d次幂
        X_poly = np.hstack([X_poly, x_power])  # 拼接特征列

    return X_poly


def fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    线性回归闭式解（最小二乘法）：适用于多项式回归（多项式回归本质是线性回归）
    参数：
        X: 特征矩阵（含常数项，shape=(n_samples, n_features)）
        y: 目标变量（shape=(n_samples,)）
    返回：
        theta: 模型参数（shape=(n_features,)），对应各特征的权重
    说明：使用np.linalg.pinv求伪逆，避免X^T X不可逆的数值问题（增稳）
    """
    # 闭式解公式：θ = (X^T X)^+ X^T y，其中^+表示伪逆
    X_T = X.T
    X_T_X = X_T @ X
    # 求伪逆（比inv更稳定，处理奇异矩阵）
    X_T_X_pinv = np.linalg.pinv(X_T_X)
    theta = X_T_X_pinv @ X_T @ y
    return theta


# ============================


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV)
    x = df['x'].to_numpy().astype(float)
    y = df['y'].to_numpy().astype(float)
    print(f"数据集规模：x.shape={x.shape}, y.shape={y.shape}")

    x_train, y_train, x_test, y_test = train_test_split_1d(x, y, test_ratio=0.2)
    mean_x = x_train.mean()
    std_x = x_train.std() + 1e-8
    x_train_std = (x_train - mean_x) / std_x
    x_test_std = (x_test - mean_x) / std_x

    degrees = [1, 2, 4, 6, 8]
    poly_results = {}

    print("\n" + "=" * 80)
    print(f"{'多项式阶数':<12} {'训练集MSE':<12} {'测试集MSE':<12}")
    print("=" * 80)
    for deg in degrees:
        X_train_poly = poly_features(x_train_std, deg)
        X_test_poly = poly_features(x_test_std, deg)

        theta = fit_closed_form(X_train_poly, y_train)

        y_train_pred = X_train_poly @ theta
        y_test_pred = X_test_poly @ theta
        train_mse = mse(y_train, y_train_pred)
        test_mse = mse(y_test, y_test_pred)

        poly_results[deg] = {
            'theta': theta,
            'train_mse': train_mse,
            'test_mse': test_mse
        }
        print(f"{deg:<12} {train_mse:<12.4f} {test_mse:<12.4f}")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train_std, y_train, s=20, color='#1f77b4', alpha=0.6, label='训练集数据')
    plt.scatter(x_test_std, y_test, s=20, color='#ff7f0e', alpha=0.6, label='测试集数据')

    xs = np.linspace(x_train_std.min() - 0.1, x_train_std.max() + 0.1, 500)  # 覆盖并扩展训练集范围
    colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#d62728', '#343a40', ]  # 曲线颜色区分
    for i, deg in enumerate(degrees):
        Xs_poly = poly_features(xs, deg)
        ys = Xs_poly @ poly_results[deg]['theta']
        plt.plot(
            xs, ys,
            color=colors[i],
            linewidth=2,
            label=f'阶数={deg}（训练MSE={poly_results[deg]["train_mse"]:.3f}）'
        )

    plt.xlabel('标准化后的x（Standardized x）', fontsize=11)
    plt.ylabel('目标y', fontsize=11)
    plt.title('Poly_demo.csv 多项式回归拟合曲线对比', fontsize=12)
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f"\n拟合曲线已保存至：{FIG_PATH}")


if __name__ == '__main__':
    main()