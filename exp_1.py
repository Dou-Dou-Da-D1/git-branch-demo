import numpy as np
import time
from scipy.ndimage import gaussian_filter


def preprocess_data(X):
    # 标准化 + 平滑
    X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X_denoised = np.array([
        gaussian_filter(x.reshape(16, 16), sigma=0.5).flatten()
        for x in X_normalized
    ])
    return X_denoised


def compute_weighted_distance(x1, x2):
    # 中心区域加权
    weights = np.ones(256)
    center_indices = [i for i in range(256) if 4 <= i % 16 <= 11 and 4 <= i // 16 <= 11]
    weights[center_indices] *= 1.5
    diff = (x1 - x2) ** 2
    return np.sqrt(np.sum(weights * diff))


def k_nearest_neighbors_advanced(train_X, train_y, test_x, k):
    distances = []
    for i in range(len(train_X)):
        dist = compute_weighted_distance(test_x, train_X[i])
        distances.append((train_y[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    class_weights = {}
    for label, dist in neighbors:
        w = 1.0 / (dist + 1e-8)
        class_weights[label] = class_weights.get(label, 0.0) + w
    return max(class_weights, key=class_weights.get)


def loo_eval_advanced(X, y, k):
    X_processed = preprocess_data(X)
    correct = 0
    total = len(X_processed)
    print(f"--- 开始使用改进算法 k={k} 进行留一法交叉验证 ---")
    start = time.time()
    for i in range(total):
        test_x = X_processed[i]
        test_y = y[i]
        train_X = np.delete(X_processed, i, axis=0)
        train_y = np.delete(y, i, axis=0)
        pred = k_nearest_neighbors_advanced(train_X, train_y, test_x, k)
        if pred == test_y:
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"完成 {i + 1}/{total} 样本...")
    end = time.time()
    acc = correct / total
    print(f"--- 验证完成，耗时: {end - start:.2f} 秒 ---")
    return acc


if __name__ == "__main__":
    print("正在加载数据集...")
    try:
        raw_data = np.loadtxt('semeion.data.txt')
        X = raw_data[:, :256]
        y = np.argmax(raw_data[:, 256:], axis=1)
        print(f"数据集加载成功！共 {len(X)} 个样本。")
    except FileNotFoundError:
        print("错误：找不到 semeion_train.txt 文件！")
        exit()

    for k in [1, 3, 5]:
        acc = loo_eval_advanced(X, y, k)
        print(f"改进算法当 k = {k} 时，留一法交叉验证的准确率为: {acc:.4f}\n")
