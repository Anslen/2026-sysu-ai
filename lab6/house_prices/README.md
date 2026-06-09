# 实验任务二：感知机房价预测

使用手写的单层感知机（线性回归 + MSE + mini-batch SGD）在 Kaggle
House Prices 数据集上完成购房价格回归训练，绘制训练/验证集的 loss 曲线
并分析过拟合情况。

## 运行

```bash
cd lab6/house_prices
uv sync
uv run python -m house_prices
```

产物：

- 终端输出每个 epoch 的训练/验证集 MSE 与 RMSE。
- `outputs/loss_curve.png` —— 训练集与验证集 MSE 曲线。
- `outputs/history.csv` —— 每个 epoch 的指标记录。

## 关键算法手写部分

| 模块 | 内容 |
| --- | --- |
| `preprocess.py` | 缺失值填充、异常值过滤（去噪）、one-hot 编码、z-score 标准化 |
| `model.py` | 前向 `y = X·w + b`、MSE 损失、解析梯度、mini-batch SGD、L2 正则 |
| `metrics.py` | RMSE / MAE / R² |
| `plot.py` | matplotlib 双曲线绘制 |

仅 `numpy` 用于矩阵运算、`matplotlib` 用于绘图，未使用任何机器学习库。
