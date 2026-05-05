# AGENTS.md — TSP 旅行商问题（遗传算法求解）

## 环境配置

**包管理器**：使用 `uv`（非 pip）。

```bash
# 在 tsp/ 目录下安装依赖
uv sync

# 运行程序
uv run python src/main.py                    # 默认 data/qa194.tsp
uv run python src/main.py data/other.tsp     # 指定数据文件

# 或直接使用 venv 中的解释器
.venv/bin/python src/main.py [参数]
```

**Python 版本**：`3.14+freethreaded`（自由线程，无 GIL）。定义在 `.python-version` 中。

**关键影响**：无 GIL 意味着 `ThreadPoolExecutor` 可实现真正的多核并行。**但 `random` 模块仍然不是线程安全的**——多线程同时调用 `random.random()` / `random.choice()` 会导致状态损坏。

**工作目录**：始终从 `tsp/` 根目录运行命令，不要 cd 到 `src/` 内部。

---

## 项目结构

```
tsp/
├── pyproject.toml           # 依赖：numpy>=2.4.4
├── .python-version          # 3.14+freethreaded
├── AGENTS.md                # 本文件
├── src/
│   ├── main.py              # 入口
│   ├── ga.py                # 遗传算法核心（选择、交叉、变异、种群进化）
│   ├── chromosome.py         # 染色体类 + OX 交叉 / 逆转变异算子
│   ├── tsp_data.py           # TSPLIB 数据解析 + 距离矩阵构建
│   └── tee_logger.py         # 日志工具（stdout + 文件双写，即时 flush）
└── data/
    └── qa194.tsp             # 194 城市 TSP 数据（Qatar）
```

---

## 代码风格约定

- `from __future__ import annotations` — 所有文件首行
- **全部类型注解**：`x: int = 0`，`data: list[str] = []`
- **现代语法**：`list[T]` 而非 `List[T]`，`dict[K,V]` 而非 `Dict[K,V]`
- `__slots__` — 数据类用它减少内存
- **模块级常量**大写：`POPULATION_SIZE: int = 100`
- 私有变量/方法用 `_` 前缀
- 英文注释和 docstring

**导入规范**（从 `tsp/` 根目录导入）：
```python
from src.ga import GeneticAlgorithm
from src.tee_logger import TeeLogger
```

---

## 遗传算法参数

可调参数集中在 `src/ga.py` 模块级常量中：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `POPULATION_SIZE` | 100 | 种群大小 |
| `MAX_GENERATIONS` | 10000 | 最大代数 |
| `MUTATION_RATE` | 0.1 | 变异概率 |
| `TOURNAMENT_SIZE` | 3 | 锦标赛选择竞争人数 |
| `ELITISM_COUNT` | 2 | 每代保留的最优个体数 |
| `REPORT_INTERVAL` | 100 | 每隔 N 代输出一次进度 |
| `CROSSOVER_METHOD` | `"ox"` | 交叉方法（Order Crossover） |
| `MUTATION_METHOD` | `"inversion"` | 变异方法（逆转变异） |
| `SEED` | `None` | 随机种子，设为整数可复现结果 |
| `THREAD_WORKERS` | `os.cpu_count()` | 进化并行线程数，默认使用全部 CPU 核心 |

---

## 性能设计

### 基因存储
基因排列使用 `tuple[int, ...]` 而非 `list[int]`，减少内存分配压力（元组不可变，`__copy__` 可直接复用）。

### 适应度评估：NumPy 向量化
`_evaluate` 和 `_report` 不再逐染色体循环，而是把所有基因堆成 `(100, 194)` 矩阵，通过 `np.roll` + 高级索引一次算出全部距离：
```python
genes = np.array([c.genes for c in population])       # (100, 194)
dists = dist_matrix[genes, np.roll(genes, -1, axis=1)].sum(axis=1)
```
避免 100 万次 Python 函数调用，速度提升 30–50 倍。

### 进化：分块并行 + 线程本地随机
`_evolve_generation` 将后代对的生成任务分成 `THREAD_WORKERS` 个 chunk，每个 chunk 在一个线程中顺序生产多对后代，共用一个线程本地的 `random.Random()` 实例：

- **任务数**：从 49 个 `submit`（每对一个）降到 `THREAD_WORKERS` 个（每线程一个 chunk）
- **线程安全**：`chromosome.py` 提供 `_rng()` 函数，为每个线程自动创建独立的 `Random` 实例
- **可复现性**：`_rng_seed_source` 由 `SEED` 初始化，所有线程的种子序列在 `SEED` 固定时可复现

---

## 多线程

- `_evolve_generation` 使用 `ThreadPoolExecutor` 分块并行，每个线程用 `_rng()` 获取独立 RNG
- `_evaluate` 和 `_report` 使用 NumPy 向量化（非多线程），由其内部 BLAS 自动利用多核
- **`random` 模块本身不是线程安全的**——所有随机操作必须通过 `_rng()` 或 `chromosome.py` 中的 `seed_rng()` 获取线程本地实例

---

## 日志

- 每次运行自动在 `log/` 目录创建时间戳日志文件（如 `log/tsp_20260505_143022.log`）
- 每条输出后立即 `flush`，程序中断不丢失已输出的记录
- 日志包含：运行起始时间、命令行参数、所有算法参数、每代进度、最终结果

---

## 数据格式

- TSP 数据文件遵循 TSPLIB 格式
- 仅支持 `EUC_2D`（二维欧几里得距离）
- 解析逻辑在 `src/tsp_data.py` 的 `read_tsp()` 和 `build_distance_matrix()` 中
