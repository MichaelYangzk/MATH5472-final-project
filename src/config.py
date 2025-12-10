# src/config.py

from pathlib import Path

# 项目根目录: src 的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RED_WINE_PATH = DATA_DIR / "winequality-red.csv"
WHITE_WINE_PATH = DATA_DIR / "winequality-white.csv"

# 结果输出路径
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# 随机种子, 保证可复现
RANDOM_STATE = 42

# 将回归的质量分数转为二分类:
# quality >= GOOD_THRESHOLD 视为 "good wine" (1), 否则 0
GOOD_THRESHOLD = 7

# 模型列表名称, 用于输出和作图
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
]
