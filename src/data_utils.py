# src/data_utils.py

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    RED_WINE_PATH,
    WHITE_WINE_PATH,
    RANDOM_STATE,
    GOOD_THRESHOLD,
)


def load_wine_data() -> pd.DataFrame:
    """
    读取红酒和白酒两个数据集, 合并为一个 DataFrame,
    并添加 wine_type 特征, 取值为 'red' 或 'white'。
    """
    
    red = pd.read_csv(RED_WINE_PATH, sep=None, engine="python")
    red["wine_type"] = "red"

    white = pd.read_csv(WHITE_WINE_PATH, sep=None, engine="python")
    white["wine_type"] = "white"


    df = pd.concat([red, white], ignore_index=True)
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始数据上添加一个二分类目标列 is_good:
    当 quality >= GOOD_THRESHOLD 时 is_good = 1, 否则 0。
    保留原始 quality 列, 以便将来做回归实验或对比。
    """
    df = df.copy()
    df["is_good"] = (df["quality"] >= GOOD_THRESHOLD).astype(int)
    return df


def train_test_split_wine(
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    完整的数据加载 + 目标生成 + 划分训练集与测试集。

    返回:
        X_train, X_test, y_train, y_test
    """
    df = load_wine_data()
    df = add_binary_target(df)

    # 特征与目标
    feature_cols = [c for c in df.columns if c not in ["quality", "is_good"]]

    X = df[feature_cols]
    y = df["is_good"]

    # 对 wine_type 做 one-hot 编码 (red / white), 其他列皆为数值
    X = pd.get_dummies(X, columns=["wine_type"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
