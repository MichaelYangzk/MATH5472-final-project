# src/train_and_evaluate.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RESULTS_DIR, RANDOM_STATE
from .data_utils import train_test_split_wine


def build_models() -> Dict[str, Any]:
    """
    构建四个待比较的分类模型:

    1. Logistic Regression (带标准化)
    2. 单棵决策树
    3. 随机森林
    4. 梯度提升树 (Gradient Boosting)
    """
    models: Dict[str, Any] = {}

    # 1. Logistic Regression (线性基准模型)
    models["logistic_regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 2. 单棵决策树 (high variance baseline)
    models["decision_tree"] = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
    )

    # 3. Random Forest (主角)
    models["random_forest"] = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # 4. Gradient Boosting (竞争者)
    models["gradient_boosting"] = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )

    return models


def evaluate_model(
    name: str,
    model: Any,
    X_test,
    y_test,
    results_dir: Path,
) -> Dict[str, float]:
    """
    对单个模型进行预测和评估, 返回各项指标, 并绘制混淆矩阵。
    """
    y_pred = model.predict(X_test)

    # 某些模型可能没有 predict_proba, 这里做兼容处理
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        try:
            scores = model.decision_function(X_test)
            # 线性拉伸到 0 到 1 区间
            scores_min = scores.min()
            scores_max = scores.max()
            if scores_max > scores_min:
                y_prob = (scores - scores_min) / (scores_max - scores_min)
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                roc_auc = float("nan")
        except Exception:
            roc_auc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 混淆矩阵图
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {name}")
    cm_path = results_dir / f"confusion_matrix_{name}.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    metrics = {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    return metrics


def plot_feature_importance(
    model,
    feature_names: List[str],
    title: str,
    out_path: Path,
):
    """
    绘制具有 feature_importances_ 属性模型的特征重要性条形图。
    例如 RandomForestClassifier, GradientBoostingClassifier。
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances)
    plt.xticks(range(len(sorted_importances)), sorted_names, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    # 确保结果目录存在
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = train_test_split_wine(test_size=0.2)

    feature_names = list(X_train.columns)

    print("Building models...")
    models = build_models()

    all_metrics: List[Dict[str, float]] = []

    print("Training and evaluating models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        print(f"Evaluating {name}...")
        metrics = evaluate_model(
            name=name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            results_dir=RESULTS_DIR,
        )
        all_metrics.append(metrics)

        # 特征重要性图: 仅对有 feature_importances_ 属性的模型绘制
        if hasattr(model, "feature_importances_"):
            out_path = RESULTS_DIR / f"feature_importance_{name}.png"
            plot_feature_importance(
                model=model,
                feature_names=feature_names,
                title=f"Feature Importance: {name}",
                out_path=out_path,
            )

    # 汇总各模型指标到 CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
