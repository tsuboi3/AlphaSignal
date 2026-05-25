"""
LightGBMモデルモジュール
勾配ブースティング決定木による株価予測
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def build_and_train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> lgb.Booster:
    """
    LightGBMモデルの構築と学習

    Args:
        X_train, y_train: 訓練データ
        X_val, y_val:     検証データ

    Returns:
        学習済みLightGBM Booster
    """
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
    }

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    print(f"  → 最適イテレーション数: {model.best_iteration}")
    return model
