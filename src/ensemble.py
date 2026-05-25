"""
アンサンブル・評価モジュール
LightGBM + Transformer の予測値をスタッキングで統合し、評価指標を算出
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict


def ensemble_predictions(
    lgbm_preds: np.ndarray,
    transformer_preds: np.ndarray,
    y_true: np.ndarray,
    nis_scores: np.ndarray = None,
) -> Tuple[np.ndarray, LinearRegression]:
    """
    線形回帰スタッキングによるアンサンブル

    Args:
        lgbm_preds:       LightGBM予測値
        transformer_preds: Transformer予測値
        y_true:           正解値（メタ学習用）
        nis_scores:       ニュースインパクト指数（オプション）

    Returns:
        ensemble_preds: アンサンブル予測値
        meta_model:     学習済みメタモデル
    """
    meta_features = [lgbm_preds.reshape(-1, 1), transformer_preds.reshape(-1, 1)]
    if nis_scores is not None:
        meta_features.append(nis_scores.reshape(-1, 1))

    X_meta = np.hstack(meta_features)

    # 訓練データの前半でメタモデルを学習
    split = len(X_meta) // 2
    meta_model = LinearRegression()
    meta_model.fit(X_meta[:split], y_true[:split])

    ensemble_preds = meta_model.predict(X_meta)
    return ensemble_preds, meta_model


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """方向性予測精度（騰落方向の一致率）"""
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    return np.mean(direction_true == direction_pred)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """モデル評価指標の計算"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    dir_acc = directional_accuracy(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE(%)': mape,
        'DirectionalAcc': dir_acc,
        'Correlation': corr,
    }

    print(f"\n  【{model_name}】")
    print(f"    RMSE         : {rmse:.4f}")
    print(f"    MAE          : {mae:.4f}")
    print(f"    MAPE         : {mape:.2f}%")
    print(f"    方向性精度   : {dir_acc:.4f} ({dir_acc * 100:.1f}%)")
    print(f"    相関係数     : {corr:.4f}")

    return metrics


def backtest_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    initial_capital: float = 100_000,
) -> Dict[str, float]:
    """
    シンプルな取引戦略バックテスト
    予測騰落方向に基づくロング/ショート戦略
    """
    signals = np.sign(np.diff(y_pred))  # 1: 買い, -1: 売り
    actual_returns = np.diff(prices) / prices[:-1]

    # シグナルに合わせたリターン（手数料0.1%）
    fee = 0.001
    strategy_returns = signals * actual_returns - np.abs(signals) * fee

    # 累積リターン
    cumulative = np.cumprod(1 + strategy_returns)
    total_return = cumulative[-1] - 1

    # シャープレシオ（年換算、252営業日）
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    # 最大ドローダウン
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    results = {
        'TotalReturn(%)': total_return * 100,
        'SharpeRatio': sharpe,
        'MaxDrawdown(%)': max_drawdown * 100,
    }

    print(f"    総リターン   : {total_return * 100:.2f}%")
    print(f"    シャープレシオ: {sharpe:.4f}")
    print(f"    最大ドローダウン: {max_drawdown * 100:.2f}%")

    return results
