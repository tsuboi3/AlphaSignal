from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import pandas as pd


@dataclass
class AlphaEngineConfig:
    d: int = 5  # 期待リターン算出ルックバック期間
    eps_clip: float = 1e-8  # ゼロ除算防止クリップ値
    l2_reg: float = 1e-4  # 特異値対策用Ridge正則化パラメータ


class AlphaEngine:
    def __init__(self, config: Optional[AlphaEngineConfig] = None):
        self.config = config or AlphaEngineConfig()

    def fit_weights(self, returns_matrix: pd.DataFrame) -> pd.Series:
        """過去リターン行列から最適結合ウェイト w(i) を計算する。

        Parameters
        ----------
        returns_matrix : pd.DataFrame
            形状 (M, N)。インデックス: 日時, カラム: シグナル名/銘柄コード。

        Returns
        -------
        pd.Series
            各シグナルのウェイト。L1ノルム合計が 1.0 となる。
        """
        if returns_matrix.empty or returns_matrix.shape[1] == 0:
            raise ValueError("returns_matrix is empty or has no columns.")

        # 直近 d 期間のデータを使用して期待リターン mu と共分散行列 sigma を計算
        recent_returns = returns_matrix.tail(self.config.d)
        if len(recent_returns) == 0:
            recent_returns = returns_matrix

        mu = recent_returns.mean().values  # (N,)
        cov = recent_returns.cov().values  # (N, N)

        # 欠損値やデータ数不足で cov に NaN が含まれる場合は単位行列に補正
        if np.isnan(cov).any() or cov.size == 0:
            cov = np.eye(returns_matrix.shape[1])
        if np.isnan(mu).any():
            mu = np.zeros(returns_matrix.shape[1])

        # Ridge正則化: Sigma_reg = Sigma + l2_reg * I
        n = returns_matrix.shape[1]
        cov_reg = cov + self.config.l2_reg * np.eye(n)

        # 最適ウェイト w = Sigma_reg^-1 * mu
        try:
            raw_weights = np.linalg.solve(cov_reg, mu)
        except np.linalg.LinAlgError:
            raw_weights = np.linalg.pinv(cov_reg) @ mu

        # L1ノルム規格化: |w| の和で割る
        l1_norm = np.sum(np.abs(raw_weights))
        if l1_norm < self.config.eps_clip:
            # すべての要素がほぼ0の場合、均等ウェイト
            weights = np.ones(n) / n
        else:
            weights = raw_weights / l1_norm

        return pd.Series(weights, index=returns_matrix.columns)

    def combine(
        self,
        current_signals: Union[pd.Series, pd.DataFrame],
        weights: pd.Series,
    ) -> Union[float, pd.Series]:
        """最新シグナル値と算出ウェイトを線形結合する。

        Parameters
        ----------
        current_signals : pd.Series or pd.DataFrame
            単一時点のシグナル値ベクトル (Series) または 複数銘柄のシグナル値 (DataFrame)
        weights : pd.Series
            fit_weights で算出された重み

        Returns
        -------
        Union[float, pd.Series]
            合成シグナル値
        """
        if isinstance(current_signals, pd.Series):
            # シグナル名/銘柄でインデックス合わせ
            aligned_weights = weights.reindex(current_signals.index).fillna(0.0)
            result = float((current_signals * aligned_weights).sum())
            return result
        elif isinstance(current_signals, pd.DataFrame):
            # 列名と weights のインデックスを合わせる
            aligned_weights = weights.reindex(current_signals.columns).fillna(0.0)
            result = current_signals.dot(aligned_weights)
            return result
        else:
            raise TypeError("current_signals must be pd.Series or pd.DataFrame")
