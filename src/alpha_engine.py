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

        m_rows, n_cols = returns_matrix.shape
        if m_rows <= self.config.d:
            raise ValueError(f"Input data row count M ({m_rows}) must be strictly greater than config.d ({self.config.d}).")

        # 欠損値・Inf処理: ffill 後、残存分を列平均値で補完
        returns_cleaned = returns_matrix.replace([np.inf, -np.inf], np.nan)
        returns_cleaned = returns_cleaned.ffill()
        returns_cleaned = returns_cleaned.fillna(returns_cleaned.mean()).fillna(0.0)

        # 直近 d 期間のデータを使用して期待リターン mu と共分散行列 sigma を計算
        recent_returns = returns_cleaned.tail(self.config.d)

        mu = recent_returns.mean().values  # (N,)
        cov = recent_returns.cov().values  # (N, N)

        # 欠損値やデータ数不足で cov に NaN が含まれる場合は単位行列に補正
        if np.isnan(cov).any() or cov.size == 0:
            cov = np.eye(n_cols)
        if np.isnan(mu).any():
            mu = np.zeros(n_cols)

        # 数値的安定性: lstsq をデフォルトとし、条件数が悪い（特異行列に近い）場合は正則化解法へフォールバック
        use_fallback = False
        try:
            cond = np.linalg.cond(cov)
            if np.isinf(cond) or cond > 1e12:
                use_fallback = True
            else:
                raw_weights, _, _, _ = np.linalg.lstsq(cov, mu, rcond=None)
        except np.linalg.LinAlgError:
            use_fallback = True

        if use_fallback:
            cov_reg = cov + self.config.l2_reg * np.eye(n_cols)
            try:
                raw_weights = np.linalg.solve(cov_reg, mu)
            except np.linalg.LinAlgError:
                raw_weights = np.linalg.pinv(cov_reg) @ mu

        # フェイルセーフ: sum(|w_raw|) < 10^-12 の場合は等加重 (w(i) = 1/N)
        l1_norm = np.sum(np.abs(raw_weights))
        if l1_norm < 1e-12:
            weights = np.ones(n_cols) / n_cols
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
