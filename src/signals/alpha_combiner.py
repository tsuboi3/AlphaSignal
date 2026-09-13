from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .factor_orthogonalizer import FactorOrthogonalizer


class AlphaCombiner:
    """
    Alpha Extraction and Signal Combination Pipeline based on
    'A Unified Framework for Alpha Extraction and Signal Combination'.
    """

    def __init__(self, method: str = "auto", alpha: float = 1e-4, min_vol: float = 1e-8):
        """
        Args:
            method: Orthogonalization method ('auto', 'ols', 'svd', 'ridge').
            alpha: Ridge regression parameter.
            min_vol: Minimum volatility threshold for clipping (default: 1e-8).
        """
        self.orthogonalizer = FactorOrthogonalizer(method=method, alpha=alpha)
        self.min_vol = min_vol

    def compute_pipeline(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        d: Optional[int] = None,
        latest_signals: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Run full alpha extraction and signal combination pipeline.

        Args:
            returns: Return matrix R of shape (M, N) or (N, M).
                     If pandas DataFrame, rows are timestamps (M) and columns are signals (N).
                     If 2D numpy array, shape is assumed to be (M, N) where M is time length
                     and N is number of signals. (If shape is (N, M) with N > M, it is preserved).
            d: Estimation window for expected returns (1 <= d <= M). Defaults to M.
            latest_signals: Latest signal values S_i for each signal (length N).
                            If None, defaults to the latest raw return R(i, M).

        Returns:
            Dictionary containing intermediate and final pipeline results:
            - 'drift_removed': X in R^{N x M}
            - 'volatility': sigma in R^N
            - 'standardized': Y in R^{N x M}
            - 'factor_matrix': Lambda in R^{N x (M-1)}
            - 'expected_returns': E in R^N
            - 'norm_expected_returns': E_norm in R^N
            - 'residual_alpha': epsilon in R^N
            - 'raw_weights': w_raw in R^N
            - 'weights': w in R^N (pd.Series if DataFrame input, else np.ndarray)
            - 'combined_signal': scalar float
        """
        is_df = isinstance(returns, pd.DataFrame)
        signal_names = None

        if is_df:
            signal_names = returns.columns
            # DataFrame shape: (M, N) -> time x signals
            # Transpose to (N, M) -> signals x time
            R = returns.values.T.astype(float)
        else:
            R = np.asarray(returns, dtype=float)
            if R.ndim != 2:
                raise ValueError(f"returns must be a 2D array or DataFrame, got shape {R.shape}")
            # Input matrix R is shape (M, N) where row dimension M is time steps
            # and column dimension N is number of signals/assets.
            # Transpose to (N, M) internally so row i is signal i and column s is time s.
            R = R.T

        N, M = R.shape

        if d is None:
            d = M
        if d < 1 or d > M:
            raise ValueError(f"Estimation window d must be between 1 and M ({M}), got {d}")

        # Eq 1: Time series drift removal
        # X(i, s) = R(i, s) - (1/M) * sum_s R(i, s)
        mean_R = np.mean(R, axis=1, keepdims=True)  # shape (N, 1)
        X = R - mean_R  # shape (N, M)

        # Eq 2: Volatility estimation
        # sigma_i^2 = (1/M) * sum_s X(i, s)^2
        sigma_sq = np.mean(X ** 2, axis=1)  # shape (N,)
        sigma = np.sqrt(sigma_sq)
        # Clipping for zero-division prevention
        sigma = np.maximum(sigma, self.min_vol)

        # Eq 3: Volatility standardization
        # Y(i, s) = X(i, s) / sigma_i
        Y = X / sigma[:, np.newaxis]  # shape (N, M)

        # Eq 4: Trimming observation period (s = 1, ..., M-1)
        Y_trimmed = Y[:, :-1]  # shape (N, M-1)

        # Eq 5 & 6: Cross-sectional mean removal
        # Lambda(i, s) = Y(i, s) - (1/N) * sum_j Y(j, s)
        cross_mean_Y = np.mean(Y_trimmed, axis=0, keepdims=True)  # shape (1, M-1)
        Lambda = Y_trimmed - cross_mean_Y  # shape (N, M-1)

        # Eq 7: Expected return estimation over recent d periods
        # E(i) = (1/d) * sum_{s=M-d+1}^M R(i, s)
        E = np.mean(R[:, -d:], axis=1)  # shape (N,)

        # Eq 8: Volatility-adjusted expected return
        # E_norm(i) = E(i) / sigma_i
        E_norm = E / sigma  # shape (N,)

        # Eq 9: Factor orthogonalization and residual alpha extraction
        # E_norm = Lambda * beta + epsilon
        epsilon = self.orthogonalizer.extract_residuals(Lambda, E_norm)  # shape (N,)

        # Eq 10 & 11: Volatility inverse-proportional weights
        # w_raw(i) = epsilon(i) / sigma_i
        w_raw = epsilon / sigma

        abs_sum_w_raw = np.sum(np.abs(w_raw))
        if abs_sum_w_raw < 1e-12:
            # Fallback to equal weights if raw weights sum to near zero
            w = np.ones(N) / N
        else:
            w = w_raw / abs_sum_w_raw

        # Eq 12: Combined signal calculation
        if latest_signals is None:
            # Default to latest raw return R(i, M)
            S = R[:, -1]
        else:
            if isinstance(latest_signals, (pd.Series, pd.DataFrame)):
                S = latest_signals.values.ravel().astype(float)
            else:
                S = np.asarray(latest_signals, dtype=float).ravel()

            if len(S) != N:
                raise ValueError(f"latest_signals length ({len(S)}) does not match signal count N ({N})")

        combined_signal = float(np.sum(w * S))

        weights_res = pd.Series(w, index=signal_names) if signal_names is not None else w

        return {
            "drift_removed": X,
            "volatility": sigma,
            "standardized": Y,
            "factor_matrix": Lambda,
            "expected_returns": E,
            "norm_expected_returns": E_norm,
            "residual_alpha": epsilon,
            "raw_weights": w_raw,
            "weights": weights_res,
            "combined_signal": combined_signal,
        }

    def combine(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        d: Optional[int] = None,
        latest_signals: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
    ) -> float:
        """
        Calculate and return combined signal value.
        """
        res = self.compute_pipeline(returns, d=d, latest_signals=latest_signals)
        return res["combined_signal"]

    def get_weights(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        d: Optional[int] = None,
    ) -> Union[np.ndarray, pd.Series]:
        """
        Calculate and return portfolio/signal weights.
        """
        res = self.compute_pipeline(returns, d=d)
        return res["weights"]
