import numpy as np


class FactorOrthogonalizer:
    """
    Factor Orthogonalization and Residual Alpha Extraction module.

    Solves E_norm = Lambda * beta + epsilon to extract the residual alpha vector epsilon in R^N.
    """

    def __init__(self, method: str = "auto", alpha: float = 1e-4):
        """
        Args:
            method: Regression method ('auto', 'ols', 'svd', 'ridge').
            alpha: Regularization strength for Ridge regression.
        """
        self.method = method
        self.alpha = alpha

    def extract_residuals(self, lambda_matrix: np.ndarray, e_norm: np.ndarray) -> np.ndarray:
        """
        Extract residual vector epsilon from E_norm = Lambda * beta + epsilon.

        Args:
            lambda_matrix: Factor matrix Lambda of shape (N, M-1).
            e_norm: Volatility-adjusted expected return vector of shape (N,).

        Returns:
            Residual vector epsilon of shape (N,).
        """
        Lambda = np.asarray(lambda_matrix, dtype=float)
        e_norm = np.asarray(e_norm, dtype=float).ravel()

        N, K = Lambda.shape

        method = self.method.lower()

        if method == "auto":
            try:
                beta, _, rank, _ = np.linalg.lstsq(Lambda, e_norm, rcond=None)
                if rank < min(N, K):
                    # Rank deficient; fallback to pseudo-inverse (SVD)
                    beta = np.linalg.pinv(Lambda) @ e_norm
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(Lambda) @ e_norm
        elif method == "ols":
            beta, _, _, _ = np.linalg.lstsq(Lambda, e_norm, rcond=None)
        elif method == "svd":
            beta = np.linalg.pinv(Lambda) @ e_norm
        elif method == "ridge":
            gram = Lambda.T @ Lambda
            reg_gram = gram + self.alpha * np.eye(K)
            beta = np.linalg.solve(reg_gram, Lambda.T @ e_norm)
        else:
            raise ValueError(f"Unknown orthogonalization method: {self.method}")

        epsilon = e_norm - Lambda @ beta
        return epsilon
