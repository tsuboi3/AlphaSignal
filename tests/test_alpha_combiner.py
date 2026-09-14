import pytest
import numpy as np
import pandas as pd

from src.signals.alpha_combiner import AlphaCombiner
from src.signals.factor_orthogonalizer import FactorOrthogonalizer


def test_l1_norm_constraint():
    r"""
    テスト1: L1ノルム制約検証
    任意のランダムリターン行列に対し、出力ウェイトの絶対値和 \sum |w(i)| が許容誤差 10^-6 以内で 1.0 に収束すること。
    """
    np.random.seed(42)
    M, N = 30, 5
    R = np.random.randn(M, N)
    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(R)

    weights = res["weights"]
    l1_norm = np.sum(np.abs(weights))
    assert pytest.approx(l1_norm, abs=1e-6) == 1.0


def test_market_common_factor_removal():
    """
    テスト2: 市場共通ファクター除去の検証
    全銘柄に同一の共通トレンド（+2% ドリフト）を付与した場合でも、
    式（5）のクロスセクショナル平均除去により共通項が完全にゼロへ相殺されること。
    """
    np.random.seed(42)
    M, N = 20, 6
    # 基本のランダムリターン
    base_returns = np.random.randn(M, N)
    # 全銘柄に同一の共通トレンド（+2% ドリフト）を付与
    common_trend = 0.02
    R = base_returns + common_trend

    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(R)

    Lambda = res["factor_matrix"]  # N x (M-1)
    # 各タイムステップにおける銘柄間の平均（クロスセクショナル平均）がゼロであることを確認
    np.testing.assert_allclose(np.mean(Lambda, axis=0), 0.0, atol=1e-12)


def test_input_dimension_exception():
    """
    テスト3: 入力次元・例外検証
    M < d の場合に適切なエラーが送出されること。
    """
    np.random.seed(42)
    M, N = 5, 10  # M = 5
    d = 10        # d = 10 > M
    R = np.random.randn(M, N)

    combiner = AlphaCombiner()
    with pytest.raises(ValueError):
        combiner.compute_pipeline(R, d=d)


def test_constant_signal_fault_tolerance():
    """
    テスト4: 定数シグナルの耐障害性
    ボラティリティがゼロのシグナルが含まれていた場合でも、ゼロ除算エラーにならず安全に処理されること。
    """
    M, N = 20, 4
    np.random.seed(42)
    R = np.random.randn(M, N)
    # 0番目のシグナルを定数（ボラティリティゼロ）にする
    R[:, 0] = 0.05

    combiner = AlphaCombiner()
    # ゼロ除算エラーや例外が発生せず正常にパイプラインが完了すること
    res = combiner.compute_pipeline(R)

    weights = res["weights"]
    assert not np.isnan(weights).any()
    assert not np.isinf(weights).any()
    # L1ノルム制約も満たされること
    assert pytest.approx(np.sum(np.abs(weights)), abs=1e-6) == 1.0


def test_drift_removal_and_volatility():
    np.random.seed(42)
    M, N = 20, 4
    # Generate random returns
    R = np.random.randn(M, N) + 0.05
    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(R)

    X = res["drift_removed"]
    sigma = res["volatility"]
    Y = res["standardized"]

    # Check drift removal: mean along time dimension should be close to 0
    np.testing.assert_allclose(np.mean(X, axis=1), 0.0, atol=1e-12)

    # Check volatility estimation: sqrt(mean(X^2))
    expected_sigma = np.sqrt(np.mean(X ** 2, axis=1))
    np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-10)

    # Check standardization: Y = X / sigma
    np.testing.assert_allclose(Y, X / sigma[:, np.newaxis], rtol=1e-10)


def test_cross_sectional_mean_removal():
    np.random.seed(42)
    M, N = 15, 5
    R = np.random.randn(M, N)
    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(R)

    Lambda = res["factor_matrix"]
    # Check shape: N x (M-1)
    assert Lambda.shape == (N, M - 1)

    # Cross-sectional mean (along axis 0) should be zero for each time step
    np.testing.assert_allclose(np.mean(Lambda, axis=0), 0.0, atol=1e-12)


def test_expected_returns_and_normalization():
    np.random.seed(42)
    M, N = 30, 3
    R = np.random.randn(M, N)
    combiner = AlphaCombiner()

    d = 10
    res = combiner.compute_pipeline(R, d=d)

    E = res["expected_returns"]
    E_norm = res["norm_expected_returns"]
    sigma = res["volatility"]

    # Check E calculation: mean of last d periods
    expected_E = np.mean(R.T[:, -d:], axis=1)
    np.testing.assert_allclose(E, expected_E, rtol=1e-10)

    # Check E_norm calculation: E / sigma
    np.testing.assert_allclose(E_norm, E / sigma, rtol=1e-10)


def test_weights_sum_constraint():
    np.random.seed(123)
    M, N = 50, 10
    R = np.random.randn(M, N)
    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(R)

    w = res["weights"]
    # Absolute sum constraint sum(|w|) = 1
    np.testing.assert_allclose(np.sum(np.abs(w)), 1.0, rtol=1e-10)


def test_dataframe_input_and_series_weights():
    M, N = 25, 3
    cols = ["Signal_A", "Signal_B", "Signal_C"]
    dates = pd.date_range("2023-01-01", periods=M)
    df_R = pd.DataFrame(np.random.randn(M, N), index=dates, columns=cols)

    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(df_R)

    weights = res["weights"]
    assert isinstance(weights, pd.Series)
    assert list(weights.index) == cols
    np.testing.assert_allclose(np.sum(np.abs(weights.values)), 1.0, rtol=1e-10)


def test_custom_latest_signals():
    M, N = 20, 4
    R = np.random.randn(M, N)
    latest_S = np.array([1.5, -0.5, 2.0, 0.0])

    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(R, latest_signals=latest_S)

    w = res["weights"]
    expected_combined = float(np.sum(w * latest_S))
    assert pytest.approx(res["combined_signal"], rel=1e-10) == expected_combined


def test_volatility_clipping():
    # Matrix with constant row (zero variance)
    R = np.array([
        [0.05, 0.05, 0.05, 0.05],
        [0.01, 0.02, -0.01, 0.03],
        [0.02, -0.01, 0.04, 0.01]
    ]).T  # shape M=4, N=3
    min_vol = 1e-8
    combiner = AlphaCombiner(min_vol=min_vol)
    res = combiner.compute_pipeline(R)

    sigma = res["volatility"]
    # First signal has 0 variance, should be clipped to min_vol
    assert sigma[0] == pytest.approx(min_vol)


def test_orthogonalizer_methods():
    np.random.seed(99)
    M, N = 10, 5
    R = np.random.randn(M, N)

    for method in ["auto", "ols", "svd", "ridge"]:
        combiner = AlphaCombiner(method=method)
        res = combiner.compute_pipeline(R)
        w = res["weights"]
        np.testing.assert_allclose(np.sum(np.abs(w)), 1.0, rtol=1e-10)


def test_m_less_than_n_input():
    np.random.seed(42)
    M, N = 10, 20
    # R shape is (M, N) = (10, 20) where time M < signals N
    R = np.random.randn(M, N)
    latest_S = np.random.randn(N)

    combiner = AlphaCombiner()
    res = combiner.compute_pipeline(R, latest_signals=latest_S)

    assert res["drift_removed"].shape == (N, M)
    assert len(res["weights"]) == N
    np.testing.assert_allclose(np.sum(np.abs(res["weights"])), 1.0, rtol=1e-10)


def test_invalid_inputs():
    R = np.random.randn(10, 4)
    combiner = AlphaCombiner()

    with pytest.raises(ValueError, match="Estimation window d must be between"):
        combiner.compute_pipeline(R, d=15)

    with pytest.raises(ValueError, match="latest_signals length"):
        combiner.compute_pipeline(R, latest_signals=np.array([1.0, 2.0]))
