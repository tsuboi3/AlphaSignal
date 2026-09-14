import pytest
import numpy as np
import pandas as pd
from src.alpha_engine import AlphaEngineConfig, AlphaEngine


def test_alpha_engine_config_defaults():
    config = AlphaEngineConfig()
    assert config.d == 5
    assert config.eps_clip == 1e-8
    assert config.l2_reg == 1e-4


def test_fit_weights_l1_norm():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=10)
    signals = ["sig1", "sig2", "sig3"]
    returns_data = np.random.randn(10, 3) * 0.02
    returns_matrix = pd.DataFrame(returns_data, index=dates, columns=signals)

    engine = AlphaEngine()
    weights = engine.fit_weights(returns_matrix)

    assert isinstance(weights, pd.Series)
    assert list(weights.index) == signals
    # L1ノルムが 1.0 になること
    l1_norm = weights.abs().sum()
    assert pytest.approx(l1_norm, abs=1e-6) == 1.0


def test_combine_series():
    signals = ["sig1", "sig2", "sig3"]
    weights = pd.Series([0.5, -0.3, 0.2], index=signals)
    current_signals = pd.Series([10.0, 5.0, -2.0], index=signals)

    engine = AlphaEngine()
    combined_val = engine.combine(current_signals, weights)

    # 10.0 * 0.5 + 5.0 * (-0.3) + (-2.0) * 0.2 = 5.0 - 1.5 - 0.4 = 3.1
    assert isinstance(combined_val, float)
    assert pytest.approx(combined_val, abs=1e-6) == 3.1


def test_combine_dataframe():
    signals = ["sig1", "sig2", "sig3"]
    weights = pd.Series([0.5, -0.3, 0.2], index=signals)

    dates = pd.date_range("2023-01-01", periods=3)
    current_signals = pd.DataFrame(
        [
            [10.0, 5.0, -2.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        index=dates,
        columns=signals,
    )

    engine = AlphaEngine()
    combined_series = engine.combine(current_signals, weights)

    assert isinstance(combined_series, pd.Series)
    assert list(combined_series.index) == list(dates)
    assert pytest.approx(combined_series.iloc[0], abs=1e-6) == 3.1
    # 1.0*0.5 + 2.0*(-0.3) + 3.0*0.2 = 0.5 - 0.6 + 0.6 = 0.5
    assert pytest.approx(combined_series.iloc[1], abs=1e-6) == 0.5
    assert pytest.approx(combined_series.iloc[2], abs=1e-6) == 0.0


def test_empty_returns_matrix():
    engine = AlphaEngine()
    with pytest.raises(ValueError):
        engine.fit_weights(pd.DataFrame())


def test_row_count_less_than_or_equal_to_d():
    engine = AlphaEngine(config=AlphaEngineConfig(d=5))
    dates = pd.date_range("2023-01-01", periods=5)
    df = pd.DataFrame(np.random.randn(5, 2), index=dates, columns=["sig1", "sig2"])
    with pytest.raises(ValueError, match="Input data row count M"):
        engine.fit_weights(df)


def test_nan_inf_imputation():
    engine = AlphaEngine(config=AlphaEngineConfig(d=3))
    dates = pd.date_range("2023-01-01", periods=6)
    # Put NaN and Inf values
    data = [
        [np.nan, 0.01],
        [0.02, np.inf],
        [np.nan, -np.inf],
        [0.03, 0.04],
        [0.01, np.nan],
        [0.02, 0.05],
    ]
    df = pd.DataFrame(data, index=dates, columns=["sig1", "sig2"])
    weights = engine.fit_weights(df)
    assert isinstance(weights, pd.Series)
    assert not weights.isna().any()
    assert pytest.approx(weights.abs().sum(), abs=1e-6) == 1.0


def test_ill_conditioned_cov_fallback():
    engine = AlphaEngine(config=AlphaEngineConfig(d=5))
    dates = pd.date_range("2023-01-01", periods=10)
    # Create collinear columns (singular/ill-conditioned matrix)
    base = np.random.randn(10, 1)
    data = np.hstack([base, base, base * 2])
    df = pd.DataFrame(data, index=dates, columns=["sig1", "sig2", "sig3"])

    weights = engine.fit_weights(df)
    assert isinstance(weights, pd.Series)
    assert not weights.isna().any()
    assert pytest.approx(weights.abs().sum(), abs=1e-6) == 1.0


def test_failsafe_zero_raw_weights():
    engine = AlphaEngine(config=AlphaEngineConfig(d=5))
    dates = pd.date_range("2023-01-01", periods=10)
    # Zero returns matrix -> mu is zeros -> raw weights zeros -> sum |w_raw| < 1e-12
    df = pd.DataFrame(np.zeros((10, 3)), index=dates, columns=["sig1", "sig2", "sig3"])

    weights = engine.fit_weights(df)
    assert isinstance(weights, pd.Series)
    # Equal weights 1/N = 1/3
    np.testing.assert_allclose(weights.values, np.array([1 / 3, 1 / 3, 1 / 3]))
