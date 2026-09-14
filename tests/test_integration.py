import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alpha_engine import AlphaEngine, AlphaEngineConfig
from src.signals.alpha_combiner import AlphaCombiner
from src.menu import resolve_ticker, validate_date
from src.database import db_manager


class TestIntegration(unittest.TestCase):
    def test_ticker_resolution(self):
        self.assertEqual(resolve_ticker("apple"), "AAPL")
        self.assertEqual(resolve_ticker("トヨタ"), "7203.T")
        self.assertEqual(resolve_ticker("NVDA"), "NVDA")

    def test_date_validation(self):
        self.assertTrue(validate_date("2024-01-01"))
        self.assertFalse(validate_date("2024-13-01"))
        self.assertFalse(validate_date("invalid-date"))

    def test_alpha_engine_pipeline(self):
        dates = pd.date_range("2023-01-01", periods=30)
        data = {
            "sig1": np.random.randn(30),
            "sig2": np.random.randn(30),
            "sig3": np.random.randn(30)
        }
        df = pd.DataFrame(data, index=dates)

        engine = AlphaEngine(AlphaEngineConfig(d=5))
        weights = engine.fit_weights(df)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights.abs().sum(), 1.0, places=4)

        combined = engine.combine(df.iloc[-1], weights)
        self.assertIsInstance(combined, float)

    def test_alpha_combiner_pipeline(self):
        dates = pd.date_range("2023-01-01", periods=30)
        data = {
            "ret1": np.random.randn(30),
            "ret2": np.random.randn(30),
            "ret3": np.random.randn(30)
        }
        df = pd.DataFrame(data, index=dates)

        combiner = AlphaCombiner(method='auto')
        res = combiner.compute_pipeline(df, d=5)

        self.assertIn("weights", res)
        self.assertIn("combined_signal", res)
        self.assertEqual(len(res["weights"]), 3)
        self.assertIsInstance(res["combined_signal"], float)

    def test_db_manager_connection(self):
        conn = db_manager._get_connection()
        self.assertIsNotNone(conn)
        conn.close()


if __name__ == '__main__':
    unittest.main()
