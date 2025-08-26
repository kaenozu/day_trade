#!/usr/bin/env python3
"""
最適化ユーティリティ

パフォーマンステストやデモンストレーション用のユーティリティ関数を提供します。
"""

import numpy as np
import pandas as pd


def create_sample_data(rows: int = 10000) -> pd.DataFrame:
    """テスト用のサンプルデータ作成"""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "open": np.random.randn(rows).cumsum() + 100,
            "high": np.random.randn(rows).cumsum() + 105,
            "low": np.random.randn(rows).cumsum() + 95,
            "close": np.random.randn(rows).cumsum() + 100,
            "volume": np.random.randint(1000000, 10000000, rows),
        }
    )
