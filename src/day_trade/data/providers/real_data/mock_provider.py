#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock Data Provider

モックデータプロバイダー（テスト用）
"""

import asyncio
import time
from datetime import datetime
import numpy as np
import pandas as pd

from .base_provider import BaseDataProvider
from .enums import DataSourceConfig, DataFetchResult, DataSource, DataQualityLevel


class MockDataProvider(BaseDataProvider):
    """モックデータプロバイダー（テスト用）"""

    def __init__(self, config: DataSourceConfig):
        """初期化
        
        Args:
            config: データソース設定
        """
        super().__init__(config)

    async def get_stock_data(
        self,
        symbol: str,
        period: str = "1mo"
    ) -> DataFetchResult:
        """模擬データ生成
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            
        Returns:
            データ取得結果
        """
        start_time = time.time()

        try:
            await asyncio.sleep(0.1)  # 模擬遅延

            # データ期間決定
            days = self._period_to_days(period)

            # 模擬株価データ生成
            np.random.seed(hash(symbol) % 2**32)
            base_price = 1000 + (hash(symbol) % 10000)

            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            prices = []
            current_price = base_price

            for _ in range(days):
                change = np.random.normal(0, 0.02)  # 2%の標準偏差
                current_price *= (1 + change)
                current_price = max(current_price, 100)  # 最低価格
                prices.append(current_price)

            # OHLCV データ作成
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                open_price = prices[i-1] if i > 0 else close
                high = max(open_price, close) * np.random.uniform(1.0, 1.02)
                low = min(open_price, close) * np.random.uniform(0.98, 1.0)
                volume = np.random.randint(100000, 10000000)

                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': volume
                })

            df = pd.DataFrame(data, index=dates)

            # 品質スコア計算
            quality_level, quality_score = self._calculate_quality_score(df)
            fetch_time = time.time() - start_time

            self._record_request()

            return DataFetchResult(
                data=df,
                source=DataSource.MOCK,
                quality_level=quality_level,
                quality_score=quality_score,
                fetch_time=fetch_time,
                metadata={'generated': True, 'base_price': base_price}
            )

        except Exception as e:
            fetch_time = time.time() - start_time
            return DataFetchResult(
                data=None,
                source=DataSource.MOCK,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=fetch_time,
                error_message=f"Mock data generation error: {e}"
            )

    def _period_to_days(self, period: str) -> int:
        """期間文字列を日数に変換
        
        Args:
            period: 期間文字列
            
        Returns:
            日数
        """
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        return period_map.get(period, 30)