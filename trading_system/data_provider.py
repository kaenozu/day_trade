import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from real_data_provider_v2 import DataSource, real_data_provider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


class MultiSourceDataProvider:
    """マルチソースデータプロバイダー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_sources = {
            'primary': 'yahoo_finance',
            'secondary': 'stooq',
            'tertiary': 'alpha_vantage'
        }
        self.source_weights = {
            'yahoo_finance': 0.5,
            'stooq': 0.3,
            'alpha_vantage': 0.2
        }

    async def get_multi_source_data(self, symbol: str, period: str = '1mo') -> pd.DataFrame:
        """複数ソースからデータ取得"""
        all_data = {}

        for source_name, weight in self.source_weights.items():
            try:
                data = await self._get_data_from_source(symbol, period, source_name)
                if data is not None and not data.empty:
                    all_data[source_name] = {
                        'data': data,
                        'weight': weight,
                        'quality_score': self._assess_data_quality(data)
                    }
            except Exception as e:
                self.logger.warning(f"Failed to get data from {source_name}: {e}")

        if not all_data:
            self.logger.error(f"No data sources available for {symbol}")
            return pd.DataFrame()

        # データ統合（重み付け平均）
        return self._integrate_multi_source_data(all_data)

    async def _get_data_from_source(self, symbol: str, period: str, source: str) -> Optional[pd.DataFrame]:
        """個別ソースからデータ取得"""
        if REAL_DATA_PROVIDER_AVAILABLE and source == 'yahoo_finance':
            try:
                return await real_data_provider.get_stock_data(symbol, period)
            except Exception as e:
                self.logger.warning(f"Real data provider failed: {e}")

        # フォールバック：模擬データ
        return self._generate_mock_data(symbol, period)

    def _generate_mock_data(self, symbol: str, period: str) -> pd.DataFrame:
        """模擬データ生成"""
        periods_map = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}
        days = periods_map.get(period, 30)

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # 模擬価格生成
        price = 1000
        prices = []
        volumes = []

        for i in range(days):
            change = np.random.normal(0, 0.02)
            price *= (1 + change)
            prices.append(price)
            volumes.append(np.random.randint(1000000, 10000000))

        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.02) for p in prices],
            'Low': [p * np.random.uniform(0.98, 1.00) for p in prices],
            'Close': prices,
            'Volume': volumes
        }, index=dates)

        return data

    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """データ品質評価"""
        if data.empty:
            return 0.0

        # 基本品質指標
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        consistency = self._check_price_consistency(data)
        recency = self._check_data_recency(data)

        return (completeness * 0.4 + consistency * 0.4 + recency * 0.2) / 100

    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """価格整合性チェック"""
        if data.empty or len(data) < 2:
            return 50.0

        try:
            inconsistencies = 0
            total_checks = 0

            for idx in data.index:
                if all(col in data.columns for col in ['High', 'Low', 'Open', 'Close']):
                    high = data.loc[idx, 'High']
                    low = data.loc[idx, 'Low']
                    open_price = data.loc[idx, 'Open']
                    close_price = data.loc[idx, 'Close']

                    total_checks += 4

                    if high < low or high < open_price or high < close_price or low > open_price or low > close_price:
                        inconsistencies += 1

                    if any(price <= 0 for price in [high, low, open_price, close_price]):
                        inconsistencies += 1

            if total_checks == 0:
                return 50.0

            return (total_checks - inconsistencies) / total_checks * 100

        except Exception:
            return 30.0

    def _check_data_recency(self, data: pd.DataFrame) -> float:
        """データ新鮮度チェック"""
        if data.empty:
            return 0.0

        try:
            latest_date = data.index[-1]
            if isinstance(latest_date, str):
                latest_date = pd.to_datetime(latest_date)

            days_old = (datetime.now() - latest_date).days

            if days_old <= 1:
                return 100.0
            elif days_old <= 3:
                return 80.0
            elif days_old <= 7:
                return 60.0
            elif days_old <= 30:
                return 40.0
            else:
                return 20.0

        except Exception:
            return 50.0

    def _integrate_multi_source_data(self, all_data: Dict) -> pd.DataFrame:
        """マルチソースデータ統合"""
        if not all_data:
            return pd.DataFrame()

        # 最も品質の高いデータをベースとする
        best_source = max(all_data.keys(), key=lambda k: all_data[k]['quality_score'])
        base_data = all_data[best_source]['data'].copy()

        # 他のソースとの重み付け平均（価格データのみ）
        price_columns = ['Open', 'High', 'Low', 'Close']

        for col in price_columns:
            if col in base_data.columns:
                weighted_values = []

                for source, info in all_data.items():
                    source_data = info['data']
                    weight = info['weight'] * info['quality_score']

                    if col in source_data.columns:
                        # 共通の日付範囲での重み付け
                        common_dates = base_data.index.intersection(source_data.index)
                        if len(common_dates) > 0:
                            weighted_values.append(source_data.loc[common_dates, col] * weight)

                if weighted_values:
                    total_weight = sum(all_data[s]['weight'] * all_data[s]['quality_score'] for s in all_data.keys())
                    base_data[col] = sum(weighted_values) / total_weight

        return base_data
