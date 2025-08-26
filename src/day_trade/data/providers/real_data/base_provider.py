#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Data Provider

データプロバイダー基底クラス
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

from .enums import DataSourceConfig, DataFetchResult, DataQualityLevel


class BaseDataProvider(ABC):
    """データプロバイダー基底クラス"""

    def __init__(self, config: DataSourceConfig):
        """初期化
        
        Args:
            config: データソース設定
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.daily_request_count = 0
        self.last_request_time = 0
        self.request_history = []

    @abstractmethod
    async def get_stock_data(
        self,
        symbol: str,
        period: str = "1mo"
    ) -> DataFetchResult:
        """株価データ取得（抽象メソッド）
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            
        Returns:
            データ取得結果
        """
        pass

    async def _wait_for_rate_limit(self):
        """レート制限対応"""
        current_time = time.time()

        # 1分間のリクエスト数制限
        minute_ago = current_time - 60
        recent_requests = [t for t in self.request_history if t > minute_ago]

        if len(recent_requests) >= self.config.rate_limit_per_minute:
            wait_time = 60 - (current_time - recent_requests[0])
            if wait_time > 0:
                self.logger.debug(f"Rate limit wait: {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        # 最小間隔制限
        if self.last_request_time > 0:
            min_interval = 60 / self.config.rate_limit_per_minute
            elapsed = current_time - self.last_request_time
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                await asyncio.sleep(wait_time)

    def _record_request(self):
        """リクエスト記録"""
        current_time = time.time()
        self.request_history.append(current_time)
        self.last_request_time = current_time
        self.request_count += 1

        # 1日前より古い記録は削除
        day_ago = current_time - 86400
        self.request_history = [
            t for t in self.request_history if t > day_ago
        ]
        self.daily_request_count = len(self.request_history)

    def _calculate_quality_score(
        self,
        data: pd.DataFrame
    ) -> Tuple[DataQualityLevel, float]:
        """データ品質スコア計算
        
        Args:
            data: データフレーム
            
        Returns:
            品質レベルとスコアのタプル
        """
        if data is None or data.empty:
            return DataQualityLevel.FAILED, 0.0

        score = 0.0

        # データ量チェック (20点)
        if len(data) >= self.config.min_data_points:
            score += 20.0
        else:
            score += 20.0 * (len(data) / self.config.min_data_points)

        # 必要列の存在チェック (20点)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        existing_columns = sum(
            1 for col in required_columns if col in data.columns
        )
        score += 20.0 * (existing_columns / len(required_columns))

        # データ完全性チェック (20点)
        total_values = len(data) * len(data.columns)
        if total_values > 0:
            completeness = 1.0 - (data.isnull().sum().sum() / total_values)
            score += 20.0 * completeness
        else:
            score += 0.0

        # 価格整合性チェック (20点)
        if (self.config.price_consistency_check and
            all(col in data.columns
                for col in ['Open', 'High', 'Low', 'Close'])):
            consistency_violations = (
                (data['High'] < data['Low']) |
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close'])
            ).sum()

            consistency_score = 1.0 - (consistency_violations / len(data))
            score += 20.0 * consistency_score
        else:
            score += 20.0  # 整合性チェック無効の場合は満点

        # 異常値チェック (20点)
        anomaly_score = 20.0
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                if (data[col] <= 0).any():
                    anomaly_score -= 5.0
                if (data[col] > self.config.max_price_threshold).any():
                    anomaly_score -= 5.0

        score += max(0.0, anomaly_score)

        # 品質レベル判定
        if score >= 90:
            level = DataQualityLevel.HIGH
        elif score >= 70:
            level = DataQualityLevel.MEDIUM
        elif score >= 50:
            level = DataQualityLevel.LOW
        else:
            level = DataQualityLevel.FAILED

        return level, score