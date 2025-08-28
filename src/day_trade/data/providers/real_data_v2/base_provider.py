#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Base Provider
リアルデータプロバイダー V2 - 基底プロバイダー

データプロバイダーの基底クラスと共通機能
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from .models import (
    DataSourceConfig, DataFetchResult, DataQualityLevel, 
    REQUIRED_COLUMNS, DEFAULT_PERIOD_DAYS
)


class BaseDataProvider(ABC):
    """データプロバイダー基底クラス"""

    def __init__(self, config: DataSourceConfig):
        """基底プロバイダー初期化
        
        Args:
            config: データソース設定
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # リクエスト管理
        self.request_count = 0
        self.daily_request_count = 0
        self.last_request_time = 0
        self.request_history = []

    @abstractmethod
    async def get_stock_data(self, 
                           symbol: str, 
                           period: str = "1mo") -> DataFetchResult:
        """株価データ取得（抽象メソッド）
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            
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

    def _calculate_quality_score(self, 
                                data: pd.DataFrame) -> Tuple[DataQualityLevel, float]:
        """データ品質スコア計算
        
        Args:
            data: 評価対象データフレーム
            
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
        existing_columns = sum(
            1 for col in REQUIRED_COLUMNS if col in data.columns
        )
        score += 20.0 * (existing_columns / len(REQUIRED_COLUMNS))

        # データ完全性チェック (20点)
        total_cells = len(data) * len(data.columns)
        if total_cells > 0:
            completeness = 1.0 - (data.isnull().sum().sum() / total_cells)
            score += 20.0 * completeness
        else:
            score += 20.0

        # 価格整合性チェック (20点)
        score += self._check_price_consistency(data)

        # 異常値チェック (20点)
        score += self._check_anomalies(data)

        # 品質レベル判定
        level = self._determine_quality_level(score)

        return level, score

    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """価格整合性チェック
        
        Args:
            data: チェック対象データフレーム
            
        Returns:
            整合性スコア（0-20）
        """
        if not self.config.price_consistency_check:
            return 20.0

        price_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in price_columns):
            return 20.0  # 価格列が不完全な場合は満点

        try:
            consistency_violations = (
                (data['High'] < data['Low']) |
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close'])
            ).sum()

            if len(data) > 0:
                consistency_score = 1.0 - (consistency_violations / len(data))
                return 20.0 * consistency_score
            else:
                return 20.0

        except Exception as e:
            self.logger.warning(f"Price consistency check failed: {e}")
            return 10.0  # エラーの場合は半分の点数

    def _check_anomalies(self, data: pd.DataFrame) -> float:
        """異常値チェック
        
        Args:
            data: チェック対象データフレーム
            
        Returns:
            異常値スコア（0-20）
        """
        anomaly_score = 20.0
        
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for col in price_columns:
            if col not in data.columns:
                continue
                
            try:
                # 負の値チェック
                if (data[col] <= 0).any():
                    anomaly_score -= 5.0
                    
                # 閾値超過チェック
                if (data[col] > self.config.max_price_threshold).any():
                    anomaly_score -= 5.0
                    
            except Exception as e:
                self.logger.warning(f"Anomaly check failed for {col}: {e}")
                anomaly_score -= 2.0

        return max(0.0, anomaly_score)

    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """品質レベル判定
        
        Args:
            score: 品質スコア
            
        Returns:
            品質レベル
        """
        if score >= 90:
            return DataQualityLevel.HIGH
        elif score >= 70:
            return DataQualityLevel.MEDIUM
        elif score >= 50:
            return DataQualityLevel.LOW
        else:
            return DataQualityLevel.FAILED

    def _period_to_days(self, period: str) -> int:
        """期間文字列を日数に変換
        
        Args:
            period: 期間文字列（例: '1mo', '3mo'）
            
        Returns:
            日数
        """
        return DEFAULT_PERIOD_DAYS.get(period, 30)

    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """データを標準フォーマットに変換
        
        Args:
            df: 変換対象データフレーム
            
        Returns:
            標準フォーマットのデータフレーム
        """
        try:
            # 日付処理
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            elif df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass

            # 数値型変換
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # NaN値を削除
            df = df.dropna()

            # 日付でソート
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()

            return df

        except Exception as e:
            self.logger.warning(f"Format conversion failed: {e}")
            return df

    def get_request_stats(self) -> dict:
        """リクエスト統計取得
        
        Returns:
            リクエスト統計情報
        """
        return {
            'total_requests': self.request_count,
            'daily_requests': self.daily_request_count,
            'daily_limit': self.config.rate_limit_per_day,
            'remaining_requests': max(
                0, self.config.rate_limit_per_day - self.daily_request_count
            ),
            'rate_limit_per_minute': self.config.rate_limit_per_minute,
            'last_request_time': self.last_request_time
        }

    def reset_request_history(self):
        """リクエスト履歴リセット"""
        self.request_history.clear()
        self.daily_request_count = 0
        self.logger.info(f"Reset request history for {self.config.name}")