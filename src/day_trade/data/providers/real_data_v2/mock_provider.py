#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Mock Provider
リアルデータプロバイダー V2 - モックプロバイダー

テスト用の模擬データ生成機能
"""

import asyncio
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from .base_provider import BaseDataProvider
from .models import DataSource, DataSourceConfig, DataFetchResult, DataQualityLevel


class MockDataProvider(BaseDataProvider):
    """モックデータプロバイダー（テスト用）"""

    def __init__(self, config: DataSourceConfig):
        """モックプロバイダー初期化
        
        Args:
            config: データソース設定
        """
        super().__init__(config)
        self.generated_symbols = set()

    async def get_stock_data(self, 
                           symbol: str, 
                           period: str = "1mo") -> DataFetchResult:
        """模擬データ生成
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            
        Returns:
            模擬データ結果
        """
        start_time = time.time()

        try:
            # 模擬遅延（実際のAPIを模倣）
            await asyncio.sleep(0.1)

            # データ期間決定
            days = self._period_to_days(period)

            # 模擬株価データ生成
            data = self._generate_mock_data(symbol, days)

            # 品質スコア計算
            quality_level, quality_score = self._calculate_quality_score(data)
            fetch_time = time.time() - start_time

            self._record_request()
            self.generated_symbols.add(symbol)

            self.logger.debug(
                f"Generated mock data for {symbol} "
                f"(quality: {quality_score:.1f})"
            )

            return DataFetchResult(
                data=data,
                source=DataSource.MOCK,
                quality_level=quality_level,
                quality_score=quality_score,
                fetch_time=fetch_time,
                metadata={
                    'generated': True,
                    'symbol': symbol,
                    'period': period,
                    'data_points': len(data)
                }
            )

        except Exception as e:
            fetch_time = time.time() - start_time
            error_msg = f"Mock data generation error: {e}"
            self.logger.error(error_msg)
            
            return DataFetchResult(
                data=None,
                source=DataSource.MOCK,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=fetch_time,
                error_message=error_msg
            )

    def _generate_mock_data(self, symbol: str, days: int) -> pd.DataFrame:
        """模擬株価データ生成
        
        Args:
            symbol: 銘柄コード（乱数シード用）
            days: 生成する日数
            
        Returns:
            模擬株価データ
        """
        # 銘柄コードから再現可能な乱数シードを生成
        np.random.seed(hash(symbol) % 2**32)
        
        # 基準価格設定（銘柄コードに基づく）
        base_price = self._calculate_base_price(symbol)
        
        # 日付範囲生成
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 価格変動生成
        prices = self._generate_price_series(base_price, days)
        
        # OHLCV データ作成
        ohlcv_data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            open_price = prices[i-1] if i > 0 else close
            high, low = self._generate_high_low(open_price, close)
            volume = self._generate_volume(symbol)
            
            ohlcv_data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })

        # DataFrameとして返却
        df = pd.DataFrame(ohlcv_data, index=dates)
        
        # データ品質の調整（設定可能な品質レベル）
        df = self._adjust_data_quality(df, symbol)
        
        return df

    def _calculate_base_price(self, symbol: str) -> float:
        """基準価格計算
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            基準価格
        """
        # 銘柄コードから価格レンジを決定
        hash_value = hash(symbol) % 10000
        
        if symbol.isdigit():
            # 日本株風（100-10000円）
            base_price = 100 + (hash_value % 9900)
        else:
            # 海外株風（1-1000ドル）
            base_price = 1 + (hash_value % 999)
            
        return float(base_price)

    def _generate_price_series(self, base_price: float, days: int) -> list:
        """価格時系列生成
        
        Args:
            base_price: 基準価格
            days: 日数
            
        Returns:
            価格のリスト
        """
        prices = []
        current_price = base_price
        
        # トレンド設定（上昇・下降・横ばい）
        trend = np.random.choice(['up', 'down', 'sideways'], p=[0.3, 0.3, 0.4])
        trend_strength = np.random.uniform(0.0001, 0.005)  # 日次トレンド
        
        for day in range(days):
            # トレンド成分
            if trend == 'up':
                trend_change = trend_strength
            elif trend == 'down':
                trend_change = -trend_strength
            else:
                trend_change = np.random.uniform(-trend_strength, trend_strength)
            
            # ランダム成分
            random_change = np.random.normal(0, 0.02)  # 2%の標準偏差
            
            # 価格更新
            total_change = trend_change + random_change
            current_price *= (1 + total_change)
            current_price = max(current_price, base_price * 0.1)  # 最低価格制限
            
            prices.append(current_price)

        return prices

    def _generate_high_low(self, open_price: float, close_price: float) -> tuple:
        """高値・安値生成
        
        Args:
            open_price: 始値
            close_price: 終値
            
        Returns:
            (高値, 安値)のタプル
        """
        # 高値は始値・終値の高い方を基準に上乗せ
        base_high = max(open_price, close_price)
        high = base_high * np.random.uniform(1.0, 1.03)  # 最大3%上乗せ
        
        # 安値は始値・終値の低い方を基準に下げ
        base_low = min(open_price, close_price)
        low = base_low * np.random.uniform(0.97, 1.0)  # 最大3%下げ
        
        return high, low

    def _generate_volume(self, symbol: str) -> int:
        """出来高生成
        
        Args:
            symbol: 銘柄コード（乱数調整用）
            
        Returns:
            出来高
        """
        # 銘柄ごとの基準出来高
        hash_value = hash(symbol + "volume") % 10000000
        base_volume = 100000 + hash_value
        
        # 日次変動（0.5倍から2倍の範囲）
        variation = np.random.uniform(0.5, 2.0)
        
        return int(base_volume * variation)

    def _adjust_data_quality(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """データ品質調整
        
        Args:
            df: 調整対象データフレーム
            symbol: 銘柄コード
            
        Returns:
            品質調整済みデータフレーム
        """
        # 品質レベルを銘柄コードに基づいて決定
        quality_seed = hash(symbol + "quality") % 100
        
        if quality_seed < 10:
            # 10%の確率で品質の悪いデータ
            df = self._introduce_quality_issues(df)
        elif quality_seed < 30:
            # 20%の確率で軽微な品質問題
            df = self._introduce_minor_issues(df)
        
        return df

    def _introduce_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """品質問題の導入
        
        Args:
            df: 対象データフレーム
            
        Returns:
            品質問題を含むデータフレーム
        """
        df_copy = df.copy()
        
        # ランダムにNaN値を導入
        if len(df_copy) > 5:
            nan_indices = np.random.choice(
                len(df_copy), 
                size=min(3, len(df_copy)//10), 
                replace=False
            )
            nan_columns = np.random.choice(
                ['Open', 'High', 'Low', 'Close'], 
                size=2, 
                replace=False
            )
            for idx in nan_indices:
                for col in nan_columns:
                    df_copy.iloc[idx, df_copy.columns.get_loc(col)] = np.nan
        
        return df_copy

    def _introduce_minor_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """軽微な品質問題の導入
        
        Args:
            df: 対象データフレーム
            
        Returns:
            軽微な問題を含むデータフレーム
        """
        df_copy = df.copy()
        
        # 価格整合性の軽微な問題（1-2箇所）
        if len(df_copy) > 2:
            problem_index = np.random.randint(0, len(df_copy))
            # 高値を低値より少し低くする
            if 'High' in df_copy.columns and 'Low' in df_copy.columns:
                low_val = df_copy.iloc[problem_index]['Low']
                df_copy.iloc[problem_index, df_copy.columns.get_loc('High')] = low_val * 0.99
        
        return df_copy

    def get_generated_symbols(self) -> list:
        """生成済み銘柄一覧取得
        
        Returns:
            生成済み銘柄コードのリスト
        """
        return list(self.generated_symbols)

    def clear_generated_history(self):
        """生成履歴クリア"""
        self.generated_symbols.clear()
        self.logger.info("Cleared mock data generation history")

    def set_quality_mode(self, mode: str):
        """品質モード設定
        
        Args:
            mode: 品質モード ('high', 'medium', 'low', 'random')
        """
        if hasattr(self, 'quality_mode'):
            self.quality_mode = mode
        else:
            self.quality_mode = mode
        
        self.logger.info(f"Set mock data quality mode to: {mode}")