#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stooq Data Provider

Stooq データプロバイダー
"""

import time
from io import StringIO
import aiohttp
import pandas as pd

from .base_provider import BaseDataProvider
from .enums import DataSourceConfig, DataFetchResult, DataSource, DataQualityLevel


class ImprovedStooqProvider(BaseDataProvider):
    """改善版Stooq プロバイダー"""

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
        """Stooqから株価データ取得（改良版）
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            
        Returns:
            データ取得結果
        """
        start_time = time.time()
        error_msg = None

        try:
            await self._wait_for_rate_limit()

            # Stooq用の銘柄コード変換
            stooq_symbol = self._convert_to_stooq_symbol(symbol)
            days = self._period_to_days(period)

            # URL構築
            url = f"{self.config.base_url}?s={stooq_symbol}&d1={days}&i=d"

            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self.config.headers) as response:
                    self._record_request()

                    if response.status == 200:
                        text = await response.text()

                        # CSVデータをDataFrameに変換
                        df = pd.read_csv(StringIO(text))

                        if not df.empty and 'Close' in df.columns:
                            # 標準フォーマットに変換
                            df = self._convert_to_standard_format(df)

                            # データ品質チェック
                            quality_level, quality_score = (
                                self._calculate_quality_score(df)
                            )
                            fetch_time = time.time() - start_time

                            self.logger.info(
                                f"Stooq fetch successful for {symbol} "
                                f"(quality: {quality_score:.1f})"
                            )

                            return DataFetchResult(
                                data=df,
                                source=DataSource.STOOQ,
                                quality_level=quality_level,
                                quality_score=quality_score,
                                fetch_time=fetch_time,
                                metadata={'stooq_symbol': stooq_symbol}
                            )
                    else:
                        error_msg = f"HTTP {response.status}"
                        self.logger.warning(
                            f"Stooq HTTP error for {symbol}: {error_msg}"
                        )

        except Exception as e:
            error_msg = f"Stooq fetch error: {e}"
            self.logger.error(f"Stooq error for {symbol}: {error_msg}")

        fetch_time = time.time() - start_time
        return DataFetchResult(
            data=None,
            source=DataSource.STOOQ,
            quality_level=DataQualityLevel.FAILED,
            quality_score=0.0,
            fetch_time=fetch_time,
            error_message=error_msg or "Unknown error"
        )

    def _convert_to_stooq_symbol(self, symbol: str) -> str:
        """Stooq用銘柄コード変換
        
        Args:
            symbol: 元の銘柄コード
            
        Returns:
            Stooq形式の銘柄コード
        """
        if symbol.isdigit():
            return f"{symbol}.jp"
        elif symbol.endswith('.T'):
            return symbol.replace('.T', '.jp')
        elif symbol.endswith('.JP'):
            return symbol.lower()
        return symbol

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

    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準フォーマットに変換
        
        Args:
            df: 元のデータフレーム
            
        Returns:
            標準フォーマットのデータフレーム
        """
        # 列名マッピング
        column_map = {
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }

        # 列名変換
        df = df.rename(columns=column_map)

        # 日付処理
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

        # 数値型変換
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # NaN値を削除
        df = df.dropna()

        return df