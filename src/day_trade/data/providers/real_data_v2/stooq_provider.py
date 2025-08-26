#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Stooq Provider
リアルデータプロバイダー V2 - Stooq プロバイダー

Stooq からのデータ取得機能
"""

import time
import logging
from io import StringIO

import aiohttp
import pandas as pd

from .base_provider import BaseDataProvider
from .models import (
    DataSource, DataSourceConfig, DataFetchResult, DataQualityLevel,
    STOOQ_SUFFIXES
)


class ImprovedStooqProvider(BaseDataProvider):
    """改善版Stooq プロバイダー"""

    def __init__(self, config: DataSourceConfig):
        """Stooq プロバイダー初期化
        
        Args:
            config: データソース設定
        """
        super().__init__(config)

    async def get_stock_data(self, 
                           symbol: str, 
                           period: str = "1mo") -> DataFetchResult:
        """Stooqから株価データ取得
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            
        Returns:
            データ取得結果
        """
        start_time = time.time()
        error_msg = None

        try:
            # レート制限待機
            await self._wait_for_rate_limit()

            # Stooq用の銘柄コード変換
            stooq_symbol = self._convert_to_stooq_symbol(symbol)
            days = self._period_to_days(period)

            # URL構築
            url = f"{self.config.base_url}?s={stooq_symbol}&d1={days}&i=d"

            # HTTPクライアント設定
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            headers = self.config.headers or {}

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    self._record_request()

                    if response.status == 200:
                        text = await response.text()

                        # CSVデータをDataFrameに変換
                        df = self._parse_csv_data(text)

                        if not df.empty and 'Close' in df.columns:
                            # 標準フォーマットに変換
                            df = self._convert_to_standard_format(df)

                            # データ品質チェック
                            quality_level, quality_score = self._calculate_quality_score(df)
                            fetch_time = time.time() - start_time

                            if quality_score >= self.config.quality_threshold:
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
                                    metadata={
                                        'stooq_symbol': stooq_symbol,
                                        'url': url
                                    }
                                )
                            else:
                                error_msg = f"Data quality too low: {quality_score:.1f}"
                        else:
                            error_msg = "Empty data or missing Close column"
                    else:
                        error_msg = f"HTTP {response.status}: {response.reason}"
                        self.logger.warning(
                            f"Stooq HTTP error for {symbol}: {error_msg}"
                        )

        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error: {e}"
            self.logger.error(f"Stooq client error for {symbol}: {error_msg}")
        except Exception as e:
            error_msg = f"Stooq fetch error: {e}"
            self.logger.error(f"Stooq error for {symbol}: {error_msg}")

        # エラー時の結果返却
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
            Stooq用銘柄コード
        """
        if symbol.isdigit():
            # 数字のみの場合は.jpを追加
            return f"{symbol}.jp"
        elif symbol.endswith('.T'):
            # .T を .jp に変換
            return symbol.replace('.T', '.jp')
        elif symbol.endswith('.JP'):
            # .JP を小文字に変換
            return symbol.lower()
        elif '.' not in symbol:
            # サフィックスがない場合は.jpを追加
            return f"{symbol}.jp"
        else:
            # その他の場合はそのまま
            return symbol

    def _parse_csv_data(self, csv_text: str) -> pd.DataFrame:
        """CSVテキストをDataFrameに変換
        
        Args:
            csv_text: CSV形式のテキストデータ
            
        Returns:
            パースされたDataFrame
        """
        try:
            # StringIOを使ってCSVをパース
            df = pd.read_csv(StringIO(csv_text))
            
            # カラム名の標準化（大文字小文字の統一）
            df.columns = df.columns.str.title()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"CSV parsing failed: {e}")
            return pd.DataFrame()

    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stooqデータを標準フォーマットに変換
        
        Args:
            df: 変換対象のDataFrame
            
        Returns:
            標準フォーマットのDataFrame
        """
        try:
            # 列名マッピング（Stooq固有）
            column_mapping = {
                'Date': 'Date',
                'Open': 'Open', 
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                # Stooqの可能性のあるカラム名
                'O': 'Open',
                'H': 'High', 
                'L': 'Low',
                'C': 'Close',
                'V': 'Volume'
            }

            # 列名を標準名に変換
            df = df.rename(columns=column_mapping)

            # 基底クラスの標準フォーマット変換を呼び出し
            df = super()._convert_to_standard_format(df)

            return df

        except Exception as e:
            self.logger.warning(f"Stooq format conversion failed: {e}")
            return df

    def get_supported_periods(self) -> list:
        """Stooqでサポートされている期間一覧
        
        Returns:
            サポート期間のリスト
        """
        return ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y']

    def build_url(self, symbol: str, period: str) -> str:
        """Stooq APIのURL構築
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            
        Returns:
            構築されたURL
        """
        stooq_symbol = self._convert_to_stooq_symbol(symbol)
        days = self._period_to_days(period)
        return f"{self.config.base_url}?s={stooq_symbol}&d1={days}&i=d"

    def validate_symbol(self, symbol: str) -> bool:
        """銘柄コードの妥当性チェック
        
        Args:
            symbol: チェック対象の銘柄コード
            
        Returns:
            True: 有効, False: 無効
        """
        if not symbol or len(symbol.strip()) == 0:
            return False

        # 基本的な形式チェック
        if symbol.isdigit() and len(symbol) == 4:
            return True  # 日本株の4桁コード

        # サフィックス付きチェック
        for suffix in STOOQ_SUFFIXES + ['.T', '.JP', '.TO', '.TYO']:
            if symbol.endswith(suffix):
                base = symbol.replace(suffix, '')
                if base.isdigit():
                    return True

        return True  # 他の形式も一旦許可