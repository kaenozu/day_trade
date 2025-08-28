#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Yahoo Finance Provider
リアルデータプロバイダー V2 - Yahoo Finance プロバイダー

Yahoo Finance からのデータ取得機能
"""

import time
import logging
from typing import List

from .base_provider import BaseDataProvider
from .models import (
    DataSource, DataSourceConfig, DataFetchResult, DataQualityLevel,
    YAHOO_SUFFIXES
)

# yfinance 依存関係の安全な読み込み
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class ImprovedYahooFinanceProvider(BaseDataProvider):
    """改善版Yahoo Finance プロバイダー"""

    def __init__(self, config: DataSourceConfig):
        """Yahoo Finance プロバイダー初期化
        
        Args:
            config: データソース設定
        """
        super().__init__(config)
        self.symbol_variations_cache = {}

    async def get_stock_data(self, 
                           symbol: str, 
                           period: str = "1mo") -> DataFetchResult:
        """Yahoo Finance から株価データ取得
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            
        Returns:
            データ取得結果
        """
        start_time = time.time()

        # yfinance の可用性チェック
        if not YFINANCE_AVAILABLE:
            return DataFetchResult(
                data=None,
                source=DataSource.YAHOO_FINANCE,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=0.0,
                error_message="yfinance not available"
            )

        try:
            # 日次レート制限チェック
            if self.daily_request_count >= self.config.rate_limit_per_day:
                return DataFetchResult(
                    data=None,
                    source=DataSource.YAHOO_FINANCE,
                    quality_level=DataQualityLevel.FAILED,
                    quality_score=0.0,
                    fetch_time=0.0,
                    error_message="Daily rate limit exceeded"
                )

            # レート制限待機
            await self._wait_for_rate_limit()

            # 複数の銘柄コード形式を試行
            symbol_variations = self._generate_symbol_variations(symbol)
            last_error = None

            for ticker_symbol in symbol_variations:
                try:
                    self.logger.debug(f"Trying symbol variation: {ticker_symbol}")

                    # yfinance でデータ取得
                    ticker = yf.Ticker(ticker_symbol)
                    data = ticker.history(period=period)

                    self._record_request()

                    if not data.empty and len(data) > 0:
                        # データ品質チェック
                        quality_level, quality_score = self._calculate_quality_score(
                            data
                        )

                        if quality_score >= self.config.quality_threshold:
                            fetch_time = time.time() - start_time

                            self.logger.info(
                                f"Successfully fetched {symbol} as {ticker_symbol} "
                                f"(quality: {quality_score:.1f})"
                            )

                            return DataFetchResult(
                                data=data,
                                source=DataSource.YAHOO_FINANCE,
                                quality_level=quality_level,
                                quality_score=quality_score,
                                fetch_time=fetch_time,
                                metadata={
                                    'symbol_used': ticker_symbol,
                                    'variations_tried': symbol_variations.index(
                                        ticker_symbol
                                    ) + 1
                                }
                            )
                        else:
                            self.logger.debug(
                                f"Data quality insufficient for {ticker_symbol}: "
                                f"{quality_score:.1f}"
                            )
                            last_error = f"Data quality too low: {quality_score:.1f}"

                except Exception as e:
                    last_error = str(e)
                    self.logger.debug(f"Failed to fetch {ticker_symbol}: {e}")
                    continue

            # 全てのバリエーションが失敗
            fetch_time = time.time() - start_time
            error_details = self._generate_error_details(
                symbol, symbol_variations, last_error
            )

            self.logger.warning(
                f"All symbol variations failed for {symbol}: {error_details}"
            )

            return DataFetchResult(
                data=None,
                source=DataSource.YAHOO_FINANCE,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=fetch_time,
                error_message=error_details,
                metadata={'variations_tried': len(symbol_variations)}
            )

        except Exception as e:
            fetch_time = time.time() - start_time
            error_msg = f"Yahoo Finance fetch error: {e}"
            self.logger.error(error_msg)

            return DataFetchResult(
                data=None,
                source=DataSource.YAHOO_FINANCE,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=fetch_time,
                error_message=error_msg
            )

    def _generate_symbol_variations(self, symbol: str) -> List[str]:
        """銘柄コードバリエーション生成
        
        Args:
            symbol: 元の銘柄コード
            
        Returns:
            試行する銘柄コードのリスト
        """
        # キャッシュ確認
        if symbol in self.symbol_variations_cache:
            return self.symbol_variations_cache[symbol]

        variations = []

        if symbol.isdigit():
            # 数字のみの銘柄コード（日本株）
            variations.extend([
                f"{symbol}.T",      # 東京証券取引所（最優先）
                f"{symbol}.JP",     # 日本
                symbol,             # そのまま
                f"{symbol}.TO",     # 東京
                f"{symbol}.TYO"     # Tokyo
            ])
        else:
            # すでにサフィックス付きの場合
            variations.append(symbol)

            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
                if base_symbol.isdigit():
                    # 数字ベースの場合は他のサフィックスも試行
                    for suffix in YAHOO_SUFFIXES:
                        candidate = f"{base_symbol}{suffix}"
                        if candidate != symbol:
                            variations.append(candidate)
                    variations.append(base_symbol)

        # 重複除去（順序保持）
        variations = list(dict.fromkeys(variations))

        # キャッシュに保存
        self.symbol_variations_cache[symbol] = variations

        return variations

    def _generate_error_details(self, 
                              symbol: str, 
                              variations: List[str], 
                              last_error: str) -> str:
        """エラー詳細情報生成
        
        Args:
            symbol: 元の銘柄コード
            variations: 試行したバリエーション
            last_error: 最後のエラーメッセージ
            
        Returns:
            詳細エラーメッセージ
        """
        details = [
            f"Symbol: {symbol}",
            f"Variations tried: {len(variations)}",
            f"Variations: {', '.join(variations)}",
            f"Last error: {last_error or 'Unknown'}",
            f"Daily requests: {self.daily_request_count}/"
            f"{self.config.rate_limit_per_day}"
        ]
        return " | ".join(details)

    def clear_symbol_cache(self):
        """銘柄バリエーションキャッシュクリア"""
        self.symbol_variations_cache.clear()
        self.logger.info("Cleared symbol variations cache")

    def get_cached_symbols(self) -> List[str]:
        """キャッシュされた銘柄一覧取得
        
        Returns:
            キャッシュされた銘柄コードのリスト
        """
        return list(self.symbol_variations_cache.keys())

    def preload_symbol_variations(self, symbols: List[str]):
        """銘柄バリエーションの事前ロード
        
        Args:
            symbols: 事前ロードする銘柄コードのリスト
        """
        for symbol in symbols:
            if symbol not in self.symbol_variations_cache:
                self._generate_symbol_variations(symbol)
        
        self.logger.info(f"Preloaded variations for {len(symbols)} symbols")