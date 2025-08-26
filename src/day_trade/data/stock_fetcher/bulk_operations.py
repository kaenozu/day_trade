"""
一括データ取得操作
複数銘柄の効率的な一括処理機能
"""

import time
from typing import Dict, List, Optional, Any
import logging

import pandas as pd

from ...utils.yfinance_import import get_yfinance
from ...utils.logging_config import log_performance_metric
from .exceptions import DataNotFoundError

# yfinance統一インポート
yf, YFINANCE_AVAILABLE = get_yfinance()


class BulkOperationsMixin:
    """一括操作機能のミックスイン"""

    def bulk_get_company_info(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, Optional[Dict]]:
        """
        複数銘柄の企業情報を一括取得（yfinance.Tickers使用）

        Args:
            codes: 銘柄コードのリスト
            batch_size: 一回の処理で取得する銘柄数
            delay: バッチ間の遅延（秒）

        Returns:
            {銘柄コード: 企業情報辞書} の辞書
        """
        if not codes:
            return {}

        results = {}
        start_time = time.time()

        # バッチごとに処理
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            batch_start = time.time()

            try:
                # yfinance.Tickersを使用して一括取得
                if not YFINANCE_AVAILABLE:
                    logging.error("yfinanceが利用できません")
                    continue

                tickers = yf.Tickers(" ".join(f"{code}.T" for code in batch_codes))

                for code in batch_codes:
                    try:
                        ticker_symbol = f"{code}.T"
                        ticker = tickers.tickers.get(ticker_symbol, None)

                        if ticker:
                            info = ticker.info
                            if info and info.get("symbol"):
                                # 企業情報を整理
                                company_info = {
                                    "name": info.get(
                                        "longName", info.get("shortName", "")
                                    ),
                                    "sector": info.get("sector", ""),
                                    "industry": info.get("industry", ""),
                                    "country": info.get("country", ""),
                                    "website": info.get("website", ""),
                                    "business_summary": info.get("businessSummary", ""),
                                    "market_cap": info.get("marketCap"),
                                    "employees": info.get("fullTimeEmployees"),
                                    "symbol": code,
                                }
                                results[code] = company_info

                                # キャッシュに保存
                                cache_key = f"company_info_{code}"
                                if hasattr(self, "_data_cache") and self._data_cache:
                                    self._data_cache.set(cache_key, company_info, ttl=3600)
                            else:
                                if hasattr(self, "logger"):
                                    self.logger.warning(f"企業情報が取得できません: {code}")
                                results[code] = None
                        else:
                            if hasattr(self, "logger"):
                                self.logger.warning(f"Tickerオブジェクトが取得できません: {code}")
                            results[code] = None

                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.error(f"個別銘柄処理エラー {code}: {e}")
                        results[code] = None

                batch_elapsed = time.time() - batch_start
                log_performance_metric(
                    "bulk_company_info_batch",
                    {
                        "batch_size": len(batch_codes),
                        "batch_index": i // batch_size,
                        "elapsed_ms": batch_elapsed * 1000,
                        "codes_processed": len(batch_codes),
                    },
                )

                # レート制限対応の遅延
                if delay > 0 and i + batch_size < len(codes):
                    time.sleep(delay)

            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"バッチ処理エラー (codes {i}-{i + batch_size - 1}): {e}")
                # バッチ失敗時は個別に処理
                for code in batch_codes:
                    try:
                        results[code] = self.get_company_info(code)
                    except Exception:
                        results[code] = None

        total_elapsed = time.time() - start_time
        successful_count = sum(1 for result in results.values() if result is not None)

        log_performance_metric(
            "bulk_company_info_complete",
            {
                "total_codes": len(codes),
                "successful_count": successful_count,
                "failure_count": len(codes) - successful_count,
                "success_rate": successful_count / len(codes) if codes else 0,
                "total_elapsed_ms": total_elapsed * 1000,
                "avg_time_per_code": (
                    (total_elapsed / len(codes)) * 1000 if codes else 0
                ),
            },
        )

        if hasattr(self, "logger"):
            self.logger.info(
                f"一括企業情報取得完了: {successful_count}/{len(codes)}件成功 "
                f"({total_elapsed:.2f}秒)"
            )

        return results

    def bulk_get_current_prices_optimized(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, Optional[Dict]]:
        """
        最適化された複数銘柄の現在価格一括取得（yf.download使用）

        Args:
            codes: 銘柄コードのリスト
            batch_size: 一回の処理で取得する銘柄数
            delay: バッチ間の遅延（秒）

        Returns:
            {銘柄コード: 価格情報辞書} の辞書
        """
        if not codes:
            return {}

        results = {}
        start_time = time.time()

        # バッチごとに処理
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            batch_start = time.time()

            try:
                # yfinance.downloadを使用した真の一括取得
                if not YFINANCE_AVAILABLE:
                    logging.error("yfinanceが利用できません")
                    continue

                symbols = [f"{code}.T" for code in batch_codes]

                # 過去2日分のデータを一括取得（最新価格と前日価格を取得するため）
                data = yf.download(
                    symbols,
                    period="2d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    prepost=True,
                    threads=True,  # 並列処理を有効化
                    progress=False,
                )

                # 各銘柄の価格情報を処理
                for code in batch_codes:
                    try:
                        symbol = f"{code}.T"

                        # データが複数銘柄の場合
                        if len(batch_codes) > 1:
                            if (
                                hasattr(data.columns, "levels")
                                and symbol in data.columns.levels[0]
                            ):
                                ticker_data = data[symbol]
                            else:
                                results[code] = None
                                continue
                        else:
                            # 単一銘柄の場合
                            ticker_data = data

                        # 最新の価格データを取得
                        if not ticker_data.empty and "Close" in ticker_data.columns:
                            # 最新の取引日のデータ
                            latest_data = (
                                ticker_data.dropna().iloc[-1]
                                if not ticker_data.dropna().empty
                                else None
                            )

                            if latest_data is not None:
                                current_price = float(latest_data["Close"])

                                # 前日価格を計算（変化率算出用）
                                previous_price = current_price  # デフォルト
                                change = 0.0
                                change_percent = 0.0

                                clean_data = ticker_data.dropna()
                                if len(clean_data) >= 2:
                                    previous_data = clean_data.iloc[-2]
                                    previous_price = float(previous_data["Close"])
                                    change = current_price - previous_price
                                    change_percent = (
                                        (change / previous_price * 100)
                                        if previous_price != 0
                                        else 0.0
                                    )

                                # 結果を構築
                                results[code] = {
                                    "current_price": current_price,
                                    "change": change,
                                    "change_percent": change_percent,
                                    "volume": int(latest_data.get("Volume", 0)),
                                    "high": float(
                                        latest_data.get("High", current_price)
                                    ),
                                    "low": float(latest_data.get("Low", current_price)),
                                    "open": float(
                                        latest_data.get("Open", current_price)
                                    ),
                                    "previous_close": previous_price,
                                    "timestamp": (
                                        latest_data.name.isoformat()
                                        if hasattr(latest_data.name, "isoformat")
                                        else None
                                    ),
                                }
                            else:
                                results[code] = None
                        else:
                            results[code] = None

                    except Exception as e:
                        if hasattr(self, "logger"):
                            self.logger.warning(f"個別処理エラー {code}: {e}")
                        results[code] = None

                batch_elapsed = time.time() - batch_start
                log_performance_metric(
                    "bulk_current_price_optimized_batch",
                    {
                        "batch_size": len(batch_codes),
                        "batch_index": i // batch_size,
                        "elapsed_ms": batch_elapsed * 1000,
                        "codes_processed": len(batch_codes),
                    },
                )

                # レート制限対応の遅延
                if delay > 0 and i + batch_size < len(codes):
                    time.sleep(delay)

            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"最適化バッチ処理エラー (codes {i}-{i + batch_size - 1}): {e}"
                    )
                # フォールバック: 既存のバルク取得メソッドを使用
                for code in batch_codes:
                    try:
                        results[code] = self.get_current_price(code)
                    except Exception:
                        results[code] = None

        total_elapsed = time.time() - start_time
        successful_count = sum(1 for result in results.values() if result is not None)

        log_performance_metric(
            "bulk_current_price_optimized_complete",
            {
                "total_codes": len(codes),
                "successful_count": successful_count,
                "failure_count": len(codes) - successful_count,
                "success_rate": successful_count / len(codes) if codes else 0,
                "total_elapsed_ms": total_elapsed * 1000,
                "avg_time_per_code": (
                    (total_elapsed / len(codes)) * 1000 if codes else 0
                ),
            },
        )

        if hasattr(self, "logger"):
            self.logger.info(
                f"最適化一括価格取得完了: {successful_count}/{len(codes)}件成功 ({total_elapsed:.2f}秒)"
            )
        return results

    def bulk_get_historical_data(
        self,
        codes: List[str],
        period: str = "1y",
        interval: str = "1d",
        batch_size: int = 50,
        delay: float = 0.1,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        複数銘柄のヒストリカルデータを一括取得

        Args:
            codes: 銘柄コードのリスト
            period: 期間（1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max）
            interval: 間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）
            batch_size: 一回の処理で取得する銘柄数
            delay: バッチ間の遅延（秒）

        Returns:
            {銘柄コード: 価格データのDataFrame} の辞書
        """
        if not codes:
            return {}

        results = {}
        start_time = time.time()

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            batch_start = time.time()
            formatted_symbols = [self._format_symbol(code) for code in batch_codes]

            try:
                # yf.downloadを使用した真の一括取得
                if not YFINANCE_AVAILABLE:
                    logging.error("yfinanceが利用できません")
                    continue

                data = yf.download(
                    formatted_symbols,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    prepost=True,
                    threads=True,  # 並列処理を有効化
                    progress=False,
                )

                for code in batch_codes:
                    symbol = self._format_symbol(code)
                    ticker_data = None

                    if len(formatted_symbols) > 1 and hasattr(data.columns, "levels"):
                        # 複数銘柄の場合、MultiIndexからデータを抽出
                        if symbol in data.columns.levels[0]:
                            ticker_data = data[symbol]
                    else:
                        # 単一銘柄の場合、直接データを使用
                        ticker_data = data

                    if ticker_data is not None and not ticker_data.empty:
                        if ticker_data.index.tz is not None:
                            ticker_data.index = ticker_data.index.tz_localize(None)
                        results[code] = ticker_data
                    else:
                        if hasattr(self, "logger"):
                            self.logger.warning(
                                f"ヒストリカルデータが取得できませんでした: {code}"
                            )
                        results[code] = None

            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"ヒストリカルデータバッチ処理エラー (codes {i}-{i + batch_size - 1}): {e}"
                    )
                # バッチ失敗時は個別に処理（フォールバック）
                for code in batch_codes:
                    try:
                        results[code] = self.get_historical_data(code, period, interval)
                    except Exception:
                        results[code] = None

            batch_elapsed = time.time() - batch_start
            log_performance_metric(
                "bulk_historical_data_batch",
                {
                    "batch_size": len(batch_codes),
                    "batch_index": i // batch_size,
                    "elapsed_ms": batch_elapsed * 1000,
                    "codes_processed": len(batch_codes),
                },
            )

            if delay > 0 and i + batch_size < len(codes):
                time.sleep(delay)

        total_elapsed = time.time() - start_time
        successful_count = sum(1 for result in results.values() if result is not None)

        log_performance_metric(
            "bulk_historical_data_complete",
            {
                "total_codes": len(codes),
                "successful_count": successful_count,
                "failure_count": len(codes) - successful_count,
                "success_rate": successful_count / len(codes) if codes else 0,
                "total_elapsed_ms": total_elapsed * 1000,
                "avg_time_per_code": (
                    (total_elapsed / len(codes)) * 1000 if codes else 0
                ),
            },
        )

        if hasattr(self, "logger"):
            self.logger.info(
                f"一括ヒストリカルデータ取得完了: {successful_count}/{len(codes)}件成功 ({total_elapsed:.2f}秒)"
            )
        return results