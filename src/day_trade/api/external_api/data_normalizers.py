#!/usr/bin/env python3
"""
外部APIクライアント - データ正規化機能
"""

import random
from datetime import datetime
from io import StringIO
from typing import Optional

import pandas as pd

from ...utils.logging_config import get_context_logger
from .enums import APIProvider
from .models import APIResponse

logger = get_context_logger(__name__)


class DataNormalizer:
    """データ正規化クラス"""

    def __init__(self):
        pass

    async def normalize_response_data(
        self, response: APIResponse
    ) -> Optional[pd.DataFrame]:
        """レスポンスデータ正規化"""
        try:
            provider = response.request.endpoint.provider
            data_type = response.request.endpoint.data_type

            if provider == APIProvider.MOCK_PROVIDER:
                return await self._normalize_mock_data(response)
            elif provider == APIProvider.YAHOO_FINANCE:
                return await self._normalize_yahoo_finance_data(response)
            elif provider == APIProvider.ALPHA_VANTAGE:
                return await self._normalize_alpha_vantage_data(response)
            else:
                # 汎用正規化
                return await self._normalize_generic_data(response)

        except Exception as e:
            logger.error(f"データ正規化エラー: {e}")
            return None

    async def _normalize_mock_data(self, response: APIResponse) -> pd.DataFrame:
        """モックデータ正規化"""
        # 模擬的なデータ正規化
        data = response.response_data

        if isinstance(data, dict) and "price_data" in data:
            price_data = data["price_data"]

            df = pd.DataFrame(
                {
                    "timestamp": [datetime.now()],
                    "open": [price_data.get("open", 1000)],
                    "high": [price_data.get("high", 1050)],
                    "low": [price_data.get("low", 950)],
                    "close": [price_data.get("close", 1025)],
                    "volume": [price_data.get("volume", 1000000)],
                    "symbol": [response.request.params.get("symbol", "UNKNOWN")],
                }
            )

            return df

        # フォールバック: 模擬データ生成
        symbol = response.request.params.get("symbol", "MOCK")
        base_price = 1000 + hash(symbol) % 2000

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [base_price * random.uniform(0.98, 1.02)],
                "high": [base_price * random.uniform(1.01, 1.05)],
                "low": [base_price * random.uniform(0.95, 0.99)],
                "close": [base_price * random.uniform(0.99, 1.03)],
                "volume": [random.randint(100000, 500000)],
                "symbol": [symbol],
            }
        )

        return df

    async def _normalize_yahoo_finance_data(
        self, response: APIResponse
    ) -> pd.DataFrame:
        """Yahoo Finance データ正規化"""
        try:
            data = response.response_data

            # Yahoo Finance API レスポンス構造に基づく解析
            if isinstance(data, dict) and "chart" in data:
                chart_data = data["chart"]["result"][0]
                timestamps = chart_data["timestamp"]
                quotes = chart_data["indicators"]["quote"][0]

                df = pd.DataFrame(
                    {
                        "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps],
                        "open": quotes["open"],
                        "high": quotes["high"],
                        "low": quotes["low"],
                        "close": quotes["close"],
                        "volume": quotes["volume"],
                        "symbol": [chart_data["meta"]["symbol"]] * len(timestamps),
                    }
                )

                # 欠損値処理
                df = df.fillna(method="ffill").fillna(method="bfill")

                return df

        except Exception as e:
            logger.error(f"Yahoo Finance データ正規化エラー: {e}")

        return pd.DataFrame()

    async def _normalize_alpha_vantage_data(
        self, response: APIResponse
    ) -> pd.DataFrame:
        """Alpha Vantage データ正規化"""
        try:
            data = response.response_data

            # Alpha Vantage API レスポンス構造に基づく解析
            if isinstance(data, dict) and "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                symbol = data["Meta Data"]["2. Symbol"]

                records = []
                for date_str, prices in time_series.items():
                    records.append(
                        {
                            "timestamp": datetime.strptime(date_str, "%Y-%m-%d"),
                            "open": float(prices["1. open"]),
                            "high": float(prices["2. high"]),
                            "low": float(prices["3. low"]),
                            "close": float(prices["4. close"]),
                            "volume": int(prices["5. volume"]),
                            "symbol": symbol,
                        }
                    )

                df = pd.DataFrame(records)
                df = df.sort_values("timestamp")

                return df

        except Exception as e:
            logger.error(f"Alpha Vantage データ正規化エラー: {e}")

        return pd.DataFrame()

    async def _normalize_generic_data(self, response: APIResponse) -> pd.DataFrame:
        """汎用データ正規化"""
        # 基本的な正規化処理
        data = response.response_data

        if isinstance(data, list) and data:
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            return pd.DataFrame()

    def parse_csv_response(self, csv_text: str) -> pd.DataFrame:
        """CSVレスポンス解析（セキュリティ強化版）"""
        try:
            # セキュリティ強化: CSVファイルサイズ制限
            if len(csv_text) > 10 * 1024 * 1024:  # 10MB制限
                logger.warning(f"CSVファイルが大きすぎます: {len(csv_text)}バイト")
                raise ValueError("CSVファイルサイズが制限を超過しています")

            # セキュリティ強化: 行数制限
            line_count = csv_text.count("\n") + 1
            if line_count > 50000:  # 50,000行制限
                logger.warning(f"CSV行数が多すぎます: {line_count}行")
                raise ValueError("CSV行数が制限を超過しています")

            # セキュリティ強化: 危険なCSVパターンチェック
            dangerous_csv_patterns = [
                "=cmd|",  # Excelコマンド実行
                "=system(",  # システムコマンド
                "@SUM(",  # Excel関数インジェクション
                "=HYPERLINK(",  # ハイパーリンクインジェクション
                "javascript:",  # JavaScriptスキーム
                "data:text/html",  # HTMLデータスキーム
            ]

            for pattern in dangerous_csv_patterns:
                if pattern.lower() in csv_text.lower():
                    logger.warning(f"危険なCSVパターンを検出: {pattern}")
                    raise ValueError("CSVデータに危険なパターンが含まれています")

            # 安全なCSV読み込み設定
            return pd.read_csv(
                StringIO(csv_text),
                nrows=50000,  # 行数制限（重複チェック）
                memory_map=False,  # メモリマップ無効
                low_memory=False,  # 低メモリモード無効（安全性優先）
                engine="python",  # Pythonエンジン使用（C拡張の脆弱性回避）
            )

        except pd.errors.EmptyDataError:
            logger.warning("空のCSVデータを受信")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            # CSV解析エラーの安全なメッセージ化
            logger.error(f"CSV解析エラー: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            # その他のエラーの安全なメッセージ化
            logger.error(f"CSV処理エラー: {str(e)}")
            return pd.DataFrame()

    def validate_normalized_data(self, df: pd.DataFrame) -> bool:
        """正規化されたデータの検証"""
        if df.empty:
            logger.warning("正規化されたデータが空です")
            return False

        # 基本的な列の存在チェック
        required_columns = ["timestamp", "symbol"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(f"必要な列が不足しています: {missing_columns}")
            return False

        # データ型チェック
        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                except Exception as e:
                    logger.error(f"timestamp列の変換に失敗: {e}")
                    return False

        # 数値列のチェック
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"数値列が非数値型です: {col}")

        logger.debug(f"データ検証成功: {len(df)}行, {len(df.columns)}列")
        return True