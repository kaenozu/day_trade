#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Quality System - 強化データ品質システム

複数ソース照合・異常値検出・品質スコアリングによる信頼性向上
Issue #795実装：データ品質向上 - リアルタイムデータ統合と異常値除去
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import sqlite3
import statistics
from collections import defaultdict, deque
import aiohttp
import time

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class DataSource(Enum):
    """データソース"""
    YAHOO_FINANCE = "Yahoo Finance"
    STOOQ = "Stooq"
    ALPHA_VANTAGE = "Alpha Vantage"
    KABUCOM = "auカブコム証券"
    RAKUTEN = "楽天証券"

class AnomalyType(Enum):
    """異常タイプ"""
    PRICE_OUTLIER = "価格外れ値"
    MISSING_DATA = "データ欠損"
    TIME_INCONSISTENCY = "時系列不整合"
    SOURCE_DIVERGENCE = "ソース間乖離"
    VOLUME_ANOMALY = "出来高異常"
    FUTURE_DATE = "未来日付"

class QualityLevel(Enum):
    """品質レベル"""
    EXCELLENT = "優秀"    # 95-100%
    GOOD = "良好"         # 85-94%
    FAIR = "普通"         # 70-84%
    POOR = "低品質"       # 50-69%
    CRITICAL = "危険"     # 0-49%

@dataclass
class DataPoint:
    """データポイント"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    source: DataSource
    quality_score: float
    anomalies: List[str] = field(default_factory=list)

@dataclass
class SourceComparison:
    """ソース比較結果"""
    symbol: str
    timestamp: datetime
    price_differences: Dict[str, float]  # ソース間価格差
    max_divergence: float               # 最大乖離率
    consensus_price: float              # コンセンサス価格
    reliability_scores: Dict[str, float] # ソース別信頼度
    quality_assessment: QualityLevel

@dataclass
class QualityReport:
    """品質レポート"""
    symbol: str
    period_start: datetime
    period_end: datetime
    overall_score: float
    source_scores: Dict[str, float]
    anomaly_count: int
    data_completeness: float
    price_consistency: float
    recommendations: List[str]

class MultiSourceDataValidator:
    """複数ソースデータ検証システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("enhanced_quality_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "quality_validation.db"
        self._init_database()

        # 検証ルール
        self.validation_rules = {
            'max_price_divergence': 3.0,      # 3%以上の乖離で警告
            'min_volume_threshold': 1000,     # 最小出来高
            'max_daily_change': 30.0,         # 1日30%以上変動で異常
            'future_date_tolerance': 0,       # 未来日付許容日数
            'missing_data_threshold': 5.0,    # 5%以上欠損で警告
        }

        # ソース別信頼度（実績ベース）
        self.source_reliability = {
            DataSource.YAHOO_FINANCE: 85.0,
            DataSource.STOOQ: 75.0,
            DataSource.ALPHA_VANTAGE: 90.0,
            DataSource.KABUCOM: 95.0,
            DataSource.RAKUTEN: 90.0
        }

        # キャッシュ
        self.validation_cache = {}
        self.cache_expiry = 300  # 5分間キャッシュ

        self.logger.info("Multi-source data validator initialized")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # データ検証テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT,
                    quality_score REAL,
                    anomaly_count INTEGER,
                    max_divergence REAL,
                    consensus_price REAL,
                    validation_time TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 異常検出ログ
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    anomaly_type TEXT,
                    severity TEXT,
                    description TEXT,
                    data_point TEXT,
                    detected_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 品質履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    overall_score REAL,
                    completeness REAL,
                    consistency REAL,
                    source_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def validate_multiple_sources(self, symbol: str, period: str = "1mo") -> SourceComparison:
        """複数ソースでの価格データ検証"""

        self.logger.info(f"Validating {symbol} across multiple sources")

        # キャッシュチェック
        cache_key = f"{symbol}_{period}"
        if cache_key in self.validation_cache:
            cache_time, result = self.validation_cache[cache_key]
            if time.time() - cache_time < self.cache_expiry:
                return result

        # 各ソースからデータ取得
        source_data = {}

        # Yahoo Finance
        if YFINANCE_AVAILABLE:
            yahoo_data = await self._fetch_yahoo_finance(symbol, period)
            if yahoo_data is not None:
                source_data[DataSource.YAHOO_FINANCE] = yahoo_data

        # Stooq
        stooq_data = await self._fetch_stooq(symbol, period)
        if stooq_data is not None:
            source_data[DataSource.STOOQ] = stooq_data

        # データ比較・検証
        if len(source_data) < 1:
            raise ValueError(f"No data sources available for {symbol}")

        comparison = await self._compare_sources(symbol, source_data)

        # キャッシュに保存
        self.validation_cache[cache_key] = (time.time(), comparison)

        # データベースに保存
        await self._save_validation_result(comparison)

        return comparison

    async def _fetch_yahoo_finance(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Yahoo Financeからデータ取得"""

        try:
            # 複数の銘柄コード形式を試行
            symbol_variations = [f"{symbol}.T", f"{symbol}.JP", symbol]

            for ticker_symbol in symbol_variations:
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    data = ticker.history(period=period)

                    if not data.empty and len(data) > 0:
                        # 異常値チェック
                        anomalies = self._detect_basic_anomalies(data, symbol, DataSource.YAHOO_FINANCE)
                        if len(anomalies) < 5:  # 異常値が少ない場合のみ採用
                            self.logger.info(f"Yahoo Finance data retrieved for {symbol} as {ticker_symbol}")
                            return data

                except Exception as e:
                    self.logger.debug(f"Yahoo Finance failed for {ticker_symbol}: {e}")
                    continue

            return None

        except Exception as e:
            self.logger.error(f"Yahoo Finance fetch error: {e}")
            return None

    async def _fetch_stooq(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Stooqからデータ取得"""

        try:
            # Stooq用の銘柄コード変換
            stooq_symbol = f"{symbol}.jp"

            # 期間を日数に変換
            days = {'1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365}.get(period, 30)

            url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&d1={days}&i=d"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        text = await response.text()

                        from io import StringIO
                        df = pd.read_csv(StringIO(text))

                        if not df.empty and 'Close' in df.columns:
                            # 列名をYahoo Finance形式に統一
                            df = df.rename(columns={
                                'Date': 'Date', 'Open': 'Open', 'High': 'High',
                                'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                            })

                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.set_index('Date')

                            self.logger.info(f"Stooq data retrieved for {symbol}")
                            return df

            return None

        except Exception as e:
            self.logger.error(f"Stooq fetch error: {e}")
            return None

    def _detect_basic_anomalies(self, data: pd.DataFrame, symbol: str, source: DataSource) -> List[str]:
        """基本的な異常値検出"""

        anomalies = []

        try:
            # 未来日付チェック
            today = datetime.now().date()
            future_dates = data.index[data.index.date > today]
            if len(future_dates) > 0:
                anomalies.append(f"未来日付: {len(future_dates)}件")

            # 価格の異常値チェック
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns:
                    # ゼロ以下の価格
                    invalid_prices = (data[col] <= 0).sum()
                    if invalid_prices > 0:
                        anomalies.append(f"{col}にゼロ以下価格: {invalid_prices}件")

                    # 極端に高い価格（100万円超）
                    extreme_prices = (data[col] > 1000000).sum()
                    if extreme_prices > 0:
                        anomalies.append(f"{col}に極端価格: {extreme_prices}件")

            # OHLC関係の整合性
            ohlc_errors = 0
            for idx in data.index:
                try:
                    h, l, o, c = data.loc[idx, ['High', 'Low', 'Open', 'Close']]
                    if h < l or h < o or h < c or l > o or l > c:
                        ohlc_errors += 1
                except:
                    continue

            if ohlc_errors > 0:
                anomalies.append(f"OHLC整合性エラー: {ohlc_errors}件")

            # 日次変動率の異常
            if 'Close' in data.columns and len(data) > 1:
                daily_returns = data['Close'].pct_change().dropna()
                extreme_returns = (abs(daily_returns) > 0.3).sum()  # 30%超変動
                if extreme_returns > 0:
                    anomalies.append(f"極端変動: {extreme_returns}件")

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            anomalies.append(f"検出エラー: {str(e)}")

        return anomalies

    async def _compare_sources(self, symbol: str, source_data: Dict[DataSource, pd.DataFrame]) -> SourceComparison:
        """ソース間データ比較"""

        if len(source_data) == 1:
            # 単一ソースの場合
            source, data = next(iter(source_data.items()))
            latest_date = data.index[-1]
            latest_price = data['Close'].iloc[-1]

            return SourceComparison(
                symbol=symbol,
                timestamp=latest_date,
                price_differences={},
                max_divergence=0.0,
                consensus_price=latest_price,
                reliability_scores={source.value: self.source_reliability[source]},
                quality_assessment=QualityLevel.FAIR
            )

        # 複数ソース比較
        price_differences = {}
        prices = {}
        reliability_scores = {}

        # 最新の共通日付を見つける
        common_dates = None
        for source, data in source_data.items():
            if common_dates is None:
                common_dates = set(data.index)
            else:
                common_dates = common_dates.intersection(set(data.index))

        if not common_dates:
            # 共通日付がない場合、最も近い日付で比較
            latest_date = max(max(data.index) for data in source_data.values())

            for source, data in source_data.items():
                closest_date = data.index[data.index <= latest_date][-1]
                prices[source.value] = data.loc[closest_date, 'Close']
                reliability_scores[source.value] = self.source_reliability[source]
        else:
            # 共通日付での比較
            latest_common_date = max(common_dates)

            for source, data in source_data.items():
                prices[source.value] = data.loc[latest_common_date, 'Close']
                reliability_scores[source.value] = self.source_reliability[source]

        # 価格差計算
        price_values = list(prices.values())
        avg_price = statistics.mean(price_values)

        for source_name, price in prices.items():
            difference = abs(price - avg_price) / avg_price * 100
            price_differences[source_name] = difference

        max_divergence = max(price_differences.values()) if price_differences else 0

        # 重み付き平均でコンセンサス価格計算
        weighted_sum = sum(price * reliability_scores[source] for source, price in prices.items())
        total_weight = sum(reliability_scores.values())
        consensus_price = weighted_sum / total_weight if total_weight > 0 else avg_price

        # 品質評価
        if max_divergence < 1.0:
            quality = QualityLevel.EXCELLENT
        elif max_divergence < 3.0:
            quality = QualityLevel.GOOD
        elif max_divergence < 5.0:
            quality = QualityLevel.FAIR
        elif max_divergence < 10.0:
            quality = QualityLevel.POOR
        else:
            quality = QualityLevel.CRITICAL

        return SourceComparison(
            symbol=symbol,
            timestamp=latest_common_date if common_dates else latest_date,
            price_differences=price_differences,
            max_divergence=max_divergence,
            consensus_price=consensus_price,
            reliability_scores=reliability_scores,
            quality_assessment=quality
        )

    async def _save_validation_result(self, comparison: SourceComparison):
        """検証結果の保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO data_validations
                    (symbol, timestamp, quality_score, max_divergence, consensus_price)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    comparison.symbol,
                    comparison.timestamp.isoformat(),
                    comparison.quality_assessment.value,
                    comparison.max_divergence,
                    comparison.consensus_price
                ))

        except Exception as e:
            self.logger.error(f"Failed to save validation result: {e}")

    async def generate_quality_report(self, symbol: str, days: int = 30) -> QualityReport:
        """品質レポート生成"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT AVG(max_divergence), COUNT(*), MIN(consensus_price), MAX(consensus_price)
                    FROM data_validations
                    WHERE symbol = ? AND timestamp >= ?
                """, (symbol, start_date.isoformat()))

                result = cursor.fetchone()
                avg_divergence, validation_count, min_price, max_price = result or (0, 0, 0, 0)

                # 総合スコア計算
                divergence_score = max(0, 100 - avg_divergence * 10) if avg_divergence else 80
                completeness_score = min(100, validation_count * 2) if validation_count else 0
                overall_score = (divergence_score + completeness_score) / 2

                # 推奨事項
                recommendations = []
                if avg_divergence > 5.0:
                    recommendations.append("データソース間の乖離が大きすぎます")
                if validation_count < 10:
                    recommendations.append("検証回数が不足しています")
                if overall_score < 70:
                    recommendations.append("データ品質の改善が必要です")

                if not recommendations:
                    recommendations.append("データ品質は良好です")

                return QualityReport(
                    symbol=symbol,
                    period_start=start_date,
                    period_end=end_date,
                    overall_score=overall_score,
                    source_scores={},  # 実装時に詳細化
                    anomaly_count=0,   # 実装時に詳細化
                    data_completeness=completeness_score,
                    price_consistency=divergence_score,
                    recommendations=recommendations
                )

        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")
            return QualityReport(
                symbol=symbol, period_start=start_date, period_end=end_date,
                overall_score=0, source_scores={}, anomaly_count=0,
                data_completeness=0, price_consistency=0,
                recommendations=["レポート生成エラー"]
            )

    async def get_reliable_price(self, symbol: str) -> Tuple[float, float]:
        """信頼性の高い価格取得"""

        try:
            comparison = await self.validate_multiple_sources(symbol, "1d")

            confidence = 100 - comparison.max_divergence
            confidence = max(0, min(100, confidence))

            return comparison.consensus_price, confidence

        except Exception as e:
            self.logger.error(f"Failed to get reliable price for {symbol}: {e}")
            return 0.0, 0.0

# グローバルインスタンス
enhanced_data_validator = MultiSourceDataValidator()

# テスト関数
async def test_enhanced_data_quality():
    """強化データ品質システムのテスト"""

    print("=== 強化データ品質システム テスト ===")

    validator = MultiSourceDataValidator()

    # テスト銘柄
    test_symbols = ["7203", "8306", "4751"]

    print(f"\n[ {len(test_symbols)}銘柄の複数ソース検証 ]")

    for symbol in test_symbols:
        print(f"\n--- {symbol} 品質検証 ---")

        try:
            # 複数ソース検証
            comparison = await validator.validate_multiple_sources(symbol, "1mo")

            print(f"品質評価: {comparison.quality_assessment.value}")
            print(f"最大乖離率: {comparison.max_divergence:.2f}%")
            print(f"コンセンサス価格: ¥{comparison.consensus_price:.2f}")

            if comparison.price_differences:
                print("ソース間価格差:")
                for source, diff in comparison.price_differences.items():
                    print(f"  {source}: {diff:.2f}%")

            # 信頼性価格取得
            reliable_price, confidence = await validator.get_reliable_price(symbol)
            print(f"信頼性価格: ¥{reliable_price:.2f} (信頼度: {confidence:.1f}%)")

            # 品質レポート生成
            report = await validator.generate_quality_report(symbol, 30)
            print(f"30日品質スコア: {report.overall_score:.1f}")
            print(f"主要推奨: {report.recommendations[0]}")

        except Exception as e:
            print(f"❌ 検証エラー: {e}")

    print(f"\n=== 強化データ品質システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_enhanced_data_quality())