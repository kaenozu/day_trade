#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Monitor - データ品質検証システム

実データの品質保証・信頼性確保・異常検知システム
Phase5-A #902実装：データ信頼性確保の必須機能
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from collections import defaultdict, deque
import statistics
from scipy import stats
import aiohttp

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
    from real_data_provider_v2 import real_data_provider, DataSource
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False

class QualityStatus(Enum):
    """品質ステータス"""
    EXCELLENT = "優秀"      # 95%以上
    GOOD = "良好"          # 85-94%
    FAIR = "普通"          # 70-84%
    POOR = "低品質"        # 50-69%
    CRITICAL = "危険"      # 50%未満

class AnomalyType(Enum):
    """異常タイプ"""
    PRICE_SPIKE = "価格急騰・急落"
    VOLUME_ANOMALY = "出来高異常"
    DATA_MISSING = "データ欠損"
    OHLC_INCONSISTENCY = "OHLC整合性エラー"
    STALE_DATA = "古いデータ"
    SOURCE_FAILURE = "データソース障害"

@dataclass
class QualityMetric:
    """品質指標"""
    metric_name: str
    value: float           # 0-100の品質スコア
    status: QualityStatus
    last_updated: datetime
    trend: str            # UP, DOWN, STABLE
    threshold: float      # 警告閾値
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataAnomaly:
    """データ異常"""
    anomaly_id: str
    symbol: str
    anomaly_type: AnomalyType
    severity: str         # CRITICAL, HIGH, MEDIUM, LOW
    detected_at: datetime
    description: str
    data_point: Dict[str, Any]
    confidence: float     # 異常検知信頼度
    suggested_action: str
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class ValidationResult:
    """検証結果"""
    symbol: str
    timestamp: datetime
    is_valid: bool
    quality_score: float
    anomalies: List[DataAnomaly]
    data_completeness: float    # データ完整性
    price_consistency: float    # 価格整合性
    temporal_consistency: float # 時系列整合性
    source_reliability: float   # ソース信頼性

class DataQualityMonitor:
    """データ品質監視システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("quality_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "quality.db"
        self._init_database()

        # 品質ルール設定
        self.quality_rules = {
            'price_change_limit': 30.0,    # 前日比30%以上変動で警告
            'volume_spike_threshold': 10.0, # 平均出来高の10倍で異常
            'missing_data_tolerance': 5.0,  # 5%以上欠損で警告
            'latency_threshold': 900,       # 15分以上遅延で警告
            'ohlc_consistency_tolerance': 0.1, # OHLC関係の許容誤差
            'stale_data_threshold': 3600    # 1時間以上古いデータで警告
        }

        # 監視統計
        self.validation_history: deque = deque(maxlen=1000)
        self.anomaly_history: deque = deque(maxlen=500)
        self.quality_metrics: Dict[str, QualityMetric] = {}

        # リアルタイム統計
        self.real_time_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'anomalies_detected': 0,
            'avg_quality_score': 0.0,
            'last_validation': None
        }

        self.logger.info("Data quality monitor initialized")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 品質メトリクステーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    quality_score REAL,
                    completeness REAL,
                    consistency REAL,
                    reliability REAL,
                    data_source TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 異常ログテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_log (
                    anomaly_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    anomaly_type TEXT,
                    severity TEXT,
                    detected_at TEXT,
                    description TEXT,
                    confidence REAL,
                    data_point TEXT,
                    is_resolved BOOLEAN,
                    resolved_at TEXT
                )
            """)

            # インデックス作成
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_symbol ON quality_metrics(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_timestamp ON quality_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_symbol ON anomaly_log(symbol)")

    async def validate_stock_data(self, symbol: str, data: pd.DataFrame,
                                 source: DataSource) -> ValidationResult:
        """株価データ品質検証"""

        timestamp = datetime.now()
        anomalies = []

        try:
            # 基本データ検証
            data_completeness = self._check_data_completeness(data)
            price_consistency = self._check_price_consistency(data)
            temporal_consistency = self._check_temporal_consistency(data)

            # 異常検知
            price_anomalies = self._detect_price_anomalies(symbol, data)
            volume_anomalies = self._detect_volume_anomalies(symbol, data)

            anomalies.extend(price_anomalies)
            anomalies.extend(volume_anomalies)

            # 総合品質スコア計算
            quality_score = self._calculate_quality_score(
                data_completeness, price_consistency, temporal_consistency, len(anomalies)
            )

            # ソース信頼性評価
            source_reliability = self._evaluate_source_reliability(source, quality_score)

            # 検証結果作成
            result = ValidationResult(
                symbol=symbol,
                timestamp=timestamp,
                is_valid=quality_score >= 70.0,  # 70点以上で有効
                quality_score=quality_score,
                anomalies=anomalies,
                data_completeness=data_completeness,
                price_consistency=price_consistency,
                temporal_consistency=temporal_consistency,
                source_reliability=source_reliability
            )

            # データベース保存
            await self._save_validation_result(result, source)

            # 統計更新
            self._update_statistics(result)

            return result

        except Exception as e:
            self.logger.error(f"Validation failed for {symbol}: {e}")

            # エラー時のデフォルト結果
            return ValidationResult(
                symbol=symbol,
                timestamp=timestamp,
                is_valid=False,
                quality_score=0.0,
                anomalies=[],
                data_completeness=0.0,
                price_consistency=0.0,
                temporal_consistency=0.0,
                source_reliability=0.0
            )

    def _check_data_completeness(self, data: pd.DataFrame) -> float:
        """データ完整性チェック"""

        if data.empty:
            return 0.0

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            return 0.0  # 必須カラムが欠損している場合は0点

        # NaN値の割合を計算
        total_cells = len(data) * len(required_columns)
        missing_cells = data[required_columns].isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100

        return completeness

    def _check_price_consistency(self, data: pd.DataFrame) -> float:
        """価格整合性チェック"""

        if data.empty or len(data) < 2:
            return 50.0  # データ不足の場合は中立スコア

        inconsistencies = 0
        total_checks = 0

        for idx in data.index:
            try:
                high = data.loc[idx, 'High']
                low = data.loc[idx, 'Low']
                open_price = data.loc[idx, 'Open']
                close_price = data.loc[idx, 'Close']

                total_checks += 4

                # OHLC関係の整合性チェック
                if high < low:
                    inconsistencies += 1
                if high < open_price:
                    inconsistencies += 1
                if high < close_price:
                    inconsistencies += 1
                if low > open_price:
                    inconsistencies += 1
                if low > close_price:
                    inconsistencies += 1

                # 異常な価格値チェック
                if any(price <= 0 for price in [high, low, open_price, close_price]):
                    inconsistencies += 1

            except (KeyError, TypeError, ValueError):
                inconsistencies += 1

        if total_checks == 0:
            return 50.0

        consistency = (total_checks - inconsistencies) / total_checks * 100
        return consistency

    def _check_temporal_consistency(self, data: pd.DataFrame) -> float:
        """時系列整合性チェック"""

        if data.empty or len(data) < 5:
            return 50.0

        # インデックスが日付順になっているかチェック
        try:
            dates = pd.to_datetime(data.index)
            is_sorted = dates.is_monotonic_increasing

            # データ間隔の一貫性チェック
            if len(dates) > 1:
                intervals = dates[1:] - dates[:-1]
                # 1日間隔が標準（土日祝除く）
                expected_interval = pd.Timedelta(days=1)

                # 間隔の変動係数を計算
                interval_days = intervals.days
                if len(interval_days) > 1:
                    cv = statistics.stdev(interval_days) / statistics.mean(interval_days)
                    interval_consistency = max(0, 100 - cv * 50)  # 変動係数に基づくスコア
                else:
                    interval_consistency = 100
            else:
                interval_consistency = 100

            # 総合時系列スコア
            temporal_score = (90 if is_sorted else 30) * 0.5 + interval_consistency * 0.5
            return temporal_score

        except Exception as e:
            self.logger.warning(f"Temporal consistency check failed: {e}")
            return 30.0

    def _detect_price_anomalies(self, symbol: str, data: pd.DataFrame) -> List[DataAnomaly]:
        """価格異常検知"""

        anomalies = []

        if data.empty or len(data) < 2:
            return anomalies

        try:
            # 日次変動率計算
            returns = data['Close'].pct_change().dropna()

            if len(returns) == 0:
                return anomalies

            # 異常な価格変動検知（Z-scoreベース）
            if len(returns) >= 5:
                z_scores = np.abs(stats.zscore(returns))

                for i, (date, z_score) in enumerate(zip(returns.index, z_scores)):
                    if z_score > 3:  # 3σを超える異常値
                        change_pct = returns[date] * 100

                        anomaly = DataAnomaly(
                            anomaly_id=f"PRICE_SPIKE_{symbol}_{date.strftime('%Y%m%d')}",
                            symbol=symbol,
                            anomaly_type=AnomalyType.PRICE_SPIKE,
                            severity="HIGH" if abs(change_pct) > 15 else "MEDIUM",
                            detected_at=datetime.now(),
                            description=f"異常な価格変動: {change_pct:+.1f}%",
                            data_point={
                                "date": date.isoformat(),
                                "change_percent": change_pct,
                                "z_score": z_score,
                                "price": float(data.loc[date, 'Close'])
                            },
                            confidence=min(95, z_score * 20),
                            suggested_action="価格データの再確認・他ソースとの照合"
                        )

                        anomalies.append(anomaly)

            # 前日比での異常検知
            if len(data) >= 2:
                latest_change = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100

                if abs(latest_change) > self.quality_rules['price_change_limit']:
                    anomaly = DataAnomaly(
                        anomaly_id=f"EXTREME_CHANGE_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        symbol=symbol,
                        anomaly_type=AnomalyType.PRICE_SPIKE,
                        severity="CRITICAL" if abs(latest_change) > 50 else "HIGH",
                        detected_at=datetime.now(),
                        description=f"極端な価格変動: {latest_change:+.1f}%",
                        data_point={
                            "change_percent": latest_change,
                            "current_price": float(data['Close'].iloc[-1]),
                            "previous_price": float(data['Close'].iloc[-2])
                        },
                        confidence=90.0,
                        suggested_action="緊急確認・取引停止検討"
                    )

                    anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Price anomaly detection failed for {symbol}: {e}")

        return anomalies

    def _detect_volume_anomalies(self, symbol: str, data: pd.DataFrame) -> List[DataAnomaly]:
        """出来高異常検知"""

        anomalies = []

        if data.empty or 'Volume' not in data.columns or len(data) < 10:
            return anomalies

        try:
            volumes = data['Volume'].dropna()

            if len(volumes) == 0:
                return anomalies

            # 平均出来高計算（過去20日）
            avg_volume = volumes.rolling(min(20, len(volumes))).mean()

            if len(avg_volume) > 0 and avg_volume.iloc[-1] > 0:
                latest_volume = volumes.iloc[-1]
                volume_ratio = latest_volume / avg_volume.iloc[-1]

                if volume_ratio > self.quality_rules['volume_spike_threshold']:
                    anomaly = DataAnomaly(
                        anomaly_id=f"VOLUME_SPIKE_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        symbol=symbol,
                        anomaly_type=AnomalyType.VOLUME_ANOMALY,
                        severity="HIGH" if volume_ratio > 20 else "MEDIUM",
                        detected_at=datetime.now(),
                        description=f"出来高急増: 平均の{volume_ratio:.1f}倍",
                        data_point={
                            "current_volume": int(latest_volume),
                            "average_volume": int(avg_volume.iloc[-1]),
                            "volume_ratio": volume_ratio
                        },
                        confidence=85.0,
                        suggested_action="材料確認・ニュースチェック"
                    )

                    anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Volume anomaly detection failed for {symbol}: {e}")

        return anomalies

    def _calculate_quality_score(self, completeness: float, price_consistency: float,
                                temporal_consistency: float, anomaly_count: int) -> float:
        """総合品質スコア計算"""

        # 基本品質スコア（重み付き平均）
        base_score = (
            completeness * 0.3 +
            price_consistency * 0.4 +
            temporal_consistency * 0.3
        )

        # 異常検知によるペナルティ
        anomaly_penalty = min(30, anomaly_count * 10)  # 異常1件につき10点減点、最大30点

        final_score = max(0, base_score - anomaly_penalty)

        return final_score

    def _evaluate_source_reliability(self, source: DataSource, quality_score: float) -> float:
        """データソース信頼性評価"""

        # ベース信頼度（ソース別）
        base_reliability = {
            DataSource.YAHOO_FINANCE: 80.0,
            DataSource.STOOQ: 70.0,
            DataSource.ALPHA_VANTAGE: 85.0,
            DataSource.MATSUI_SECURITIES: 95.0,
            DataSource.GMO_CLICK: 90.0
        }.get(source, 50.0)

        # 品質スコアによる調整
        quality_factor = quality_score / 100
        adjusted_reliability = base_reliability * quality_factor

        return adjusted_reliability

    async def _save_validation_result(self, result: ValidationResult, source: DataSource):
        """検証結果のデータベース保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 品質メトリクス保存
                conn.execute("""
                    INSERT INTO quality_metrics
                    (symbol, timestamp, quality_score, completeness, consistency, reliability, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.symbol,
                    result.timestamp.isoformat(),
                    result.quality_score,
                    result.data_completeness,
                    result.price_consistency,
                    result.source_reliability,
                    source.value
                ))

                # 異常ログ保存
                for anomaly in result.anomalies:
                    conn.execute("""
                        INSERT OR REPLACE INTO anomaly_log
                        (anomaly_id, symbol, anomaly_type, severity, detected_at,
                         description, confidence, data_point, is_resolved)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        anomaly.anomaly_id,
                        anomaly.symbol,
                        anomaly.anomaly_type.value,
                        anomaly.severity,
                        anomaly.detected_at.isoformat(),
                        anomaly.description,
                        anomaly.confidence,
                        json.dumps(anomaly.data_point),
                        anomaly.is_resolved
                    ))

        except Exception as e:
            self.logger.error(f"Failed to save validation result: {e}")

    def _update_statistics(self, result: ValidationResult):
        """統計情報更新"""

        self.real_time_stats['total_validations'] += 1

        if result.is_valid:
            self.real_time_stats['successful_validations'] += 1

        self.real_time_stats['anomalies_detected'] += len(result.anomalies)

        # 移動平均で平均品質スコア更新
        total = self.real_time_stats['total_validations']
        current_avg = self.real_time_stats['avg_quality_score']
        new_avg = (current_avg * (total - 1) + result.quality_score) / total
        self.real_time_stats['avg_quality_score'] = new_avg

        self.real_time_stats['last_validation'] = datetime.now()

        # 履歴追加
        self.validation_history.append(result)
        self.anomaly_history.extend(result.anomalies)

    def get_quality_statistics(self) -> Dict[str, Any]:
        """品質統計取得"""

        return {
            'real_time_stats': self.real_time_stats,
            'validation_success_rate': (
                self.real_time_stats['successful_validations'] /
                max(1, self.real_time_stats['total_validations']) * 100
            ),
            'recent_anomalies': len([
                a for a in self.anomaly_history
                if a.detected_at > datetime.now() - timedelta(hours=24)
            ]),
            'quality_trend': self._calculate_quality_trend(),
            'active_anomalies': len([a for a in self.anomaly_history if not a.is_resolved]),
        }

    def _calculate_quality_trend(self) -> str:
        """品質トレンド計算"""

        if len(self.validation_history) < 10:
            return "INSUFFICIENT_DATA"

        recent_scores = [r.quality_score for r in list(self.validation_history)[-10:]]
        older_scores = [r.quality_score for r in list(self.validation_history)[-20:-10]] if len(self.validation_history) >= 20 else []

        if not older_scores:
            return "STABLE"

        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)

        change = recent_avg - older_avg

        if change > 5:
            return "IMPROVING"
        elif change < -5:
            return "DEGRADING"
        else:
            return "STABLE"

    async def generate_quality_report(self) -> Dict[str, Any]:
        """品質レポート生成"""

        try:
            stats = self.get_quality_statistics()

            with sqlite3.connect(self.db_path) as conn:
                # 過去24時間の統計
                yesterday = datetime.now() - timedelta(hours=24)

                cursor = conn.execute("""
                    SELECT AVG(quality_score), COUNT(*), data_source
                    FROM quality_metrics
                    WHERE timestamp >= ?
                    GROUP BY data_source
                """, (yesterday.isoformat(),))

                source_stats = {}
                for avg_score, count, source in cursor.fetchall():
                    source_stats[source] = {
                        'avg_quality': avg_score,
                        'validation_count': count
                    }

                # 異常統計
                cursor = conn.execute("""
                    SELECT anomaly_type, severity, COUNT(*)
                    FROM anomaly_log
                    WHERE detected_at >= ? AND is_resolved = 0
                    GROUP BY anomaly_type, severity
                """, (yesterday.isoformat(),))

                anomaly_stats = {}
                for anomaly_type, severity, count in cursor.fetchall():
                    key = f"{anomaly_type}_{severity}"
                    anomaly_stats[key] = count

            return {
                'summary': stats,
                'source_performance': source_stats,
                'anomaly_breakdown': anomaly_stats,
                'recommendations': self._generate_quality_recommendations(stats),
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")
            return {'error': f"Report generation failed: {e}"}

    def _generate_quality_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """品質改善推奨事項生成"""

        recommendations = []

        success_rate = stats['validation_success_rate']
        avg_quality = stats['real_time_stats']['avg_quality_score']
        trend = stats['quality_trend']

        if success_rate < 90:
            recommendations.append(f"検証成功率{success_rate:.1f}%が低い状態です。データソースの見直しを推奨")

        if avg_quality < 70:
            recommendations.append(f"平均品質スコア{avg_quality:.1f}が低い状態です。品質閾値の調整を検討")

        if trend == "DEGRADING":
            recommendations.append("品質トレンドが悪化しています。システムの点検を推奨")

        if stats['active_anomalies'] > 10:
            recommendations.append(f"未解決異常が{stats['active_anomalies']}件あります。異常対応の強化を推奨")

        if not recommendations:
            recommendations.append("データ品質は良好な状態です。現在の監視体制を継続してください")

        return recommendations

# グローバルインスタンス
quality_monitor = DataQualityMonitor()

# テスト関数
async def test_data_quality_monitor():
    """データ品質監視システムのテスト"""

    print("=== データ品質監視システム テスト ===")

    monitor = DataQualityMonitor()

    if not REAL_DATA_PROVIDER_AVAILABLE:
        print("実データプロバイダーが利用できません")
        return

    # テスト銘柄
    test_symbols = ["7203", "4751", "6861"]

    print(f"\n[ {len(test_symbols)}銘柄の品質検証テスト ]")

    validation_results = []

    for symbol in test_symbols:
        print(f"\n--- {symbol} 品質検証 ---")

        # データ取得
        data = await real_data_provider.get_stock_data(symbol, "1mo")

        if data is not None:
            # 品質検証実行
            result = await monitor.validate_stock_data(symbol, data, DataSource.YAHOO_FINANCE)
            validation_results.append(result)

            print(f"総合品質スコア: {result.quality_score:.1f}/100")
            print(f"データ完整性: {result.data_completeness:.1f}%")
            print(f"価格整合性: {result.price_consistency:.1f}%")
            print(f"時系列整合性: {result.temporal_consistency:.1f}%")
            print(f"ソース信頼性: {result.source_reliability:.1f}%")
            print(f"検証結果: {'✅ 有効' if result.is_valid else '❌ 無効'}")

            if result.anomalies:
                print(f"検出異常: {len(result.anomalies)}件")
                for anomaly in result.anomalies[:3]:  # 最大3件表示
                    print(f"  • {anomaly.anomaly_type.value}: {anomaly.description}")
            else:
                print("異常検出: なし")
        else:
            print("❌ データ取得に失敗しました")

    # 統計情報表示
    print(f"\n[ 品質統計 ]")
    stats = monitor.get_quality_statistics()

    print(f"総検証回数: {stats['real_time_stats']['total_validations']}")
    print(f"検証成功率: {stats['validation_success_rate']:.1f}%")
    print(f"平均品質スコア: {stats['real_time_stats']['avg_quality_score']:.1f}")
    print(f"品質トレンド: {stats['quality_trend']}")
    print(f"アクティブ異常: {stats['active_anomalies']}件")

    # 品質レポート生成
    print(f"\n[ 品質レポート生成 ]")
    report = await monitor.generate_quality_report()

    if 'error' not in report:
        print("レポート生成成功")

        recommendations = report.get('recommendations', [])
        if recommendations:
            print("改善提案:")
            for rec in recommendations:
                print(f"  • {rec}")
    else:
        print(f"レポート生成エラー: {report['error']}")

    print(f"\n=== データ品質監視システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_data_quality_monitor())