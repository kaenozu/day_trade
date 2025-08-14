#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Condition Monitor - 市場状況監視システム

Issue #798実装：ライブ環境での実地テスト
リアルタイム市場分析と予測システム性能監視
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path

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

class MarketCondition(Enum):
    """市場状況"""
    BULL_STRONG = "強気市場"
    BULL_MODERATE = "穏健な上昇市場"
    NEUTRAL = "中立市場"
    BEAR_MODERATE = "穏健な下降市場"
    BEAR_STRONG = "弱気市場"
    HIGH_VOLATILITY = "高ボラティリティ"

@dataclass
class MarketMetrics:
    """市場メトリクス"""
    condition: MarketCondition
    volatility_index: float
    trend_strength: float
    momentum_score: float
    fear_greed_index: float
    prediction_accuracy: float
    timestamp: datetime
    supporting_indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictionPerformance:
    """予測性能追跡"""
    symbol: str
    prediction: int
    actual_direction: Optional[int]
    confidence: float
    price_at_prediction: float
    price_after_1h: Optional[float]
    price_after_4h: Optional[float]
    price_after_1d: Optional[float]
    accuracy_1h: Optional[bool]
    accuracy_4h: Optional[bool]
    accuracy_1d: Optional[bool]
    timestamp: datetime

class MarketConditionMonitor:
    """市場状況監視システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース設定
        self.db_path = Path("trading_data/market_monitor.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # 監視対象銘柄（代表的な指標）
        self.market_indices = {
            "7203": "トヨタ自動車（大型株代表）",
            "8306": "三菱UFJ（金融セクター）",
            "4751": "サイバーエージェント（グロース株）",
            "6861": "キーエンス（技術株）",
            "9984": "ソフトバンクG（投機株）"
        }

        # 予測性能追跡
        self.predictions: List[PredictionPerformance] = []
        self.market_history: List[MarketMetrics] = []

        # アラート閾値
        self.volatility_alert_threshold = 30.0
        self.accuracy_alert_threshold = 0.45

        self._init_database()
        self.logger.info("Market condition monitor initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 市場メトリクステーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        volatility_index REAL NOT NULL,
                        trend_strength REAL NOT NULL,
                        momentum_score REAL NOT NULL,
                        fear_greed_index REAL NOT NULL,
                        prediction_accuracy REAL NOT NULL,
                        supporting_indicators TEXT
                    )
                ''')

                # 予測性能テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        prediction INTEGER NOT NULL,
                        actual_direction INTEGER,
                        confidence REAL NOT NULL,
                        price_at_prediction REAL NOT NULL,
                        price_after_1h REAL,
                        price_after_4h REAL,
                        price_after_1d REAL,
                        accuracy_1h INTEGER,
                        accuracy_4h INTEGER,
                        accuracy_1d INTEGER,
                        timestamp TEXT NOT NULL
                    )
                ''')

                # アラート履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def analyze_market_condition(self) -> MarketMetrics:
        """市場状況分析"""

        try:
            # 各銘柄のデータ収集と分析
            symbol_analyses = {}

            for symbol, description in self.market_indices.items():
                try:
                    analysis = await self._analyze_symbol_condition(symbol)
                    symbol_analyses[symbol] = analysis
                except Exception as e:
                    self.logger.warning(f"銘柄分析エラー {symbol}: {e}")
                    continue

            if not symbol_analyses:
                raise Exception("市場データ取得失敗")

            # 総合市場指標計算
            volatility_scores = [a['volatility'] for a in symbol_analyses.values()]
            trend_scores = [a['trend_strength'] for a in symbol_analyses.values()]
            momentum_scores = [a['momentum'] for a in symbol_analyses.values()]

            avg_volatility = np.mean(volatility_scores)
            avg_trend = np.mean(trend_scores)
            avg_momentum = np.mean(momentum_scores)

            # 恐怖貪欲指数（簡易版）
            fear_greed = self._calculate_fear_greed_index(symbol_analyses)

            # 予測精度取得
            current_accuracy = await self._get_current_prediction_accuracy()

            # 市場状況判定
            market_condition = self._determine_market_condition(
                avg_volatility, avg_trend, avg_momentum, fear_greed
            )

            # メトリクス作成
            metrics = MarketMetrics(
                condition=market_condition,
                volatility_index=avg_volatility,
                trend_strength=avg_trend,
                momentum_score=avg_momentum,
                fear_greed_index=fear_greed,
                prediction_accuracy=current_accuracy,
                timestamp=datetime.now(),
                supporting_indicators={
                    'symbol_count': len(symbol_analyses),
                    'volatility_range': max(volatility_scores) - min(volatility_scores),
                    'trend_consistency': np.std(trend_scores),
                    'momentum_strength': max(momentum_scores)
                }
            )

            # 履歴に追加
            self.market_history.append(metrics)

            # データベース保存
            await self._save_market_metrics(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"市場分析エラー: {e}")
            # デフォルトメトリクス返却
            return MarketMetrics(
                condition=MarketCondition.NEUTRAL,
                volatility_index=20.0,
                trend_strength=0.0,
                momentum_score=0.0,
                fear_greed_index=50.0,
                prediction_accuracy=0.5,
                timestamp=datetime.now()
            )

    async def _analyze_symbol_condition(self, symbol: str) -> Dict[str, float]:
        """個別銘柄状況分析"""

        # 過去30日のデータ取得
        from real_data_provider_v2 import real_data_provider
        data = await real_data_provider.get_stock_data(symbol, "1mo")

        if data is None or len(data) < 10:
            raise Exception(f"データ不足: {symbol}")

        # 価格データ
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # ボラティリティ計算（20日）
        returns = close.pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100  # 年率％

        # トレンド強度（20日移動平均との乖離）
        sma_20 = close.rolling(20).mean()
        trend_strength = ((close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100

        # モメンタム（10日ROC）
        momentum = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # ボリューム分析
        avg_volume = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1]

        return {
            'volatility': volatility if not np.isnan(volatility) else 20.0,
            'trend_strength': trend_strength if not np.isnan(trend_strength) else 0.0,
            'momentum': momentum if not np.isnan(momentum) else 0.0,
            'rsi': rsi if not np.isnan(rsi) else 50.0,
            'volume_ratio': volume_ratio if not np.isnan(volume_ratio) else 1.0
        }

    def _calculate_fear_greed_index(self, symbol_analyses: Dict[str, Dict[str, float]]) -> float:
        """恐怖貪欲指数計算（簡易版）"""

        # RSI平均（過熱度）
        avg_rsi = np.mean([a['rsi'] for a in symbol_analyses.values()])
        rsi_score = (avg_rsi - 30) / 40 * 50  # 30-70を0-50にマップ

        # ボラティリティ（恐怖度）
        avg_vol = np.mean([a['volatility'] for a in symbol_analyses.values()])
        vol_score = max(0, 50 - (avg_vol - 15) * 2)  # 高ボラティリティは恐怖

        # モメンタム（トレンド強度）
        avg_momentum = np.mean([a['momentum'] for a in symbol_analyses.values()])
        momentum_score = max(0, min(50, avg_momentum * 2 + 25))

        # 総合スコア
        fear_greed = (rsi_score + vol_score + momentum_score) / 3
        return max(0, min(100, fear_greed))

    def _determine_market_condition(self, volatility: float, trend: float,
                                  momentum: float, fear_greed: float) -> MarketCondition:
        """市場状況判定"""

        # 高ボラティリティ判定
        if volatility > 35:
            return MarketCondition.HIGH_VOLATILITY

        # トレンドベース判定
        if trend > 5 and momentum > 3 and fear_greed > 70:
            return MarketCondition.BULL_STRONG
        elif trend > 2 and momentum > 1 and fear_greed > 55:
            return MarketCondition.BULL_MODERATE
        elif trend < -5 and momentum < -3 and fear_greed < 30:
            return MarketCondition.BEAR_STRONG
        elif trend < -2 and momentum < -1 and fear_greed < 45:
            return MarketCondition.BEAR_MODERATE
        else:
            return MarketCondition.NEUTRAL

    async def _get_current_prediction_accuracy(self) -> float:
        """現在の予測精度取得"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 過去24時間の予測精度
                yesterday = (datetime.now() - timedelta(days=1)).isoformat()

                cursor.execute('''
                    SELECT AVG(CASE WHEN accuracy_1h = 1 THEN 1.0 ELSE 0.0 END) as accuracy
                    FROM prediction_performance
                    WHERE timestamp > ? AND accuracy_1h IS NOT NULL
                ''', (yesterday,))

                result = cursor.fetchone()
                if result and result[0] is not None:
                    return float(result[0])
                else:
                    return 0.5  # デフォルト値

        except Exception as e:
            self.logger.warning(f"精度取得エラー: {e}")
            return 0.5

    async def track_prediction_performance(self, symbol: str, prediction: int,
                                         confidence: float, current_price: float):
        """予測性能追跡開始"""

        performance = PredictionPerformance(
            symbol=symbol,
            prediction=prediction,
            actual_direction=None,
            confidence=confidence,
            price_at_prediction=current_price,
            price_after_1h=None,
            price_after_4h=None,
            price_after_1d=None,
            accuracy_1h=None,
            accuracy_4h=None,
            accuracy_1d=None,
            timestamp=datetime.now()
        )

        self.predictions.append(performance)

        # 後続チェックをスケジュール
        asyncio.create_task(self._check_prediction_accuracy(performance))

    async def _check_prediction_accuracy(self, performance: PredictionPerformance):
        """予測精度チェック（時間経過後）"""

        try:
            # 1時間後チェック
            await asyncio.sleep(3600)  # 1時間待機
            performance.price_after_1h = await self._get_current_price(performance.symbol)
            if performance.price_after_1h:
                actual_direction_1h = 1 if performance.price_after_1h > performance.price_at_prediction else 0
                performance.accuracy_1h = (performance.prediction == actual_direction_1h)

            # 4時間後チェック
            await asyncio.sleep(10800)  # 3時間追加待機
            performance.price_after_4h = await self._get_current_price(performance.symbol)
            if performance.price_after_4h:
                actual_direction_4h = 1 if performance.price_after_4h > performance.price_at_prediction else 0
                performance.accuracy_4h = (performance.prediction == actual_direction_4h)

            # 1日後チェック
            await asyncio.sleep(72000)  # 20時間追加待機
            performance.price_after_1d = await self._get_current_price(performance.symbol)
            if performance.price_after_1d:
                actual_direction_1d = 1 if performance.price_after_1d > performance.price_at_prediction else 0
                performance.accuracy_1d = (performance.prediction == actual_direction_1d)
                performance.actual_direction = actual_direction_1d

            # データベース保存
            await self._save_prediction_performance(performance)

        except Exception as e:
            self.logger.error(f"予測精度チェックエラー: {e}")

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """現在価格取得"""

        try:
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_latest_stock_price(symbol)
            return data.get('current_price') if data else None
        except Exception as e:
            self.logger.warning(f"価格取得エラー {symbol}: {e}")
            return None

    async def _save_market_metrics(self, metrics: MarketMetrics):
        """市場メトリクス保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO market_metrics
                    (timestamp, condition, volatility_index, trend_strength,
                     momentum_score, fear_greed_index, prediction_accuracy, supporting_indicators)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp.isoformat(),
                    metrics.condition.value,
                    metrics.volatility_index,
                    metrics.trend_strength,
                    metrics.momentum_score,
                    metrics.fear_greed_index,
                    metrics.prediction_accuracy,
                    json.dumps(metrics.supporting_indicators)
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"メトリクス保存エラー: {e}")

    async def _save_prediction_performance(self, performance: PredictionPerformance):
        """予測性能保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_performance
                    (symbol, prediction, actual_direction, confidence, price_at_prediction,
                     price_after_1h, price_after_4h, price_after_1d,
                     accuracy_1h, accuracy_4h, accuracy_1d, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.symbol,
                    performance.prediction,
                    performance.actual_direction,
                    performance.confidence,
                    performance.price_at_prediction,
                    performance.price_after_1h,
                    performance.price_after_4h,
                    performance.price_after_1d,
                    int(performance.accuracy_1h) if performance.accuracy_1h is not None else None,
                    int(performance.accuracy_4h) if performance.accuracy_4h is not None else None,
                    int(performance.accuracy_1d) if performance.accuracy_1d is not None else None,
                    performance.timestamp.isoformat()
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"予測性能保存エラー: {e}")

    async def generate_market_report(self) -> Dict[str, Any]:
        """市場レポート生成"""

        try:
            # 最新市場分析
            current_metrics = await self.analyze_market_condition()

            # 予測性能統計
            accuracy_stats = await self._calculate_accuracy_statistics()

            # アラート検出
            alerts = await self._detect_market_alerts(current_metrics)

            # レポート作成
            report = {
                "timestamp": datetime.now().isoformat(),
                "market_condition": current_metrics.condition.value,
                "metrics": {
                    "volatility_index": current_metrics.volatility_index,
                    "trend_strength": current_metrics.trend_strength,
                    "momentum_score": current_metrics.momentum_score,
                    "fear_greed_index": current_metrics.fear_greed_index,
                    "prediction_accuracy": current_metrics.prediction_accuracy
                },
                "accuracy_stats": accuracy_stats,
                "alerts": alerts,
                "supporting_data": current_metrics.supporting_indicators
            }

            return report

        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")
            return {"error": str(e)}

    async def _calculate_accuracy_statistics(self) -> Dict[str, float]:
        """精度統計計算"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 過去7日間の統計
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()

                cursor.execute('''
                    SELECT
                        AVG(CASE WHEN accuracy_1h = 1 THEN 1.0 ELSE 0.0 END) as accuracy_1h,
                        AVG(CASE WHEN accuracy_4h = 1 THEN 1.0 ELSE 0.0 END) as accuracy_4h,
                        AVG(CASE WHEN accuracy_1d = 1 THEN 1.0 ELSE 0.0 END) as accuracy_1d,
                        COUNT(*) as total_predictions
                    FROM prediction_performance
                    WHERE timestamp > ?
                ''', (week_ago,))

                result = cursor.fetchone()

                if result:
                    return {
                        "accuracy_1h": result[0] or 0.5,
                        "accuracy_4h": result[1] or 0.5,
                        "accuracy_1d": result[2] or 0.5,
                        "total_predictions": result[3] or 0
                    }
                else:
                    return {
                        "accuracy_1h": 0.5,
                        "accuracy_4h": 0.5,
                        "accuracy_1d": 0.5,
                        "total_predictions": 0
                    }

        except Exception as e:
            self.logger.warning(f"統計計算エラー: {e}")
            return {
                "accuracy_1h": 0.5,
                "accuracy_4h": 0.5,
                "accuracy_1d": 0.5,
                "total_predictions": 0
            }

    async def _detect_market_alerts(self, metrics: MarketMetrics) -> List[Dict[str, str]]:
        """市場アラート検出"""

        alerts = []

        # 高ボラティリティアラート
        if metrics.volatility_index > self.volatility_alert_threshold:
            alerts.append({
                "type": "HIGH_VOLATILITY",
                "severity": "WARNING",
                "message": f"高ボラティリティ検出: {metrics.volatility_index:.1f}%"
            })

        # 予測精度低下アラート
        if metrics.prediction_accuracy < self.accuracy_alert_threshold:
            alerts.append({
                "type": "LOW_ACCURACY",
                "severity": "ERROR",
                "message": f"予測精度低下: {metrics.prediction_accuracy:.1%}"
            })

        # 極端な市場状況アラート
        if metrics.condition in [MarketCondition.BULL_STRONG, MarketCondition.BEAR_STRONG]:
            alerts.append({
                "type": "EXTREME_MARKET",
                "severity": "INFO",
                "message": f"極端な市場状況: {metrics.condition.value}"
            })

        # 恐怖貪欲指数異常
        if metrics.fear_greed_index < 20 or metrics.fear_greed_index > 80:
            emotion = "極度の恐怖" if metrics.fear_greed_index < 20 else "極度の貪欲"
            alerts.append({
                "type": "EXTREME_EMOTION",
                "severity": "WARNING",
                "message": f"市場感情異常: {emotion} (指数: {metrics.fear_greed_index:.0f})"
            })

        return alerts

    def display_live_dashboard(self, metrics: MarketMetrics, accuracy_stats: Dict[str, float]):
        """ライブダッシュボード表示"""

        print(f"\n=== 📊 市場状況リアルタイム監視 ===")
        print(f"時刻: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # 市場状況
        condition_emoji = {
            MarketCondition.BULL_STRONG: "🚀",
            MarketCondition.BULL_MODERATE: "📈",
            MarketCondition.NEUTRAL: "➡️",
            MarketCondition.BEAR_MODERATE: "📉",
            MarketCondition.BEAR_STRONG: "💥",
            MarketCondition.HIGH_VOLATILITY: "⚡"
        }

        print(f"\n🏪 市場状況: {condition_emoji.get(metrics.condition, '❓')} {metrics.condition.value}")

        # 主要指標
        print(f"\n📊 主要指標:")
        print(f"  ボラティリティ指数: {metrics.volatility_index:.1f}%")
        print(f"  トレンド強度: {metrics.trend_strength:.1f}%")
        print(f"  モメンタムスコア: {metrics.momentum_score:.1f}")
        print(f"  恐怖貪欲指数: {metrics.fear_greed_index:.0f}/100")

        # 予測性能
        print(f"\n🎯 予測性能:")
        print(f"  現在精度: {metrics.prediction_accuracy:.1%}")
        print(f"  1時間後精度: {accuracy_stats['accuracy_1h']:.1%}")
        print(f"  4時間後精度: {accuracy_stats['accuracy_4h']:.1%}")
        print(f"  1日後精度: {accuracy_stats['accuracy_1d']:.1%}")
        print(f"  総予測数: {accuracy_stats['total_predictions']}")

        # 技術的詳細
        if metrics.supporting_indicators:
            print(f"\n🔧 技術的詳細:")
            for key, value in metrics.supporting_indicators.items():
                print(f"  {key}: {value:.2f}")

# グローバルインスタンス
market_condition_monitor = MarketConditionMonitor()

# テスト実行
async def run_market_monitoring_test():
    """市場監視テスト実行"""

    print("=== 🔍 市場状況監視システムテスト ===")

    # 市場レポート生成
    report = await market_condition_monitor.generate_market_report()

    if "error" not in report:
        print(f"\n📊 市場レポート生成成功")
        print(f"市場状況: {report['market_condition']}")
        print(f"ボラティリティ: {report['metrics']['volatility_index']:.1f}%")
        print(f"予測精度: {report['metrics']['prediction_accuracy']:.1%}")

        if report['alerts']:
            print(f"\n⚠️ アラート: {len(report['alerts'])}件")
            for alert in report['alerts']:
                print(f"  {alert['severity']}: {alert['message']}")
    else:
        print(f"❌ エラー: {report['error']}")

    return report

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_market_monitoring_test())