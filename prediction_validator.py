#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Validator - 予測精度検証システム

デイトレード予測の精度検証・改善システム
バックテスト・リアルタイム検証・精度向上アルゴリズム統合
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import statistics
from collections import defaultdict, deque

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
    from enhanced_symbol_manager import EnhancedSymbolManager
    ENHANCED_SYMBOLS_AVAILABLE = True
except ImportError:
    ENHANCED_SYMBOLS_AVAILABLE = False

class PredictionResult(Enum):
    """予測結果"""
    CORRECT = "的中"
    INCORRECT = "外れ"
    PARTIAL = "部分的中"
    PENDING = "検証待ち"

class ValidationPeriod(Enum):
    """検証期間"""
    INTRADAY = "日中"          # 当日決済
    DAILY = "1日"              # 翌日決済
    WEEKLY = "1週間"           # 1週間以内
    MONTHLY = "1ヶ月"          # 1ヶ月以内

@dataclass
class Prediction:
    """予測データ"""
    prediction_id: str
    symbol: str
    name: str
    prediction_date: datetime
    prediction_type: str          # 買い, 売り, 検討, 様子見
    target_price: float          # 目標価格
    stop_loss: float            # 損切り価格
    confidence: float           # 信頼度(0-100)
    expected_return: float      # 期待リターン(%)
    risk_level: str            # リスクレベル
    validation_period: ValidationPeriod
    reasoning: str             # 予測根拠

    # 結果データ
    actual_result: Optional[PredictionResult] = None
    actual_price: Optional[float] = None
    actual_return: Optional[float] = None
    validation_date: Optional[datetime] = None
    performance_score: Optional[float] = None

@dataclass
class ValidationMetrics:
    """検証指標"""
    total_predictions: int
    correct_predictions: int
    accuracy_rate: float        # 的中率(%)
    avg_return: float          # 平均リターン(%)
    win_rate: float           # 勝率(%)
    avg_win: float            # 平均勝ち幅(%)
    avg_loss: float           # 平均負け幅(%)
    profit_factor: float      # プロフィットファクター
    max_drawdown: float       # 最大ドローダウン(%)
    sharpe_ratio: float       # シャープレシオ
    hit_ratio_by_confidence: Dict[str, float]  # 信頼度別的中率

@dataclass
class ModelPerformance:
    """モデル性能分析"""
    model_name: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    best_performing_sectors: List[str]
    worst_performing_sectors: List[str]
    reliability_score: float      # 信頼性スコア(0-100)

class PredictionValidator:
    """
    予測精度検証システム
    93%精度目標の継続的検証・改善
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("validation_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "predictions.db"
        self._init_database()

        # メモリ内キャッシュ
        self.recent_predictions: deque = deque(maxlen=1000)
        self.validation_history: List[ValidationMetrics] = []

        # 精度目標設定
        self.target_accuracy = 93.0  # 93%精度目標
        self.minimum_confidence = 70.0  # 最低信頼度

        # 拡張システム統合
        if ENHANCED_SYMBOLS_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()

        self.logger.info("Prediction validator initialized with 93% accuracy target")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    prediction_date TEXT NOT NULL,
                    prediction_type TEXT,
                    target_price REAL,
                    stop_loss REAL,
                    confidence REAL,
                    expected_return REAL,
                    risk_level TEXT,
                    validation_period TEXT,
                    reasoning TEXT,
                    actual_result TEXT,
                    actual_price REAL,
                    actual_return REAL,
                    validation_date TEXT,
                    performance_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction_date
                ON predictions(prediction_date)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol
                ON predictions(symbol)
            """)

    async def record_prediction(self, prediction: Prediction) -> bool:
        """
        予測記録

        Args:
            prediction: 予測データ

        Returns:
            記録成功フラグ
        """
        try:
            # データベースに保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions (
                        prediction_id, symbol, name, prediction_date, prediction_type,
                        target_price, stop_loss, confidence, expected_return, risk_level,
                        validation_period, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.prediction_id,
                    prediction.symbol,
                    prediction.name,
                    prediction.prediction_date.isoformat(),
                    prediction.prediction_type,
                    prediction.target_price,
                    prediction.stop_loss,
                    prediction.confidence,
                    prediction.expected_return,
                    prediction.risk_level,
                    prediction.validation_period.value,
                    prediction.reasoning
                ))

            # メモリキャッシュに追加
            self.recent_predictions.append(prediction)

            self.logger.info(f"Recorded prediction for {prediction.symbol}: {prediction.prediction_type}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to record prediction: {e}")
            return False

    async def validate_prediction(self, prediction_id: str,
                                actual_price: float,
                                validation_date: datetime = None) -> bool:
        """
        予測結果検証

        Args:
            prediction_id: 予測ID
            actual_price: 実際の価格
            validation_date: 検証日時

        Returns:
            検証成功フラグ
        """
        try:
            if validation_date is None:
                validation_date = datetime.now()

            # 予測データ取得
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM predictions WHERE prediction_id = ?
                """, (prediction_id,))
                row = cursor.fetchone()

                if not row:
                    self.logger.error(f"Prediction not found: {prediction_id}")
                    return False

                # 結果判定
                prediction_type = row[4]  # prediction_type
                target_price = row[5]     # target_price
                stop_loss = row[6]        # stop_loss
                confidence = row[7]       # confidence
                expected_return = row[8]  # expected_return

                # 実際のリターン計算（仮定：予測時の価格から）
                predicted_price = target_price  # 簡易実装
                actual_return = (actual_price - predicted_price) / predicted_price * 100

                # 結果判定
                result = self._judge_prediction_result(
                    prediction_type, target_price, stop_loss, actual_price, expected_return, actual_return
                )

                # パフォーマンススコア計算
                performance_score = self._calculate_performance_score(
                    result, confidence, expected_return, actual_return
                )

                # 結果更新
                conn.execute("""
                    UPDATE predictions SET
                        actual_result = ?, actual_price = ?, actual_return = ?,
                        validation_date = ?, performance_score = ?
                    WHERE prediction_id = ?
                """, (
                    result.value, actual_price, actual_return,
                    validation_date.isoformat(), performance_score, prediction_id
                ))

                self.logger.info(f"Validated prediction {prediction_id}: {result.value}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to validate prediction: {e}")
            return False

    def _judge_prediction_result(self, prediction_type: str, target_price: float,
                                stop_loss: float, actual_price: float,
                                expected_return: float, actual_return: float) -> PredictionResult:
        """予測結果判定"""

        if prediction_type in ["買い", "強い買い"]:
            if actual_return >= expected_return * 0.8:  # 期待の80%以上
                return PredictionResult.CORRECT
            elif actual_return > 0:  # プラスだが期待未満
                return PredictionResult.PARTIAL
            else:  # マイナス
                return PredictionResult.INCORRECT

        elif prediction_type in ["売り", "強い売り"]:
            if actual_return <= -abs(expected_return) * 0.8:  # 期待の80%以上下落
                return PredictionResult.CORRECT
            elif actual_return < 0:  # マイナスだが期待未満
                return PredictionResult.PARTIAL
            else:  # プラス（予想外れ）
                return PredictionResult.INCORRECT

        else:  # 検討、様子見
            if abs(actual_return) <= 2.0:  # ±2%以内
                return PredictionResult.CORRECT
            else:
                return PredictionResult.PARTIAL

    def _calculate_performance_score(self, result: PredictionResult, confidence: float,
                                   expected_return: float, actual_return: float) -> float:
        """パフォーマンススコア計算"""

        base_score = 0
        if result == PredictionResult.CORRECT:
            base_score = 100
        elif result == PredictionResult.PARTIAL:
            base_score = 60
        else:
            base_score = 0

        # 信頼度ボーナス/ペナルティ
        confidence_factor = confidence / 100

        # リターン精度ボーナス
        if expected_return != 0:
            accuracy_factor = min(1.0, 1 - abs(expected_return - actual_return) / abs(expected_return))
        else:
            accuracy_factor = 0.5

        final_score = base_score * confidence_factor * (0.5 + 0.5 * accuracy_factor)
        return max(0, min(100, final_score))

    async def get_current_accuracy(self, days_back: int = 30) -> ValidationMetrics:
        """
        現在の精度指標取得

        Args:
            days_back: 対象期間（日数）

        Returns:
            検証指標
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM predictions
                    WHERE prediction_date >= ? AND actual_result IS NOT NULL
                    ORDER BY prediction_date DESC
                """, (cutoff_date.isoformat(),))

                rows = cursor.fetchall()

                if not rows:
                    return ValidationMetrics(
                        total_predictions=0, correct_predictions=0, accuracy_rate=0.0,
                        avg_return=0.0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                        profit_factor=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
                        hit_ratio_by_confidence={}
                    )

                # 基本統計
                total_predictions = len(rows)
                correct_predictions = sum(1 for row in rows if row[12] == "的中")  # actual_result
                partial_correct = sum(1 for row in rows if row[12] == "部分的中")

                accuracy_rate = (correct_predictions + partial_correct * 0.5) / total_predictions * 100

                # リターン統計
                returns = [row[14] for row in rows if row[14] is not None]  # actual_return
                avg_return = statistics.mean(returns) if returns else 0.0

                winning_returns = [r for r in returns if r > 0]
                losing_returns = [r for r in returns if r < 0]

                win_rate = len(winning_returns) / len(returns) * 100 if returns else 0.0
                avg_win = statistics.mean(winning_returns) if winning_returns else 0.0
                avg_loss = statistics.mean(losing_returns) if losing_returns else 0.0

                # プロフィットファクター
                total_wins = sum(winning_returns) if winning_returns else 0
                total_losses = abs(sum(losing_returns)) if losing_returns else 1
                profit_factor = total_wins / total_losses if total_losses > 0 else 0

                # 最大ドローダウン
                cumulative_returns = np.cumsum(returns) if returns else [0]
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = cumulative_returns - running_max
                max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0

                # シャープレシオ
                sharpe_ratio = (avg_return / statistics.stdev(returns)) if len(returns) > 1 else 0.0

                # 信頼度別的中率
                hit_ratio_by_confidence = self._calculate_confidence_hit_ratio(rows)

                return ValidationMetrics(
                    total_predictions=total_predictions,
                    correct_predictions=correct_predictions,
                    accuracy_rate=accuracy_rate,
                    avg_return=avg_return,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    profit_factor=profit_factor,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    hit_ratio_by_confidence=hit_ratio_by_confidence
                )

        except Exception as e:
            self.logger.error(f"Failed to calculate accuracy: {e}")
            return ValidationMetrics(
                total_predictions=0, correct_predictions=0, accuracy_rate=0.0,
                avg_return=0.0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                profit_factor=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
                hit_ratio_by_confidence={}
            )

    def _calculate_confidence_hit_ratio(self, rows: List[tuple]) -> Dict[str, float]:
        """信頼度別的中率計算"""

        confidence_groups = {
            "90%以上": [],
            "80-89%": [],
            "70-79%": [],
            "60-69%": [],
            "60%未満": []
        }

        for row in rows:
            confidence = row[7] if row[7] else 0  # confidence
            result = row[12]  # actual_result

            if confidence >= 90:
                confidence_groups["90%以上"].append(result)
            elif confidence >= 80:
                confidence_groups["80-89%"].append(result)
            elif confidence >= 70:
                confidence_groups["70-79%"].append(result)
            elif confidence >= 60:
                confidence_groups["60-69%"].append(result)
            else:
                confidence_groups["60%未満"].append(result)

        hit_ratios = {}
        for group, results in confidence_groups.items():
            if results:
                correct = sum(1 for r in results if r in ["的中", "部分的中"])
                hit_ratios[group] = correct / len(results) * 100
            else:
                hit_ratios[group] = 0.0

        return hit_ratios

    async def generate_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""

        try:
            # 現在の精度
            current_metrics = await self.get_current_accuracy(30)

            # 長期トレンド（90日）
            long_term_metrics = await self.get_current_accuracy(90)

            # 最近の傾向分析
            recent_trend = await self._analyze_recent_trend()

            # 改善提案
            improvement_suggestions = self._generate_improvement_suggestions(current_metrics)

            return {
                "current_performance": {
                    "period": "過去30日",
                    "accuracy_rate": current_metrics.accuracy_rate,
                    "total_predictions": current_metrics.total_predictions,
                    "win_rate": current_metrics.win_rate,
                    "avg_return": current_metrics.avg_return,
                    "profit_factor": current_metrics.profit_factor,
                    "sharpe_ratio": current_metrics.sharpe_ratio,
                    "target_achievement": "達成" if current_metrics.accuracy_rate >= self.target_accuracy else "未達成"
                },
                "long_term_performance": {
                    "period": "過去90日",
                    "accuracy_rate": long_term_metrics.accuracy_rate,
                    "total_predictions": long_term_metrics.total_predictions,
                    "trend": "上昇" if current_metrics.accuracy_rate > long_term_metrics.accuracy_rate else "下降"
                },
                "confidence_analysis": current_metrics.hit_ratio_by_confidence,
                "recent_trend": recent_trend,
                "improvement_suggestions": improvement_suggestions,
                "system_status": {
                    "target_accuracy": self.target_accuracy,
                    "minimum_confidence": self.minimum_confidence,
                    "validation_status": "正常稼働"
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {"error": f"Report generation failed: {e}"}

    async def _analyze_recent_trend(self) -> Dict[str, Any]:
        """最近の傾向分析"""

        try:
            # 直近7日vs前7日の比較
            recent_7d = await self.get_current_accuracy(7)
            previous_7d_start = datetime.now() - timedelta(days=14)
            previous_7d_end = datetime.now() - timedelta(days=7)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM predictions
                    WHERE prediction_date >= ? AND prediction_date < ?
                    AND actual_result IS NOT NULL
                """, (previous_7d_start.isoformat(), previous_7d_end.isoformat()))

                previous_rows = cursor.fetchall()

                if previous_rows:
                    previous_correct = sum(1 for row in previous_rows if row[12] in ["的中", "部分的中"])
                    previous_accuracy = previous_correct / len(previous_rows) * 100
                else:
                    previous_accuracy = 0

                accuracy_change = recent_7d.accuracy_rate - previous_accuracy

                return {
                    "recent_7d_accuracy": recent_7d.accuracy_rate,
                    "previous_7d_accuracy": previous_accuracy,
                    "accuracy_change": accuracy_change,
                    "trend_direction": "改善" if accuracy_change > 0 else "悪化" if accuracy_change < 0 else "横ばい",
                    "recent_predictions": recent_7d.total_predictions
                }

        except Exception as e:
            self.logger.error(f"Failed to analyze recent trend: {e}")
            return {"error": "Trend analysis failed"}

    def _generate_improvement_suggestions(self, metrics: ValidationMetrics) -> List[str]:
        """改善提案生成"""

        suggestions = []

        if metrics.accuracy_rate < self.target_accuracy:
            suggestions.append(f"精度{metrics.accuracy_rate:.1f}%が目標93%を下回っています。信頼度の低い予測を控えることを推奨")

        if metrics.win_rate < 60:
            suggestions.append("勝率が60%を下回っています。損切りライン見直しとリスク管理強化を推奨")

        if metrics.profit_factor < 1.5:
            suggestions.append("プロフィットファクターが低い状態です。利確目標の最適化を推奨")

        if metrics.max_drawdown > 15:
            suggestions.append("最大ドローダウンが15%を超えています。ポジションサイズの見直しを推奨")

        # 信頼度別分析
        high_confidence_hit_rate = metrics.hit_ratio_by_confidence.get("90%以上", 0)
        if high_confidence_hit_rate < 95:
            suggestions.append("高信頼度予測の的中率が低下しています。予測モデルの再調整を推奨")

        if not suggestions:
            suggestions.append("現在のパフォーマンスは良好です。この調子で継続してください")

        return suggestions

# テスト関数
async def test_prediction_validator():
    """予測精度検証システムのテスト"""
    print("=== 予測精度検証システム テスト ===")

    validator = PredictionValidator()

    print(f"目標精度: {validator.target_accuracy}%")
    print(f"最低信頼度: {validator.minimum_confidence}%")

    # サンプル予測データ作成・記録
    print(f"\n[ サンプル予測データ作成 ]")
    sample_predictions = []

    for i in range(20):
        prediction = Prediction(
            prediction_id=f"TEST_{i:03d}",
            symbol="7203",
            name="トヨタ自動車",
            prediction_date=datetime.now() - timedelta(days=i),
            prediction_type="買い" if i % 3 == 0 else "検討",
            target_price=3000 + np.random.randint(-100, 100),
            stop_loss=2800,
            confidence=np.random.uniform(70, 95),
            expected_return=np.random.uniform(2, 8),
            risk_level="低リスク",
            validation_period=ValidationPeriod.DAILY,
            reasoning="テスト予測データ"
        )

        await validator.record_prediction(prediction)

        # 一部の予測に検証結果を追加
        if i < 15:  # 最新15件は検証済みとする
            actual_price = 3000 + np.random.randint(-150, 150)
            await validator.validate_prediction(prediction.prediction_id, actual_price)

        sample_predictions.append(prediction)

    print(f"作成した予測データ: {len(sample_predictions)}件")

    # 現在の精度取得
    print(f"\n[ 現在の精度分析 ]")
    current_metrics = await validator.get_current_accuracy(30)

    print(f"総予測数: {current_metrics.total_predictions}")
    print(f"的中数: {current_metrics.correct_predictions}")
    print(f"精度: {current_metrics.accuracy_rate:.1f}%")
    print(f"勝率: {current_metrics.win_rate:.1f}%")
    print(f"平均リターン: {current_metrics.avg_return:.2f}%")
    print(f"プロフィットファクター: {current_metrics.profit_factor:.2f}")

    # パフォーマンスレポート生成
    print(f"\n[ パフォーマンスレポート ]")
    report = await validator.generate_performance_report()

    if "error" not in report:
        current_perf = report["current_performance"]
        print(f"目標達成状況: {current_perf['target_achievement']}")
        print(f"システム状況: {report['system_status']['validation_status']}")

        print(f"\n信頼度別的中率:")
        for level, rate in report["confidence_analysis"].items():
            print(f"  {level}: {rate:.1f}%")

        print(f"\n改善提案:")
        for suggestion in report["improvement_suggestions"]:
            print(f"  • {suggestion}")
    else:
        print(f"レポート生成エラー: {report['error']}")

    print(f"\n=== 予測精度検証システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(test_prediction_validator())