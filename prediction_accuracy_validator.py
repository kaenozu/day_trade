#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Accuracy Validator - 予測精度検証システム

既存システムの予測精度を実測定・評価するシステム
Phase5-B #904実装：93%精度目標の現実的評価
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
import statistics
import sqlite3
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

# 既存システムのインポート
try:
    from real_data_provider_v2 import real_data_provider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False

try:
    from advanced_technical_analyzer import AdvancedTechnicalAnalyzer
    ADVANCED_TECHNICAL_AVAILABLE = True
except ImportError:
    ADVANCED_TECHNICAL_AVAILABLE = False

try:
    from ensemble_signal_generator import EnsembleSignalGenerator
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from daytrade import DayTradeOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

class PredictionType(Enum):
    """予測タイプ"""
    PRICE_DIRECTION = "価格方向"      # 上昇/下降の方向性
    PRICE_TARGET = "価格目標"        # 具体的な価格予測
    SIGNAL_ACCURACY = "シグナル精度"  # 売買シグナルの的中率
    TREND_PREDICTION = "トレンド予測" # トレンド継続/転換

class ValidationMethod(Enum):
    """検証手法"""
    FORWARD_TESTING = "フォワードテスト"   # 未来データでの検証
    WALK_FORWARD = "ウォークフォワード"     # 時系列分割検証
    CROSS_VALIDATION = "交差検証"          # k-fold交差検証
    MONTE_CARLO = "モンテカルロ"           # 確率的シミュレーション

@dataclass
class PredictionRecord:
    """予測記録"""
    prediction_id: str
    symbol: str
    prediction_time: datetime
    prediction_type: PredictionType
    predicted_value: Any           # 予測値（方向、価格等）
    confidence: float             # 信頼度
    prediction_horizon: int       # 予測期間（分）

    # 実測結果
    actual_time: Optional[datetime] = None
    actual_value: Any = None
    is_correct: Optional[bool] = None
    accuracy_score: float = 0.0

    # メタデータ
    model_used: str = ""
    features_used: List[str] = field(default_factory=list)
    market_conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccuracyMetrics:
    """精度指標"""
    overall_accuracy: float
    direction_accuracy: float      # 方向予測精度
    price_accuracy: float         # 価格予測精度
    signal_accuracy: float        # シグナル精度

    # 詳細統計
    total_predictions: int
    correct_predictions: int
    false_positives: int
    false_negatives: int

    # 信頼度別精度
    high_confidence_accuracy: float    # 80%以上信頼度
    medium_confidence_accuracy: float  # 60-80%信頼度
    low_confidence_accuracy: float     # 60%未満信頼度

    # 期間別精度
    short_term_accuracy: float     # 1時間以内
    medium_term_accuracy: float    # 1-6時間
    long_term_accuracy: float      # 6時間以上

class PredictionAccuracyValidator:
    """予測精度検証システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("prediction_validation")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "predictions.db"
        self._init_database()

        # システム初期化
        self.orchestrator = None
        if ORCHESTRATOR_AVAILABLE:
            self.orchestrator = DayTradeOrchestrator()

        self.ensemble = None
        if ENSEMBLE_AVAILABLE:
            self.ensemble = EnsembleSignalGenerator()

        self.technical_analyzer = None
        if ADVANCED_TECHNICAL_AVAILABLE:
            self.technical_analyzer = AdvancedTechnicalAnalyzer()

        # 検証設定（即効改善適用）
        self.validation_config = {
            'min_confidence_threshold': 70.0,  # 70%未満の予測を除外（信頼度フィルタリング強化）
            'prediction_horizons': [15, 30, 60, 180, 360],  # 分
            'tolerance_percentage': 2.0,  # 価格予測の許容誤差%
            'sample_size': 100,          # 検証サンプル数
        }

        # 結果保存
        self.prediction_records: List[PredictionRecord] = []
        self.accuracy_history: deque = deque(maxlen=1000)

        self.logger.info("Prediction accuracy validator initialized")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 予測記録テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_records (
                    prediction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    prediction_time TEXT NOT NULL,
                    prediction_type TEXT,
                    predicted_value TEXT,
                    confidence REAL,
                    prediction_horizon INTEGER,
                    actual_time TEXT,
                    actual_value TEXT,
                    is_correct BOOLEAN,
                    accuracy_score REAL,
                    model_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 精度履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    validation_time TEXT NOT NULL,
                    overall_accuracy REAL,
                    direction_accuracy REAL,
                    price_accuracy REAL,
                    signal_accuracy REAL,
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    validation_method TEXT,
                    symbol_tested TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def validate_current_system_accuracy(self, symbols: List[str],
                                             validation_hours: int = 24) -> AccuracyMetrics:
        """現在のシステムの予測精度を検証"""

        self.logger.info(f"Starting accuracy validation for {len(symbols)} symbols over {validation_hours} hours")

        # 検証期間設定
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=validation_hours)

        all_predictions = []

        for symbol in symbols:
            self.logger.info(f"Validating predictions for {symbol}")

            # 過去データ取得
            historical_data = await self._get_validation_data(symbol, start_time, end_time)

            if historical_data is None or historical_data.empty:
                self.logger.warning(f"No historical data for {symbol}")
                continue

            # 予測生成と検証
            symbol_predictions = await self._generate_and_validate_predictions(
                symbol, historical_data, start_time, end_time
            )
            all_predictions.extend(symbol_predictions)

        # 精度指標計算
        accuracy_metrics = self._calculate_accuracy_metrics(all_predictions)

        # データベース保存
        await self._save_validation_results(accuracy_metrics, symbols)

        return accuracy_metrics

    async def _get_validation_data(self, symbol: str, start_time: datetime,
                                 end_time: datetime) -> Optional[pd.DataFrame]:
        """検証用データ取得"""

        try:
            if REAL_DATA_PROVIDER_AVAILABLE:
                # 長期間データ取得
                data = await real_data_provider.get_stock_data(symbol, period="3mo")

                if data is not None and not data.empty:
                    # タイムゾーン調整
                    data.index = pd.to_datetime(data.index)
                    if data.index.tz is not None:
                        data.index = data.index.tz_convert(None)

                    # 期間フィルタリング
                    start_ts = pd.Timestamp(start_time)
                    end_ts = pd.Timestamp(end_time)

                    filtered_data = data[(data.index >= start_ts) & (data.index <= end_ts)]
                    return filtered_data

            return None

        except Exception as e:
            self.logger.error(f"Failed to get validation data for {symbol}: {e}")
            return None

    async def _generate_and_validate_predictions(self, symbol: str, data: pd.DataFrame,
                                               start_time: datetime, end_time: datetime) -> List[PredictionRecord]:
        """予測生成と検証実行"""

        predictions = []

        # より現実的な予測間隔でサンプリング（5日おき）
        sample_indices = range(10, len(data) - 1, 5)  # 10日目から5日おきにサンプリング

        for i in sample_indices:
            current_time = data.index[i]
            future_time = data.index[i + 1] if i + 1 < len(data) else None

            if future_time is None:
                continue

            try:
                # 現在時点までのデータ
                historical_subset = data.iloc[:i+1]

                if len(historical_subset) < 10:  # 最低データ量
                    continue

                # 各システムで予測生成
                prediction_results = await self._run_prediction_systems(
                    symbol, historical_subset, current_time
                )

                # 実測値取得
                actual_price = data.iloc[i + 1]['Close']
                actual_direction = 1 if actual_price > data.iloc[i]['Close'] else -1

                # 予測記録作成（信頼度フィルタリング適用）
                for pred_result in prediction_results:
                    # 信頼度フィルタリング
                    if pred_result['confidence'] < self.validation_config['min_confidence_threshold']:
                        continue

                    prediction = PredictionRecord(
                        prediction_id=f"{symbol}_{current_time.strftime('%Y%m%d_%H%M%S')}_{pred_result['model']}",
                        symbol=symbol,
                        prediction_time=current_time,
                        prediction_type=pred_result['type'],
                        predicted_value=pred_result['value'],
                        confidence=pred_result['confidence'],
                        prediction_horizon=pred_result['horizon'],
                        actual_time=future_time,
                        actual_value=actual_price if pred_result['type'] == PredictionType.PRICE_TARGET else actual_direction,
                        model_used=pred_result['model']
                    )

                    # 精度計算
                    prediction.is_correct, prediction.accuracy_score = self._calculate_prediction_accuracy(
                        prediction, data.iloc[i]['Close'], actual_price
                    )

                    predictions.append(prediction)

            except Exception as e:
                self.logger.error(f"Prediction generation failed for {symbol} at {current_time}: {e}")
                continue

        return predictions

    async def _run_prediction_systems(self, symbol: str, data: pd.DataFrame,
                                    current_time: datetime) -> List[Dict[str, Any]]:
        """各予測システムの実行"""

        results = []

        try:
            # 1. Advanced Technical Analyzer
            if self.technical_analyzer:
                analysis = await self.technical_analyzer.analyze_symbol(symbol, period="1mo")

                if analysis:
                    # 価格方向予測
                    direction = 1 if analysis.composite_score > 60 else -1
                    results.append({
                        'model': 'AdvancedTechnical',
                        'type': PredictionType.PRICE_DIRECTION,
                        'value': direction,
                        'confidence': analysis.composite_score,
                        'horizon': 60  # 1時間
                    })

                    # 価格目標予測
                    current_price = analysis.current_price
                    if analysis.composite_score > 70:
                        target_price = current_price * 1.02  # 2%上昇予測
                    elif analysis.composite_score < 40:
                        target_price = current_price * 0.98  # 2%下落予測
                    else:
                        target_price = current_price  # 横ばい予測

                    results.append({
                        'model': 'AdvancedTechnical',
                        'type': PredictionType.PRICE_TARGET,
                        'value': target_price,
                        'confidence': analysis.composite_score,
                        'horizon': 60
                    })

            # 2. Ensemble Signal Generator
            if self.ensemble:
                try:
                    signals = await self.ensemble.generate_ensemble_signals([symbol])

                    if symbol in signals:
                        signal = signals[symbol]
                        direction = 1 if signal.signal_type == "BUY" else -1

                        results.append({
                            'model': 'EnsembleSignal',
                            'type': PredictionType.SIGNAL_ACCURACY,
                            'value': direction,
                            'confidence': signal.confidence,
                            'horizon': 30  # 30分
                        })
                except Exception as e:
                    self.logger.debug(f"Ensemble prediction failed: {e}")

            # 3. Simple Technical Indicators（最適化パラメータ）
            if len(data) >= 10:
                # 移動平均ベース予測（3/8パラメータに最適化）
                sma_short = data['Close'].rolling(3).mean().iloc[-1]
                sma_long = data['Close'].rolling(8).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]

                if sma_short > sma_long:
                    direction = 1
                    confidence = min(80, abs((sma_short - sma_long) / sma_long * 100) * 10)
                else:
                    direction = -1
                    confidence = min(80, abs((sma_long - sma_short) / sma_short * 100) * 10)

                results.append({
                    'model': 'SimpleMA',
                    'type': PredictionType.PRICE_DIRECTION,
                    'value': direction,
                    'confidence': max(50, confidence),  # 最低50%信頼度保証
                    'horizon': 15  # 15分
                })

            # 4. Momentum-based prediction
            if len(data) >= 5:
                # 短期モメンタム予測
                recent_returns = data['Close'].pct_change().iloc[-3:].mean()
                current_price = data['Close'].iloc[-1]

                if recent_returns > 0.01:  # 1%以上上昇トレンド
                    direction = 1
                    confidence = min(90, abs(recent_returns * 1000))
                elif recent_returns < -0.01:  # 1%以上下落トレンド
                    direction = -1
                    confidence = min(90, abs(recent_returns * 1000))
                else:
                    direction = 0  # 横ばい
                    confidence = 60

                if direction != 0:  # 横ばい以外の場合
                    results.append({
                        'model': 'Momentum',
                        'type': PredictionType.PRICE_DIRECTION,
                        'value': direction,
                        'confidence': max(50, confidence),
                        'horizon': 30  # 30分
                    })

        except Exception as e:
            self.logger.error(f"Prediction system execution failed: {e}")

        return results

    def _calculate_prediction_accuracy(self, prediction: PredictionRecord,
                                     current_price: float, actual_price: float) -> Tuple[bool, float]:
        """予測精度計算"""

        try:
            if prediction.prediction_type == PredictionType.PRICE_DIRECTION:
                # 方向予測の場合
                actual_direction = 1 if actual_price > current_price else -1
                predicted_direction = prediction.predicted_value

                is_correct = actual_direction == predicted_direction
                accuracy_score = 100.0 if is_correct else 0.0

                return is_correct, accuracy_score

            elif prediction.prediction_type == PredictionType.PRICE_TARGET:
                # 価格予測の場合
                predicted_price = prediction.predicted_value
                error_percentage = abs(actual_price - predicted_price) / actual_price * 100

                tolerance = self.validation_config['tolerance_percentage']
                is_correct = error_percentage <= tolerance

                # 誤差に基づくスコア計算
                accuracy_score = max(0, 100 - error_percentage * 10)

                return is_correct, accuracy_score

            elif prediction.prediction_type == PredictionType.SIGNAL_ACCURACY:
                # シグナル精度の場合
                actual_direction = 1 if actual_price > current_price else -1
                predicted_signal = prediction.predicted_value

                is_correct = actual_direction == predicted_signal
                accuracy_score = 100.0 if is_correct else 0.0

                return is_correct, accuracy_score

        except Exception as e:
            self.logger.error(f"Accuracy calculation failed: {e}")

        return False, 0.0

    def _calculate_accuracy_metrics(self, predictions: List[PredictionRecord]) -> AccuracyMetrics:
        """精度指標の計算"""

        if not predictions:
            return AccuracyMetrics(
                overall_accuracy=0.0, direction_accuracy=0.0, price_accuracy=0.0,
                signal_accuracy=0.0, total_predictions=0, correct_predictions=0,
                false_positives=0, false_negatives=0, high_confidence_accuracy=0.0,
                medium_confidence_accuracy=0.0, low_confidence_accuracy=0.0,
                short_term_accuracy=0.0, medium_term_accuracy=0.0, long_term_accuracy=0.0
            )

        # 基本統計
        total_predictions = len(predictions)
        correct_predictions = sum(1 for p in predictions if p.is_correct)
        overall_accuracy = correct_predictions / total_predictions * 100

        # タイプ別精度
        direction_preds = [p for p in predictions if p.prediction_type == PredictionType.PRICE_DIRECTION]
        price_preds = [p for p in predictions if p.prediction_type == PredictionType.PRICE_TARGET]
        signal_preds = [p for p in predictions if p.prediction_type == PredictionType.SIGNAL_ACCURACY]

        direction_accuracy = (sum(1 for p in direction_preds if p.is_correct) / len(direction_preds) * 100) if direction_preds else 0
        price_accuracy = (sum(1 for p in price_preds if p.is_correct) / len(price_preds) * 100) if price_preds else 0
        signal_accuracy = (sum(1 for p in signal_preds if p.is_correct) / len(signal_preds) * 100) if signal_preds else 0

        # 信頼度別精度
        high_conf = [p for p in predictions if p.confidence >= 80]
        med_conf = [p for p in predictions if 60 <= p.confidence < 80]
        low_conf = [p for p in predictions if p.confidence < 60]

        high_confidence_accuracy = (sum(1 for p in high_conf if p.is_correct) / len(high_conf) * 100) if high_conf else 0
        medium_confidence_accuracy = (sum(1 for p in med_conf if p.is_correct) / len(med_conf) * 100) if med_conf else 0
        low_confidence_accuracy = (sum(1 for p in low_conf if p.is_correct) / len(low_conf) * 100) if low_conf else 0

        # 期間別精度
        short_term = [p for p in predictions if p.prediction_horizon <= 30]
        medium_term = [p for p in predictions if 30 < p.prediction_horizon <= 180]
        long_term = [p for p in predictions if p.prediction_horizon > 180]

        short_term_accuracy = (sum(1 for p in short_term if p.is_correct) / len(short_term) * 100) if short_term else 0
        medium_term_accuracy = (sum(1 for p in medium_term if p.is_correct) / len(medium_term) * 100) if medium_term else 0
        long_term_accuracy = (sum(1 for p in long_term if p.is_correct) / len(long_term) * 100) if long_term else 0

        # False Positive/Negative計算（買いシグナルベース）
        buy_signals = [p for p in predictions if p.predicted_value == 1]
        actual_ups = [p for p in predictions if p.actual_value == 1]

        false_positives = len([p for p in buy_signals if p.actual_value != 1])
        false_negatives = len([p for p in actual_ups if p.predicted_value != 1])

        return AccuracyMetrics(
            overall_accuracy=overall_accuracy,
            direction_accuracy=direction_accuracy,
            price_accuracy=price_accuracy,
            signal_accuracy=signal_accuracy,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            false_positives=false_positives,
            false_negatives=false_negatives,
            high_confidence_accuracy=high_confidence_accuracy,
            medium_confidence_accuracy=medium_confidence_accuracy,
            low_confidence_accuracy=low_confidence_accuracy,
            short_term_accuracy=short_term_accuracy,
            medium_term_accuracy=medium_term_accuracy,
            long_term_accuracy=long_term_accuracy
        )

    async def _save_validation_results(self, metrics: AccuracyMetrics, symbols: List[str]):
        """検証結果の保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO accuracy_history
                    (validation_time, overall_accuracy, direction_accuracy, price_accuracy,
                     signal_accuracy, total_predictions, correct_predictions,
                     validation_method, symbol_tested)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    metrics.overall_accuracy,
                    metrics.direction_accuracy,
                    metrics.price_accuracy,
                    metrics.signal_accuracy,
                    metrics.total_predictions,
                    metrics.correct_predictions,
                    "FORWARD_TESTING",
                    ",".join(symbols)
                ))

        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")

    def evaluate_93_percent_target(self, current_accuracy: float) -> Dict[str, Any]:
        """93%精度目標の現実性評価"""

        gap_to_target = 93.0 - current_accuracy

        evaluation = {
            'current_accuracy': current_accuracy,
            'target_accuracy': 93.0,
            'gap_percentage': gap_to_target,
            'achievability_assessment': '',
            'required_improvements': [],
            'estimated_effort': '',
            'recommendations': []
        }

        if current_accuracy >= 93.0:
            evaluation['achievability_assessment'] = '✅ 既に目標達成済み'
            evaluation['estimated_effort'] = '不要'

        elif current_accuracy >= 85.0:
            evaluation['achievability_assessment'] = '🟡 達成可能（高難易度）'
            evaluation['estimated_effort'] = '高（3-6ヶ月）'
            evaluation['required_improvements'] = [
                'アンサンブル手法の最適化',
                'ハイパーパラメータチューニング',
                'より高品質なデータソース統合'
            ]

        elif current_accuracy >= 70.0:
            evaluation['achievability_assessment'] = '🟠 達成困難（要大幅改善）'
            evaluation['estimated_effort'] = '非常に高（6-12ヶ月）'
            evaluation['required_improvements'] = [
                '機械学習モデルの根本的見直し',
                '特徴量エンジニアリング強化',
                '代替アルゴリズムの導入',
                'より多様なデータソース統合'
            ]

        else:
            evaluation['achievability_assessment'] = '❌ 現実的でない'
            evaluation['estimated_effort'] = '極めて高（1年以上）'
            evaluation['required_improvements'] = [
                'システム全体の再設計',
                '高度な機械学習手法の導入',
                'リアルタイム市場データの統合',
                'プロフェッショナルレベルの分析ツール'
            ]

        # 推奨事項
        if gap_to_target > 0:
            evaluation['recommendations'] = [
                f'現在の精度{current_accuracy:.1f}%から{gap_to_target:.1f}%ポイントの改善が必要',
                '段階的な目標設定（例：75% → 80% → 85% → 90% → 93%）',
                'A/Bテストによる改善効果の定量評価',
                '他の成功事例やベンチマークとの比較研究'
            ]

        return evaluation

# テスト関数
async def test_prediction_accuracy_validator():
    """予測精度検証システムのテスト"""

    print("=== 予測精度検証システム テスト ===")

    validator = PredictionAccuracyValidator()

    # テスト銘柄
    test_symbols = ["7203", "8306", "4751"]
    validation_hours = 720  # 30日分のデータで検証

    print(f"\n[ {len(test_symbols)}銘柄の予測精度検証 ]")
    print(f"検証期間: 過去{validation_hours}時間")

    try:
        # 精度検証実行
        print("\n検証実行中...")
        metrics = await validator.validate_current_system_accuracy(test_symbols, validation_hours)

        # 結果表示
        print(f"\n[ 予測精度結果 ]")
        print(f"総合精度: {metrics.overall_accuracy:.1f}%")
        print(f"方向予測精度: {metrics.direction_accuracy:.1f}%")
        print(f"価格予測精度: {metrics.price_accuracy:.1f}%")
        print(f"シグナル精度: {metrics.signal_accuracy:.1f}%")

        print(f"\n[ 統計情報 ]")
        print(f"総予測数: {metrics.total_predictions}")
        print(f"正解数: {metrics.correct_predictions}")
        print(f"誤報: {metrics.false_positives}")
        print(f"見逃し: {metrics.false_negatives}")

        print(f"\n[ 信頼度別精度 ]")
        print(f"高信頼度(80%+): {metrics.high_confidence_accuracy:.1f}%")
        print(f"中信頼度(60-80%): {metrics.medium_confidence_accuracy:.1f}%")
        print(f"低信頼度(60%未満): {metrics.low_confidence_accuracy:.1f}%")

        print(f"\n[ 期間別精度 ]")
        print(f"短期(30分以下): {metrics.short_term_accuracy:.1f}%")
        print(f"中期(30分-3時間): {metrics.medium_term_accuracy:.1f}%")
        print(f"長期(3時間超): {metrics.long_term_accuracy:.1f}%")

        # 93%目標評価
        print(f"\n[ 93%精度目標の評価 ]")
        evaluation = validator.evaluate_93_percent_target(metrics.overall_accuracy)

        print(f"現在精度: {evaluation['current_accuracy']:.1f}%")
        print(f"目標精度: {evaluation['target_accuracy']}%")
        print(f"達成性評価: {evaluation['achievability_assessment']}")
        print(f"必要努力: {evaluation['estimated_effort']}")

        if evaluation['required_improvements']:
            print(f"\n必要な改善:")
            for improvement in evaluation['required_improvements']:
                print(f"  • {improvement}")

        if evaluation['recommendations']:
            print(f"\n推奨事項:")
            for rec in evaluation['recommendations']:
                print(f"  • {rec}")

    except Exception as e:
        print(f"❌ 検証エラー: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== 予測精度検証システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_prediction_accuracy_validator())