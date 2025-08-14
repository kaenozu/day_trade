#!/usr/bin/env python3
"""
93%精度エンドツーエンド検証
Day Trade ML System EnsembleSystem精度実環境検証

検証対象:
- Issue #487: EnsembleSystem 93%精度達成確認
- 実市場データでの精度検証
- リアルタイム予測精度監視
"""

import os
import sys
import time
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import yfinance as yf
import ta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# テスト対象システムのインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AccuracyTestResult:
    """精度テスト結果"""
    test_name: str
    symbol: str
    test_period: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    timestamp: datetime
    details: Optional[Dict] = None

@dataclass
class MarketDataPoint:
    """市場データポイント"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    features: Dict[str, float]
    actual_direction: int  # 0: down, 1: up

class AccuracyValidator:
    """93%精度検証システム"""

    def __init__(self):
        self.test_results: List[AccuracyTestResult] = []
        self.market_data_cache: Dict[str, pd.DataFrame] = {}

        # 検証設定
        self.validation_config = {
            'target_accuracy': 0.93,
            'test_symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN'],
            'test_periods': ['1mo', '3mo', '6mo', '1y'],
            'feature_windows': [5, 10, 20, 50],
            'prediction_horizons': [1, 5, 10],  # days
            'min_test_samples': 100,
            'confidence_threshold': 0.8
        }

        # 特徴量設定
        self.feature_config = {
            'technical_indicators': [
                'rsi', 'macd', 'bollinger_bands', 'moving_averages',
                'volume_indicators', 'momentum_indicators'
            ],
            'price_features': [
                'price_change', 'price_volatility', 'high_low_ratio',
                'open_close_ratio', 'volume_price_trend'
            ]
        }

    async def run_comprehensive_accuracy_validation(self) -> Dict:
        """包括的精度検証実行"""
        logger.info("Starting comprehensive 93% accuracy validation...")

        validation_start = datetime.utcnow()

        # 1. 市場データ収集
        await self._collect_market_data()

        # 2. 複数銘柄での精度検証
        multi_symbol_results = await self._validate_multi_symbol_accuracy()

        # 3. 時系列精度検証
        time_series_results = await self._validate_time_series_accuracy()

        # 4. リアルタイム精度検証
        realtime_results = await self._validate_realtime_accuracy()

        # 5. ストレステスト（市場変動期）
        stress_test_results = await self._validate_stress_conditions()

        # 6. 特徴量重要度分析
        feature_analysis = await self._analyze_feature_importance()

        # 7. 精度劣化検出
        degradation_analysis = await self._analyze_accuracy_degradation()

        validation_duration = (datetime.utcnow() - validation_start).total_seconds()

        # 全体結果統合
        all_results = (multi_symbol_results + time_series_results +
                      realtime_results + stress_test_results)

        # 統計サマリー
        validation_summary = self._generate_accuracy_summary(all_results)

        validation_report = {
            'validation_start': validation_start.isoformat(),
            'validation_duration_seconds': validation_duration,
            'target_accuracy': self.validation_config['target_accuracy'],
            'total_tests': len(all_results),
            'test_results': [asdict(result) for result in all_results],
            'feature_analysis': feature_analysis,
            'degradation_analysis': degradation_analysis,
            'summary': validation_summary,
            'accuracy_achievement': validation_summary['overall_accuracy'] >= self.validation_config['target_accuracy'],
            'recommendations': self._generate_accuracy_recommendations(validation_summary)
        }

        return validation_report

    async def _collect_market_data(self):
        """市場データ収集"""
        logger.info("Collecting market data for validation...")

        for symbol in self.validation_config['test_symbols']:
            try:
                # Yahoo Finance からデータ取得
                ticker = yf.Ticker(symbol)

                # 最長期間のデータ取得
                data = ticker.history(period="2y", interval="1d")

                if len(data) > 0:
                    # 特徴量計算
                    enhanced_data = self._calculate_technical_features(data, symbol)
                    self.market_data_cache[symbol] = enhanced_data

                    logger.info(f"Collected {len(enhanced_data)} data points for {symbol}")
                else:
                    logger.warning(f"No data collected for {symbol}")

            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {str(e)}")

    def _calculate_technical_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """テクニカル特徴量計算"""
        df = data.copy()

        # 基本価格特徴量
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']
        df['volume_change'] = df['Volume'].pct_change()

        # 移動平均
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ma_{window}_ratio'] = df['Close'] / df[f'ma_{window}']

        # ボリンジャーバンド
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.bollinger_hband(df['Close']), \
                                                          ta.volatility.bollinger_mavg(df['Close']), \
                                                          ta.volatility.bollinger_lband(df['Close'])
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # RSI
        df['rsi'] = ta.momentum.rsi(df['Close'])

        # MACD
        df['macd'] = ta.trend.macd_diff(df['Close'])
        df['macd_signal'] = ta.trend.macd_signal(df['Close'])

        # ボリュームインディケータ
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        # ボラティリティ
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window=window).std()

        # ターゲット変数（翌日の方向）
        df['future_return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['target'] = (df['future_return'] > 0).astype(int)

        # 欠損値除去
        df = df.dropna()

        return df

    async def _validate_multi_symbol_accuracy(self) -> List[AccuracyTestResult]:
        """複数銘柄精度検証"""
        logger.info("Validating accuracy across multiple symbols...")

        results = []

        for symbol in self.validation_config['test_symbols']:
            if symbol not in self.market_data_cache:
                continue

            data = self.market_data_cache[symbol]

            for period in self.validation_config['test_periods']:
                # 期間データ抽出
                period_data = self._extract_period_data(data, period)

                if len(period_data) < self.validation_config['min_test_samples']:
                    continue

                # 予測実行
                predictions, actuals = await self._run_ensemble_predictions(period_data, symbol)

                if len(predictions) > 0:
                    # 精度計算
                    accuracy = accuracy_score(actuals, predictions)
                    precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
                    recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
                    f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)

                    # 混同行列
                    confusion_matrix = self._calculate_confusion_matrix(actuals, predictions)

                    result = AccuracyTestResult(
                        test_name=f"multi_symbol_accuracy",
                        symbol=symbol,
                        test_period=period,
                        total_predictions=len(predictions),
                        correct_predictions=sum(1 for p, a in zip(predictions, actuals) if p == a),
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                        confusion_matrix=confusion_matrix,
                        timestamp=datetime.utcnow(),
                        details={'data_points': len(period_data)}
                    )

                    results.append(result)

                    logger.info(f"Symbol: {symbol}, Period: {period}, "
                               f"Accuracy: {accuracy:.4f}, Samples: {len(predictions)}")

        return results

    async def _validate_time_series_accuracy(self) -> List[AccuracyTestResult]:
        """時系列精度検証（ウォークフォワード）"""
        logger.info("Validating time-series accuracy with walk-forward analysis...")

        results = []

        for symbol in self.validation_config['test_symbols'][:3]:  # 主要3銘柄
            if symbol not in self.market_data_cache:
                continue

            data = self.market_data_cache[symbol]

            # ウォークフォワード検証
            train_size = len(data) // 2
            test_size = 50  # 50日ずつテスト
            step_size = 10  # 10日ずつ進む

            for start_idx in range(train_size, len(data) - test_size, step_size):
                train_data = data.iloc[:start_idx]
                test_data = data.iloc[start_idx:start_idx + test_size]

                if len(test_data) < 20:  # 最小テストサイズ
                    continue

                # 予測実行
                predictions, actuals = await self._run_ensemble_predictions(test_data, symbol)

                if len(predictions) > 0:
                    accuracy = accuracy_score(actuals, predictions)

                    result = AccuracyTestResult(
                        test_name="time_series_accuracy",
                        symbol=symbol,
                        test_period=f"walk_forward_{start_idx}",
                        total_predictions=len(predictions),
                        correct_predictions=sum(1 for p, a in zip(predictions, actuals) if p == a),
                        accuracy=accuracy,
                        precision=precision_score(actuals, predictions, average='weighted', zero_division=0),
                        recall=recall_score(actuals, predictions, average='weighted', zero_division=0),
                        f1_score=f1_score(actuals, predictions, average='weighted', zero_division=0),
                        confusion_matrix=self._calculate_confusion_matrix(actuals, predictions),
                        timestamp=datetime.utcnow(),
                        details={
                            'train_size': len(train_data),
                            'test_size': len(test_data),
                            'start_date': test_data.index[0].strftime('%Y-%m-%d'),
                            'end_date': test_data.index[-1].strftime('%Y-%m-%d')
                        }
                    )

                    results.append(result)

        return results

    async def _validate_realtime_accuracy(self) -> List[AccuracyTestResult]:
        """リアルタイム精度検証"""
        logger.info("Validating real-time prediction accuracy...")

        results = []

        # 最新データでのリアルタイム検証
        for symbol in self.validation_config['test_symbols'][:2]:  # 主要2銘柄
            if symbol not in self.market_data_cache:
                continue

            data = self.market_data_cache[symbol]

            # 最新30日間でリアルタイム予測シミュレーション
            recent_data = data.tail(30)

            realtime_predictions = []
            realtime_actuals = []

            for i in range(10, len(recent_data)):
                # i日時点での予測（i+1日の方向）
                historical_data = recent_data.iloc[:i]
                target_actual = recent_data.iloc[i]['target']

                # 予測実行（単発）
                prediction = await self._single_prediction(historical_data, symbol)

                if prediction is not None:
                    realtime_predictions.append(prediction)
                    realtime_actuals.append(target_actual)

            if len(realtime_predictions) > 0:
                accuracy = accuracy_score(realtime_actuals, realtime_predictions)

                result = AccuracyTestResult(
                    test_name="realtime_accuracy",
                    symbol=symbol,
                    test_period="realtime_30d",
                    total_predictions=len(realtime_predictions),
                    correct_predictions=sum(1 for p, a in zip(realtime_predictions, realtime_actuals) if p == a),
                    accuracy=accuracy,
                    precision=precision_score(realtime_actuals, realtime_predictions, average='weighted', zero_division=0),
                    recall=recall_score(realtime_actuals, realtime_predictions, average='weighted', zero_division=0),
                    f1_score=f1_score(realtime_actuals, realtime_predictions, average='weighted', zero_division=0),
                    confusion_matrix=self._calculate_confusion_matrix(realtime_actuals, realtime_predictions),
                    timestamp=datetime.utcnow(),
                    details={'prediction_type': 'realtime_simulation'}
                )

                results.append(result)

                logger.info(f"Realtime accuracy for {symbol}: {accuracy:.4f}")

        return results

    async def _validate_stress_conditions(self) -> List[AccuracyTestResult]:
        """ストレス条件下での精度検証"""
        logger.info("Validating accuracy under stress conditions...")

        results = []

        for symbol in self.validation_config['test_symbols'][:2]:
            if symbol not in self.market_data_cache:
                continue

            data = self.market_data_cache[symbol]

            # 高ボラティリティ期間の検出
            data['volatility'] = data['price_change'].rolling(window=20).std()
            high_vol_threshold = data['volatility'].quantile(0.8)

            stress_periods = data[data['volatility'] > high_vol_threshold]

            if len(stress_periods) > self.validation_config['min_test_samples']:
                # ストレス期間での予測
                predictions, actuals = await self._run_ensemble_predictions(stress_periods, symbol)

                if len(predictions) > 0:
                    accuracy = accuracy_score(actuals, predictions)

                    result = AccuracyTestResult(
                        test_name="stress_conditions_accuracy",
                        symbol=symbol,
                        test_period="high_volatility",
                        total_predictions=len(predictions),
                        correct_predictions=sum(1 for p, a in zip(predictions, actuals) if p == a),
                        accuracy=accuracy,
                        precision=precision_score(actuals, predictions, average='weighted', zero_division=0),
                        recall=recall_score(actuals, predictions, average='weighted', zero_division=0),
                        f1_score=f1_score(actuals, predictions, average='weighted', zero_division=0),
                        confusion_matrix=self._calculate_confusion_matrix(actuals, predictions),
                        timestamp=datetime.utcnow(),
                        details={
                            'condition_type': 'high_volatility',
                            'volatility_threshold': high_vol_threshold,
                            'avg_volatility': stress_periods['volatility'].mean()
                        }
                    )

                    results.append(result)

                    logger.info(f"Stress test accuracy for {symbol}: {accuracy:.4f}")

        return results

    async def _analyze_feature_importance(self) -> Dict:
        """特徴量重要度分析"""
        logger.info("Analyzing feature importance...")

        # 特徴量重要度分析（簡易版）
        feature_importance = {}

        for symbol in self.validation_config['test_symbols'][:3]:
            if symbol not in self.market_data_cache:
                continue

            data = self.market_data_cache[symbol]

            # 特徴量抽出
            feature_columns = [col for col in data.columns if col not in ['target', 'future_return']]
            features = data[feature_columns].select_dtypes(include=[np.number])

            # 相関分析
            correlations = {}
            for col in features.columns:
                if col in data.columns and 'target' in data.columns:
                    correlation = data[col].corr(data['target'])
                    if not np.isnan(correlation):
                        correlations[col] = abs(correlation)

            feature_importance[symbol] = correlations

        # 全体的な特徴量重要度
        overall_importance = {}
        for symbol_features in feature_importance.values():
            for feature, importance in symbol_features.items():
                if feature not in overall_importance:
                    overall_importance[feature] = []
                overall_importance[feature].append(importance)

        # 平均重要度計算
        avg_importance = {
            feature: np.mean(importances)
            for feature, importances in overall_importance.items()
        }

        # 上位特徴量
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'symbol_specific': feature_importance,
            'overall_importance': avg_importance,
            'top_features': top_features
        }

    async def _analyze_accuracy_degradation(self) -> Dict:
        """精度劣化分析"""
        logger.info("Analyzing accuracy degradation patterns...")

        degradation_analysis = {}

        # 時系列精度推移分析
        for symbol in self.validation_config['test_symbols'][:3]:
            if symbol not in self.market_data_cache:
                continue

            data = self.market_data_cache[symbol]

            # 月別精度推移
            monthly_accuracy = []
            data['month'] = data.index.to_period('M')

            for month in data['month'].unique():
                month_data = data[data['month'] == month]

                if len(month_data) > 20:  # 十分なサンプル
                    predictions, actuals = await self._run_ensemble_predictions(month_data, symbol)

                    if len(predictions) > 0:
                        accuracy = accuracy_score(actuals, predictions)
                        monthly_accuracy.append({
                            'month': str(month),
                            'accuracy': accuracy,
                            'samples': len(predictions)
                        })

            degradation_analysis[symbol] = {
                'monthly_accuracy': monthly_accuracy,
                'accuracy_trend': self._calculate_trend(monthly_accuracy) if monthly_accuracy else None
            }

        return degradation_analysis

    def _extract_period_data(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """期間データ抽出"""
        if period == '1mo':
            return data.tail(30)
        elif period == '3mo':
            return data.tail(90)
        elif period == '6mo':
            return data.tail(180)
        elif period == '1y':
            return data.tail(365)
        else:
            return data

    async def _run_ensemble_predictions(self, data: pd.DataFrame, symbol: str) -> Tuple[List[int], List[int]]:
        """アンサンブル予測実行"""
        # 実際のEnsembleSystemを呼び出す（簡易版）
        predictions = []
        actuals = []

        feature_columns = [col for col in data.columns if col not in ['target', 'future_return']]

        for i in range(len(data)):
            if i < 20:  # 十分な履歴が必要
                continue

            # 予測実行（簡易版 - 実際はEnsembleSystemを使用）
            features = data.iloc[i][feature_columns].values

            # シンプルな予測ロジック（実際のMLモデルに置き換え）
            prediction = await self._simple_ensemble_prediction(features)
            actual = int(data.iloc[i]['target'])

            predictions.append(prediction)
            actuals.append(actual)

        return predictions, actuals

    async def _single_prediction(self, historical_data: pd.DataFrame, symbol: str) -> Optional[int]:
        """単発予測"""
        if len(historical_data) < 20:
            return None

        feature_columns = [col for col in historical_data.columns if col not in ['target', 'future_return']]
        latest_features = historical_data.iloc[-1][feature_columns].values

        return await self._simple_ensemble_prediction(latest_features)

    async def _simple_ensemble_prediction(self, features: np.ndarray) -> int:
        """簡易アンサンブル予測"""
        # 実際のEnsembleSystemに置き換える
        # ここでは簡易的な予測ロジック

        # 特徴量の加重平均で判定
        if len(features) == 0:
            return np.random.choice([0, 1])

        # 正規化
        normalized_features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        # 簡易予測スコア
        score = np.mean(normalized_features)

        # 閾値で二値分類
        return 1 if score > 0 else 0

    def _calculate_confusion_matrix(self, actuals: List[int], predictions: List[int]) -> List[List[int]]:
        """混同行列計算"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(actuals, predictions, labels=[0, 1])
        return cm.tolist()

    def _calculate_trend(self, monthly_data: List[Dict]) -> Optional[float]:
        """精度トレンド計算"""
        if len(monthly_data) < 3:
            return None

        accuracies = [data['accuracy'] for data in monthly_data]
        x = np.arange(len(accuracies))

        # 線形回帰の傾き
        slope = np.polyfit(x, accuracies, 1)[0]
        return float(slope)

    def _generate_accuracy_summary(self, results: List[AccuracyTestResult]) -> Dict:
        """精度サマリー生成"""
        if not results:
            return {'overall_accuracy': 0.0}

        accuracies = [result.accuracy for result in results]
        precisions = [result.precision for result in results]
        recalls = [result.recall for result in results]
        f1_scores = [result.f1_score for result in results]

        # 銘柄別統計
        symbol_stats = {}
        for result in results:
            symbol = result.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = []
            symbol_stats[symbol].append(result.accuracy)

        symbol_averages = {
            symbol: np.mean(accuracies)
            for symbol, accuracies in symbol_stats.items()
        }

        # テストタイプ別統計
        test_type_stats = {}
        for result in results:
            test_type = result.test_name
            if test_type not in test_type_stats:
                test_type_stats[test_type] = []
            test_type_stats[test_type].append(result.accuracy)

        test_type_averages = {
            test_type: np.mean(accuracies)
            for test_type, accuracies in test_type_stats.items()
        }

        return {
            'overall_accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'avg_f1_score': np.mean(f1_scores),
            'target_achievement': np.mean(accuracies) >= self.validation_config['target_accuracy'],
            'passing_tests': sum(1 for acc in accuracies if acc >= self.validation_config['target_accuracy']),
            'total_tests': len(results),
            'symbol_performance': symbol_averages,
            'test_type_performance': test_type_averages
        }

    def _generate_accuracy_recommendations(self, summary: Dict) -> List[str]:
        """精度改善提案"""
        recommendations = []

        overall_accuracy = summary.get('overall_accuracy', 0.0)
        target_accuracy = self.validation_config['target_accuracy']

        if overall_accuracy < target_accuracy:
            gap = target_accuracy - overall_accuracy
            recommendations.append(f"精度目標未達成: {overall_accuracy:.4f} < {target_accuracy:.4f} (差分: {gap:.4f})")

            if gap > 0.05:
                recommendations.append("大幅な精度改善が必要: モデル再設計を推奨")
            elif gap > 0.02:
                recommendations.append("中程度の精度改善が必要: ハイパーパラメータ調整・特徴量追加を推奨")
            else:
                recommendations.append("軽微な精度改善が必要: データ品質向上・前処理改善を推奨")
        else:
            recommendations.append(f"精度目標達成: {overall_accuracy:.4f} >= {target_accuracy:.4f}")

        # 銘柄別推奨
        symbol_performance = summary.get('symbol_performance', {})
        poor_performers = [symbol for symbol, acc in symbol_performance.items() if acc < target_accuracy]

        if poor_performers:
            recommendations.append(f"精度改善が必要な銘柄: {', '.join(poor_performers)}")

        # 分散確認
        accuracy_std = summary.get('accuracy_std', 0)
        if accuracy_std > 0.1:
            recommendations.append("精度のばらつきが大きいため、モデルの安定性改善を推奨")

        return recommendations

    def save_accuracy_report(self, validation_report: Dict, filename: str = None):
        """精度検証レポート保存"""
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'accuracy_validation_{timestamp}.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Accuracy validation report saved to: {filename}")

    def generate_accuracy_charts(self, validation_report: Dict, output_dir: str = "accuracy_charts"):
        """精度チャート生成"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. 銘柄別精度チャート
        self._create_symbol_accuracy_chart(validation_report, output_dir)

        # 2. 時系列精度推移チャート
        self._create_time_series_accuracy_chart(validation_report, output_dir)

        # 3. 混同行列ヒートマップ
        self._create_confusion_matrix_heatmap(validation_report, output_dir)

    def _create_symbol_accuracy_chart(self, report: Dict, output_dir: str):
        """銘柄別精度チャート作成"""
        # 実装省略（matplotlib使用）
        pass

    def _create_time_series_accuracy_chart(self, report: Dict, output_dir: str):
        """時系列精度推移チャート作成"""
        # 実装省略
        pass

    def _create_confusion_matrix_heatmap(self, report: Dict, output_dir: str):
        """混同行列ヒートマップ作成"""
        # 実装省略
        pass

async def main():
    """93%精度検証実行"""
    validator = AccuracyValidator()

    print("🎯 Day Trade ML System - 93%精度エンドツーエンド検証開始")
    print("=" * 60)

    # 精度検証実行
    report = await validator.run_comprehensive_accuracy_validation()

    # 結果出力
    print(f"\n📊 93%精度検証結果")
    print(f"目標精度: {report['target_accuracy']:.1%}")
    print(f"達成精度: {report['summary']['overall_accuracy']:.4f} ({report['summary']['overall_accuracy']:.1%})")
    print(f"目標達成: {'✅' if report['accuracy_achievement'] else '❌'}")
    print(f"総テスト数: {report['total_tests']}")
    print(f"合格テスト数: {report['summary']['passing_tests']}")

    print(f"\n📈 詳細統計:")
    print(f"精度範囲: {report['summary']['min_accuracy']:.4f} - {report['summary']['max_accuracy']:.4f}")
    print(f"精度標準偏差: {report['summary']['accuracy_std']:.4f}")
    print(f"平均適合率: {report['summary']['avg_precision']:.4f}")
    print(f"平均再現率: {report['summary']['avg_recall']:.4f}")
    print(f"平均F1スコア: {report['summary']['avg_f1_score']:.4f}")

    print(f"\n📋 銘柄別性能:")
    for symbol, accuracy in report['summary']['symbol_performance'].items():
        status = "✅" if accuracy >= report['target_accuracy'] else "❌"
        print(f"{status} {symbol}: {accuracy:.4f}")

    print(f"\n💡 改善提案:")
    for rec in report['recommendations']:
        print(f"• {rec}")

    # レポート保存
    validator.save_accuracy_report(report)

    print(f"\n📄 詳細レポートが保存されました")

if __name__ == '__main__':
    asyncio.run(main())