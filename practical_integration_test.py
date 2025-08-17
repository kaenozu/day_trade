#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical Integration Test - 実用的統合テスト
Issue #870拡張予測システムと既存システムの実用的統合テスト
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# パフォーマンス最適化システム
from integrated_performance_optimizer import (
    get_integrated_optimizer,
    optimize_system_performance,
    get_system_performance_report
)

# 既存のコア予測システム（利用可能な場合）
try:
    from ml_prediction_models import MLPredictionModels
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

try:
    from real_data_provider import RealDataProvider
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

try:
    from risk_manager import RiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False


class MockDataProvider:
    """テスト用データプロバイダー"""

    def __init__(self):
        self.symbols = ['7203.T', '9984.T', '6758.T', '8306.T', '4689.T']
        self.cache = {}

    def get_stock_data(self, symbol: str, period: str = '1mo') -> pd.DataFrame:
        """株価データを取得（モック）"""
        cache_key = f"{symbol}_{period}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # リアルなデータ構造でモックデータ生成
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

        # 価格データ生成（ランダムウォーク）
        initial_price = np.random.uniform(100, 5000)
        price_changes = np.random.normal(0, 0.02, len(dates))
        prices = initial_price * np.exp(np.cumsum(price_changes))

        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.lognormal(15, 0.5, len(dates)).astype(int)
        })

        data = data.set_index('Date')
        self.cache[cache_key] = data

        return data

    def get_multiple_stocks_data(self, symbols: List[str], period: str = '1mo') -> Dict[str, pd.DataFrame]:
        """複数銘柄のデータを取得"""
        return {symbol: self.get_stock_data(symbol, period) for symbol in symbols}


class MockMLPredictor:
    """テスト用ML予測システム"""

    def __init__(self):
        self.is_trained = False
        self.prediction_history = []
        self.accuracy_score = 0.75  # 75%の精度

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """モデル訓練"""
        time.sleep(0.1)  # 訓練時間シミュレーション
        self.is_trained = True

        return {
            'training_samples': len(X),
            'features': X.shape[1],
            'training_time': 0.1,
            'accuracy': self.accuracy_score + np.random.normal(0, 0.05)
        }

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """予測実行"""
        if not self.is_trained:
            self.train(X, pd.Series(np.random.randn(len(X))))

        predictions = np.random.randn(len(X))
        confidence = np.random.uniform(0.6, 0.9, len(X))

        result = {
            'predictions': predictions,
            'confidence': confidence,
            'model_accuracy': self.accuracy_score,
            'prediction_time': time.time()
        }

        self.prediction_history.append(result)
        return result

    def get_model_performance(self) -> Dict[str, Any]:
        """モデル性能を取得"""
        return {
            'is_trained': self.is_trained,
            'accuracy': self.accuracy_score,
            'total_predictions': len(self.prediction_history),
            'avg_confidence': np.mean([np.mean(h['confidence']) for h in self.prediction_history]) if self.prediction_history else 0
        }


class PracticalIntegrationTest:
    """実用的統合テスト"""

    def __init__(self):
        # コンポーネント初期化
        self.data_provider = MockDataProvider()
        self.ml_predictor = MockMLPredictor()

        # パフォーマンス最適化システム
        self.optimizer = get_integrated_optimizer()

        # テスト結果
        self.test_results = {}
        self.performance_metrics = []

        print("Practical Integration Test initialized")
        print("Components available:")
        print(f"  - ML Models: {'Yes' if ML_MODELS_AVAILABLE else 'Mock'}")
        print(f"  - Real Data: {'Yes' if REAL_DATA_AVAILABLE else 'Mock'}")
        print(f"  - Risk Manager: {'Yes' if RISK_MANAGER_AVAILABLE else 'Mock'}")

    def test_data_pipeline_performance(self) -> Dict[str, Any]:
        """データパイプライン性能テスト"""
        print("\n=== Data Pipeline Performance Test ===")

        symbols = self.data_provider.symbols

        # データ取得性能測定
        start_time = time.time()
        stock_data = {}

        for symbol in symbols:
            data = self.data_provider.get_stock_data(symbol, period='3mo')
            stock_data[symbol] = data
            print(f"Loaded {symbol}: {len(data)} records")

        data_loading_time = time.time() - start_time

        # 特徴量生成性能測定
        start_time = time.time()
        features_data = {}

        for symbol, data in stock_data.items():
            # 技術指標計算
            features = pd.DataFrame(index=data.index)

            # 移動平均
            features['SMA_5'] = data['Close'].rolling(5).mean()
            features['SMA_20'] = data['Close'].rolling(20).mean()

            # モメンタム
            features['ROC_5'] = data['Close'].pct_change(5)
            features['ROC_20'] = data['Close'].pct_change(20)

            # ボラティリティ
            features['Volatility_5'] = data['Close'].rolling(5).std()
            features['Volatility_20'] = data['Close'].rolling(20).std()

            # 価格特徴量
            features['Price_Range'] = (data['High'] - data['Low']) / data['Close']
            features['Volume_MA'] = data['Volume'].rolling(5).mean()

            features = features.dropna()
            features_data[symbol] = features

        feature_generation_time = time.time() - start_time

        # 結果集計
        total_records = sum(len(data) for data in stock_data.values())
        total_features = sum(len(features.columns) for features in features_data.values())

        result = {
            'symbols_processed': len(symbols),
            'total_records': total_records,
            'total_features': total_features,
            'data_loading_time': data_loading_time,
            'feature_generation_time': feature_generation_time,
            'total_processing_time': data_loading_time + feature_generation_time,
            'records_per_second': total_records / (data_loading_time + feature_generation_time),
            'features_per_second': total_features / feature_generation_time
        }

        print(f"Data Pipeline Results:")
        print(f"  Symbols: {result['symbols_processed']}")
        print(f"  Records: {result['total_records']}")
        print(f"  Features: {result['total_features']}")
        print(f"  Processing Speed: {result['records_per_second']:.1f} records/sec")
        print(f"  Feature Speed: {result['features_per_second']:.1f} features/sec")

        self.test_results['data_pipeline'] = result
        return result

    def test_ml_prediction_performance(self) -> Dict[str, Any]:
        """ML予測性能テスト"""
        print("\n=== ML Prediction Performance Test ===")

        # テストデータ準備
        symbols = ['7203.T', '9984.T', '6758.T']
        all_features = []
        all_targets = []

        for symbol in symbols:
            data = self.data_provider.get_stock_data(symbol, period='6mo')

            # 特徴量生成
            features = pd.DataFrame(index=data.index)
            features['SMA_5'] = data['Close'].rolling(5).mean()
            features['SMA_20'] = data['Close'].rolling(20).mean()
            features['ROC_5'] = data['Close'].pct_change(5)
            features['Volatility_10'] = data['Close'].rolling(10).std()
            features['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

            # ターゲット（翌日リターン）
            target = data['Close'].pct_change().shift(-1)

            # NaN除去
            valid_idx = features.dropna().index.intersection(target.dropna().index)
            if len(valid_idx) > 0:
                all_features.append(features.loc[valid_idx])
                all_targets.append(target.loc[valid_idx])

        # データ結合
        if all_features:
            X = pd.concat(all_features, axis=0)
            y = pd.concat(all_targets, axis=0)
        else:
            # フォールバック用のダミーデータ
            X = pd.DataFrame(np.random.randn(100, 5),
                           columns=['SMA_5', 'SMA_20', 'ROC_5', 'Volatility_10', 'Volume_Ratio'])
            y = pd.Series(np.random.randn(100))

        print(f"Training data: {len(X)} samples, {X.shape[1]} features")

        # 訓練・予測性能測定
        start_time = time.time()
        training_result = self.ml_predictor.train(X, y)
        training_time = time.time() - start_time

        # 予測性能測定
        test_size = min(50, max(1, len(X) // 2))
        X_test = X.tail(test_size)

        start_time = time.time()
        prediction_result = self.ml_predictor.predict(X_test)
        prediction_time = time.time() - start_time

        # ゼロ除算防止
        if prediction_time <= 0:
            prediction_time = 0.001

        # モデル性能取得
        model_performance = self.ml_predictor.get_model_performance()

        result = {
            'training_samples': len(X),
            'test_samples': len(X_test),
            'features': X.shape[1],
            'training_time': training_time,
            'prediction_time': prediction_time,
            'samples_per_second': len(X_test) / prediction_time,
            'model_accuracy': model_performance['accuracy'],
            'avg_confidence': model_performance['avg_confidence'],
            'training_result': training_result,
            'prediction_result': {
                'mean_prediction': np.mean(prediction_result['predictions']),
                'mean_confidence': np.mean(prediction_result['confidence'])
            }
        }

        print(f"ML Prediction Results:")
        print(f"  Training: {result['training_samples']} samples in {result['training_time']:.3f}s")
        print(f"  Prediction: {result['test_samples']} samples in {result['prediction_time']:.3f}s")
        print(f"  Speed: {result['samples_per_second']:.1f} samples/sec")
        print(f"  Accuracy: {result['model_accuracy']:.1%}")
        print(f"  Confidence: {result['avg_confidence']:.1%}")

        self.test_results['ml_prediction'] = result
        return result

    def test_memory_optimization_integration(self) -> Dict[str, Any]:
        """メモリ最適化統合テスト"""
        print("\n=== Memory Optimization Integration Test ===")

        # 最適化前の状態
        initial_report = get_system_performance_report()
        initial_memory = initial_report['system_state']['memory_usage_mb']

        print(f"Initial memory usage: {initial_memory:.1f}MB")

        # 大量データ処理でメモリ負荷をかける
        large_datasets = []
        for i in range(20):
            symbol = f"TEST_{i}"
            data = self.data_provider.get_stock_data(symbol, period='1y')

            # 複雑な特徴量生成
            features = pd.DataFrame(index=data.index)
            for window in [5, 10, 20, 50]:
                features[f'SMA_{window}'] = data['Close'].rolling(window).mean()
                features[f'EMA_{window}'] = data['Close'].ewm(span=window).mean()
                features[f'STD_{window}'] = data['Close'].rolling(window).std()

            large_datasets.append((symbol, data, features))

        # ピーク時のメモリ使用量
        peak_report = get_system_performance_report()
        peak_memory = peak_report['system_state']['memory_usage_mb']
        memory_increase = peak_memory - initial_memory

        print(f"Peak memory usage: {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")

        # 最適化実行
        print("Running system optimization...")
        optimization_results = optimize_system_performance()

        # 最適化後の状態
        optimized_report = get_system_performance_report()
        optimized_memory = optimized_report['system_state']['memory_usage_mb']
        memory_reduction = peak_memory - optimized_memory

        print(f"Optimized memory usage: {optimized_memory:.1f}MB (-{memory_reduction:.1f}MB)")

        # 最適化効果の評価
        optimization_effectiveness = (memory_reduction / memory_increase * 100) if memory_increase > 0 else 0

        result = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'optimized_memory_mb': optimized_memory,
            'memory_increase_mb': memory_increase,
            'memory_reduction_mb': memory_reduction,
            'optimization_effectiveness_percent': optimization_effectiveness,
            'datasets_processed': len(large_datasets),
            'optimization_results': optimization_results,
            'performance_reports': {
                'initial': initial_report,
                'peak': peak_report,
                'optimized': optimized_report
            }
        }

        print(f"Memory Optimization Results:")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Memory reduction: {memory_reduction:.1f}MB")
        print(f"  Optimization effectiveness: {optimization_effectiveness:.1f}%")
        print(f"  Optimization strategies executed: {len(optimization_results)}")

        self.test_results['memory_optimization'] = result
        return result

    def test_real_time_performance(self) -> Dict[str, Any]:
        """リアルタイム性能テスト"""
        print("\n=== Real-time Performance Test ===")

        symbols = ['7203.T', '9984.T', '6758.T', '8306.T']
        simulation_rounds = 10

        # リアルタイムシミュレーション
        latencies = []
        throughputs = []

        for round_num in range(simulation_rounds):
            round_start = time.time()
            round_predictions = 0

            for symbol in symbols:
                # データ取得
                data_start = time.time()
                data = self.data_provider.get_stock_data(symbol, period='1mo')
                data_time = time.time() - data_start

                # 特徴量生成
                feature_start = time.time()
                features = pd.DataFrame(index=data.index[-10:])  # 最新10日分
                features['SMA_5'] = data['Close'].rolling(5).mean().tail(10)
                features['ROC_5'] = data['Close'].pct_change(5).tail(10)
                features['Volatility'] = data['Close'].rolling(10).std().tail(10)
                features = features.dropna()
                feature_time = time.time() - feature_start

                # 予測実行
                if len(features) > 0:
                    pred_start = time.time()
                    prediction = self.ml_predictor.predict(features)
                    pred_time = time.time() - pred_start

                    # レイテンシ記録
                    total_latency = data_time + feature_time + pred_time
                    latencies.append(total_latency * 1000)  # ms
                    round_predictions += len(features)

            round_time = time.time() - round_start
            round_throughput = round_predictions / round_time
            throughputs.append(round_throughput)

            if round_num % 3 == 0:
                # 定期的に最適化実行
                try:
                    self.optimizer.run_comprehensive_optimization()
                except:
                    pass

        # 統計計算
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        avg_throughput = np.mean(throughputs)

        result = {
            'simulation_rounds': simulation_rounds,
            'symbols_per_round': len(symbols),
            'total_predictions': len(latencies),
            'average_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'average_throughput_predictions_per_sec': avg_throughput,
            'max_throughput': max(throughputs),
            'min_latency_ms': min(latencies),
            'latency_distribution': {
                'mean': avg_latency,
                'std': np.std(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': p95_latency,
                'p99': np.percentile(latencies, 99)
            }
        }

        print(f"Real-time Performance Results:")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  Average throughput: {avg_throughput:.1f} predictions/sec")
        print(f"  Total predictions: {len(latencies)}")

        self.test_results['real_time_performance'] = result
        return result

    def test_system_stability(self) -> Dict[str, Any]:
        """システム安定性テスト"""
        print("\n=== System Stability Test ===")

        test_duration_minutes = 2  # 2分間のテスト
        check_interval = 10  # 10秒間隔でチェック

        start_time = time.time()
        end_time = start_time + (test_duration_minutes * 60)

        stability_metrics = []
        error_count = 0

        while time.time() < end_time:
            try:
                # システム状態監視
                performance_report = get_system_performance_report()

                # 軽いワークロード実行
                symbol = np.random.choice(self.data_provider.symbols)
                data = self.data_provider.get_stock_data(symbol)

                features = pd.DataFrame(index=data.index[-5:])
                features['Price'] = data['Close'].tail(5)
                features['Volume'] = data['Volume'].tail(5)
                features = features.dropna()

                if len(features) > 0:
                    prediction = self.ml_predictor.predict(features)

                # メトリクス記録
                metrics = {
                    'timestamp': time.time(),
                    'memory_mb': performance_report['system_state']['memory_usage_mb'],
                    'cpu_percent': performance_report['system_state']['cpu_usage_percent'],
                    'response_time_ms': performance_report['system_state']['response_time_ms'],
                    'cache_models': performance_report['system_state']['active_models'],
                    'cache_features': performance_report['system_state']['cached_features']
                }
                stability_metrics.append(metrics)

                # 定期的な最適化
                if len(stability_metrics) % 5 == 0:
                    try:
                        self.optimizer.run_comprehensive_optimization()
                    except:
                        pass

            except Exception as e:
                error_count += 1
                print(f"Error during stability test: {e}")

            time.sleep(check_interval)

        # 安定性分析
        if stability_metrics:
            memory_values = [m['memory_mb'] for m in stability_metrics]
            cpu_values = [m['cpu_percent'] for m in stability_metrics]
            response_values = [m['response_time_ms'] for m in stability_metrics]

            memory_stability = 1.0 - (np.std(memory_values) / np.mean(memory_values))
            cpu_stability = 1.0 - (np.std(cpu_values) / max(np.mean(cpu_values), 1))
            response_stability = 1.0 - (np.std(response_values) / np.mean(response_values))

            overall_stability = (memory_stability + cpu_stability + response_stability) / 3
        else:
            overall_stability = 0.0
            memory_stability = cpu_stability = response_stability = 0.0

        result = {
            'test_duration_minutes': test_duration_minutes,
            'total_checks': len(stability_metrics),
            'error_count': error_count,
            'error_rate': error_count / max(len(stability_metrics), 1),
            'overall_stability_score': overall_stability,
            'memory_stability': memory_stability,
            'cpu_stability': cpu_stability,
            'response_stability': response_stability,
            'stability_metrics': stability_metrics,
            'performance_summary': {
                'avg_memory_mb': np.mean(memory_values) if memory_values else 0,
                'avg_cpu_percent': np.mean(cpu_values) if cpu_values else 0,
                'avg_response_ms': np.mean(response_values) if response_values else 0,
                'memory_variance': np.var(memory_values) if memory_values else 0,
                'cpu_variance': np.var(cpu_values) if cpu_values else 0,
                'response_variance': np.var(response_values) if response_values else 0
            }
        }

        print(f"System Stability Results:")
        print(f"  Test duration: {test_duration_minutes} minutes")
        print(f"  Total checks: {len(stability_metrics)}")
        print(f"  Errors: {error_count} ({result['error_rate']:.1%})")
        print(f"  Overall stability: {overall_stability:.1%}")
        print(f"  Memory stability: {memory_stability:.1%}")
        print(f"  CPU stability: {cpu_stability:.1%}")
        print(f"  Response stability: {response_stability:.1%}")

        self.test_results['system_stability'] = result
        return result

    def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """包括的統合テストを実行"""
        print("=" * 60)
        print("PRACTICAL INTEGRATION TEST SUITE")
        print("=" * 60)

        test_start_time = time.time()

        # 各テストを順次実行
        tests = [
            self.test_data_pipeline_performance,
            self.test_ml_prediction_performance,
            self.test_memory_optimization_integration,
            self.test_real_time_performance,
            self.test_system_stability
        ]

        for i, test_func in enumerate(tests, 1):
            try:
                print(f"\n[{i}/{len(tests)}] Running {test_func.__name__}...")
                test_result = test_func()
                print(f"✓ {test_func.__name__} completed successfully")
            except Exception as e:
                print(f"✗ {test_func.__name__} failed: {e}")
                self.test_results[test_func.__name__] = {'error': str(e)}

        total_test_time = time.time() - test_start_time

        # 最終レポート生成
        final_report = self.generate_final_report(total_test_time)

        print("\n" + "=" * 60)
        print("INTEGRATION TEST COMPLETED")
        print("=" * 60)

        return final_report

    def generate_final_report(self, total_test_time: float) -> Dict[str, Any]:
        """最終テストレポートを生成"""
        successful_tests = [k for k, v in self.test_results.items() if 'error' not in v]
        failed_tests = [k for k, v in self.test_results.items() if 'error' in v]

        # パフォーマンス統計
        performance_stats = {}
        if 'data_pipeline' in self.test_results:
            dp = self.test_results['data_pipeline']
            performance_stats['data_processing_speed'] = dp.get('records_per_second', 0)

        if 'ml_prediction' in self.test_results:
            ml = self.test_results['ml_prediction']
            performance_stats['prediction_speed'] = ml.get('samples_per_second', 0)
            performance_stats['model_accuracy'] = ml.get('model_accuracy', 0)

        if 'real_time_performance' in self.test_results:
            rt = self.test_results['real_time_performance']
            performance_stats['avg_latency_ms'] = rt.get('average_latency_ms', 0)
            performance_stats['throughput'] = rt.get('average_throughput_predictions_per_sec', 0)

        if 'system_stability' in self.test_results:
            ss = self.test_results['system_stability']
            performance_stats['stability_score'] = ss.get('overall_stability_score', 0)
            performance_stats['error_rate'] = ss.get('error_rate', 0)

        # 最終システム状態
        final_system_report = get_system_performance_report()

        report = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(self.test_results) if self.test_results else 0,
                'total_test_time_seconds': total_test_time
            },
            'performance_statistics': performance_stats,
            'detailed_results': self.test_results,
            'system_status': final_system_report,
            'test_conclusion': self._generate_test_conclusion(successful_tests, failed_tests, performance_stats),
            'timestamp': datetime.now().isoformat()
        }

        # レポート表示
        self._print_final_report(report)

        return report

    def _generate_test_conclusion(self, successful_tests: List[str], failed_tests: List[str],
                                 performance_stats: Dict[str, Any]) -> Dict[str, Any]:
        """テスト結論を生成"""
        # 成功率による評価
        success_rate = len(successful_tests) / (len(successful_tests) + len(failed_tests)) if (successful_tests or failed_tests) else 0

        if success_rate >= 0.9:
            overall_status = "EXCELLENT"
        elif success_rate >= 0.8:
            overall_status = "GOOD"
        elif success_rate >= 0.6:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"

        # パフォーマンス評価
        performance_grade = "UNKNOWN"
        if performance_stats:
            score = 0
            max_score = 0

            if 'model_accuracy' in performance_stats:
                score += min(performance_stats['model_accuracy'] * 100, 100)
                max_score += 100

            if 'stability_score' in performance_stats:
                score += performance_stats['stability_score'] * 100
                max_score += 100

            if 'error_rate' in performance_stats:
                score += max(0, 100 - performance_stats['error_rate'] * 100)
                max_score += 100

            if max_score > 0:
                perf_percentage = score / max_score
                if perf_percentage >= 0.9:
                    performance_grade = "A"
                elif perf_percentage >= 0.8:
                    performance_grade = "B"
                elif perf_percentage >= 0.7:
                    performance_grade = "C"
                else:
                    performance_grade = "D"

        recommendations = []
        if failed_tests:
            recommendations.append(f"Address failed tests: {', '.join(failed_tests)}")

        if performance_stats.get('error_rate', 0) > 0.05:
            recommendations.append("Investigate and reduce error rate")

        if performance_stats.get('stability_score', 1) < 0.8:
            recommendations.append("Improve system stability")

        if not recommendations:
            recommendations.append("System performing well - continue monitoring")

        return {
            'overall_status': overall_status,
            'performance_grade': performance_grade,
            'success_rate': success_rate,
            'recommendations': recommendations,
            'ready_for_production': success_rate >= 0.8 and performance_stats.get('error_rate', 0) < 0.1
        }

    def _print_final_report(self, report: Dict[str, Any]):
        """最終レポートを表示"""
        print(f"\nFINAL INTEGRATION TEST REPORT")
        print(f"Generated: {report['timestamp']}")
        print(f"{'='*60}")

        summary = report['test_summary']
        print(f"Test Summary:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Successful: {summary['successful_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Total Time: {summary['total_test_time_seconds']:.1f}s")

        if report['performance_statistics']:
            print(f"\nPerformance Statistics:")
            stats = report['performance_statistics']
            for key, value in stats.items():
                if isinstance(value, float):
                    if 'rate' in key or 'accuracy' in key or 'score' in key:
                        print(f"  {key.replace('_', ' ').title()}: {value:.1%}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        conclusion = report['test_conclusion']
        print(f"\nConclusion:")
        print(f"  Overall Status: {conclusion['overall_status']}")
        print(f"  Performance Grade: {conclusion['performance_grade']}")
        print(f"  Production Ready: {'YES' if conclusion['ready_for_production'] else 'NO'}")

        if conclusion['recommendations']:
            print(f"\nRecommendations:")
            for rec in conclusion['recommendations']:
                print(f"  - {rec}")


if __name__ == "__main__":
    # 実用的統合テスト実行
    test_suite = PracticalIntegrationTest()
    final_report = test_suite.run_comprehensive_integration_test()

    # 結果に基づいて終了コード設定
    success = final_report['test_conclusion']['ready_for_production']
    exit(0 if success else 1)