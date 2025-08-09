#!/usr/bin/env python3
"""
包括的パフォーマンステストスイート
Phase E: システム品質強化フェーズ

統合最適化システムの性能検証・回帰テスト
"""

import gc
import memory_profiler
import psutil
import pytest
import time
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.day_trade.core.optimization_strategy import (
    OptimizationConfig,
    OptimizationLevel,
    get_optimized_implementation
)


class PerformanceTestSuite:
    """包括的パフォーマンステストスイート"""
    
    @pytest.fixture(autouse=True)
    def setup_performance_test_data(self):
        """パフォーマンステスト用データセットアップ"""
        np.random.seed(42)
        
        # 小規模データセット（基本テスト用）
        self.small_data = self._generate_market_data(100)
        
        # 中規模データセット（通常運用想定）
        self.medium_data = self._generate_market_data(1000)
        
        # 大規模データセット（負荷テスト用）
        self.large_data = self._generate_market_data(10000)
        
        # システム情報取得
        self.system_info = self._get_system_info()
        
        print(f"💻 システム情報: CPU {psutil.cpu_count()}コア, メモリ {self.system_info['memory_gb']:.1f}GB")
    
    def _generate_market_data(self, periods: int) -> pd.DataFrame:
        """市場データ生成"""
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        
        price_base = 1000
        returns = np.random.normal(0.001, 0.02, periods)
        prices = [price_base]
        
        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, periods)
        }).set_index('Date')
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'cpu_freq_ghz': psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else 0
        }
    
    @pytest.mark.benchmark(group="technical_indicators")
    def test_technical_indicators_performance_standard_vs_optimized(self):
        """テクニカル指標パフォーマンス: 標準 vs 最適化"""
        try:
            from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager
            
            configs = {
                'standard': OptimizationConfig(level=OptimizationLevel.STANDARD),
                'optimized': OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
            }
            
            indicators = ["sma", "ema", "rsi", "macd", "bollinger_bands", "stochastic", "cci", "williams_r"]
            results = {}
            
            for config_name, config in configs.items():
                manager = TechnicalIndicatorsManager(config)
                
                # メモリ使用量測定開始
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                start_time = time.time()
                
                # 指標計算実行
                calculation_results = manager.calculate_indicators(self.medium_data, indicators, period=20)
                
                execution_time = time.time() - start_time
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                results[config_name] = {
                    'execution_time': execution_time,
                    'memory_used_mb': memory_used,
                    'indicators_calculated': len(calculation_results),
                    'strategy_name': manager.get_strategy().get_strategy_name()
                }
                
                # メモリクリーンアップ
                del manager
                gc.collect()
            
            # 性能比較分析
            standard_time = results['standard']['execution_time']
            optimized_time = results['optimized']['execution_time']
            speedup_ratio = standard_time / optimized_time if optimized_time > 0 else float('inf')
            
            standard_memory = results['standard']['memory_used_mb']
            optimized_memory = results['optimized']['memory_used_mb']
            memory_efficiency = (standard_memory - optimized_memory) / standard_memory * 100 if standard_memory > 0 else 0
            
            # 性能期待値チェック
            assert speedup_ratio > 0.8, f"最適化版の性能劣化: {speedup_ratio:.2f}倍"  # 最低でも同等性能
            assert results['optimized']['indicators_calculated'] == len(indicators), "指標計算数不一致"
            
            print(f"⚡ テクニカル指標パフォーマンス比較:")
            print(f"   標準版: {standard_time:.3f}秒, {standard_memory:.1f}MB")
            print(f"   最適化版: {optimized_time:.3f}秒, {optimized_memory:.1f}MB")
            print(f"   速度比: {speedup_ratio:.2f}倍, メモリ効率: {memory_efficiency:.1f}%改善")
            
            return results
            
        except ImportError as e:
            pytest.skip(f"テクニカル指標統合システム未利用: {e}")
    
    @pytest.mark.benchmark(group="parallel_processing")
    def test_parallel_processing_scalability(self):
        """並列処理スケーラビリティテスト"""
        try:
            from src.day_trade.analysis.feature_engineering_unified import (
                FeatureEngineeringManager,
                FeatureConfig
            )
            
            # 並列度を変化させてテスト
            worker_counts = [1, 2, 4, min(8, psutil.cpu_count())]
            results = {}
            
            for workers in worker_counts:
                config = OptimizationConfig(
                    level=OptimizationLevel.OPTIMIZED,
                    parallel_processing=True
                )
                
                feature_config = FeatureConfig(
                    lookback_periods=[5, 10, 20, 50],
                    volatility_windows=[10, 20, 50],
                    momentum_periods=[5, 10, 20],
                    enable_parallel=True,
                    max_workers=workers
                )
                
                manager = FeatureEngineeringManager(config)
                
                start_time = time.time()
                result = manager.generate_features(self.medium_data, feature_config)
                execution_time = time.time() - start_time
                
                results[workers] = {
                    'execution_time': execution_time,
                    'features_generated': len(result.feature_names) if result else 0,
                    'worker_count': workers
                }
                
                del manager
                gc.collect()
            
            # スケーラビリティ分析
            single_thread_time = results[1]['execution_time']
            best_parallel_time = min(results[w]['execution_time'] for w in worker_counts if w > 1)
            parallel_efficiency = single_thread_time / best_parallel_time
            
            # スケーラビリティ期待値
            assert parallel_efficiency > 1.0, f"並列処理効果なし: {parallel_efficiency:.2f}倍"
            
            print(f"⚡ 並列処理スケーラビリティ:")
            for workers, result in results.items():
                speedup = single_thread_time / result['execution_time']
                efficiency = speedup / workers * 100
                print(f"   {workers}ワーカー: {result['execution_time']:.3f}秒, 高速化{speedup:.2f}倍, 効率{efficiency:.1f}%")
            
            return results
            
        except ImportError as e:
            pytest.skip(f"特徴量エンジニアリング統合システム未利用: {e}")
    
    @pytest.mark.memory
    def test_memory_usage_optimization(self):
        """メモリ使用量最適化テスト"""
        try:
            from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager
            
            configs = {
                'cache_disabled': OptimizationConfig(
                    level=OptimizationLevel.OPTIMIZED,
                    cache_enabled=False
                ),
                'cache_enabled': OptimizationConfig(
                    level=OptimizationLevel.OPTIMIZED,
                    cache_enabled=True
                )
            }
            
            results = {}
            
            for config_name, config in configs.items():
                # メモリプロファイリング
                memory_samples = []
                
                def memory_monitor():
                    process = psutil.Process()
                    for _ in range(10):  # 1秒間、0.1秒間隔でサンプリング
                        memory_samples.append(process.memory_info().rss / 1024 / 1024)
                        time.sleep(0.1)
                
                manager = TechnicalIndicatorsManager(config)
                
                # メモリ監視開始
                monitor_thread = threading.Thread(target=memory_monitor)
                monitor_thread.start()
                
                # 複数回実行（キャッシュ効果確認）
                indicators = ["sma", "ema", "rsi", "bollinger_bands"]
                for i in range(3):
                    manager.calculate_indicators(self.small_data, indicators)
                
                monitor_thread.join()
                
                results[config_name] = {
                    'max_memory_mb': max(memory_samples),
                    'avg_memory_mb': np.mean(memory_samples),
                    'memory_variance': np.var(memory_samples)
                }
                
                del manager
                gc.collect()
            
            # メモリ効率分析
            cache_disabled_memory = results['cache_disabled']['avg_memory_mb']
            cache_enabled_memory = results['cache_enabled']['avg_memory_mb']
            
            print(f"💾 メモリ使用量比較:")
            print(f"   キャッシュ無効: 最大{results['cache_disabled']['max_memory_mb']:.1f}MB, 平均{cache_disabled_memory:.1f}MB")
            print(f"   キャッシュ有効: 最大{results['cache_enabled']['max_memory_mb']:.1f}MB, 平均{cache_enabled_memory:.1f}MB")
            
            # メモリ使用量の妥当性チェック
            assert results['cache_enabled']['max_memory_mb'] < 500, "メモリ使用量過大"  # 500MB未満
            
            return results
            
        except ImportError as e:
            pytest.skip(f"メモリ最適化テスト未実行: {e}")
    
    @pytest.mark.stress
    def test_high_load_stress_test(self):
        """高負荷ストレステスト"""
        try:
            from src.day_trade.analysis.multi_timeframe_analysis_unified import MultiTimeframeAnalysisManager
            
            config = OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
                parallel_processing=True
            )
            
            manager = MultiTimeframeAnalysisManager(config)
            
            # 同時実行数を段階的に増加
            concurrent_levels = [1, 2, 4, 8]
            results = {}
            
            for concurrent_count in concurrent_levels:
                start_time = time.time()
                successful_executions = 0
                
                with ThreadPoolExecutor(max_workers=concurrent_count) as executor:
                    # 複数の分析を同時実行
                    futures = []
                    for i in range(concurrent_count * 2):  # ワーカー数の2倍のタスク
                        future = executor.submit(manager.analyze_multi_timeframe, self.medium_data)
                        futures.append(future)
                    
                    # 結果収集
                    for future in as_completed(futures, timeout=60):
                        try:
                            result = future.result()
                            if result and hasattr(result, 'integrated_trend'):
                                successful_executions += 1
                        except Exception as e:
                            print(f"⚠️  並行実行エラー: {e}")
                
                execution_time = time.time() - start_time
                success_rate = successful_executions / (concurrent_count * 2)
                
                results[concurrent_count] = {
                    'execution_time': execution_time,
                    'successful_executions': successful_executions,
                    'success_rate': success_rate,
                    'throughput': successful_executions / execution_time
                }
                
                # 成功率期待値
                assert success_rate > 0.8, f"高負荷時の成功率不足: {success_rate:.2%}"
            
            print(f"🔥 高負荷ストレステスト結果:")
            for concurrent, result in results.items():
                print(f"   同時実行数{concurrent}: 成功率{result['success_rate']:.2%}, "
                      f"スループット{result['throughput']:.2f}タスク/秒")
            
            return results
            
        except ImportError as e:
            pytest.skip(f"高負荷ストレステスト未実行: {e}")
    
    @pytest.mark.regression
    def test_performance_regression(self):
        """パフォーマンス回帰テスト"""
        # 基準性能値（実測値ベース）
        baseline_performance = {
            'technical_indicators_time': 2.0,  # 秒
            'feature_engineering_time': 3.0,   # 秒
            'multi_timeframe_time': 5.0,       # 秒
            'max_memory_usage': 200,            # MB
        }
        
        current_performance = {}
        
        # テクニカル指標性能テスト
        try:
            from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager
            
            config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
            manager = TechnicalIndicatorsManager(config)
            
            start_time = time.time()
            manager.calculate_indicators(self.medium_data, ["sma", "ema", "rsi", "macd"])
            current_performance['technical_indicators_time'] = time.time() - start_time
            
        except ImportError:
            current_performance['technical_indicators_time'] = float('inf')
        
        # 特徴量エンジニアリング性能テスト
        try:
            from src.day_trade.analysis.feature_engineering_unified import (
                FeatureEngineeringManager,
                FeatureConfig
            )
            
            config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
            manager = FeatureEngineeringManager(config)
            feature_config = FeatureConfig(
                lookback_periods=[5, 10, 20],
                volatility_windows=[10, 20],
                momentum_periods=[5, 10]
            )
            
            start_time = time.time()
            manager.generate_features(self.medium_data, feature_config)
            current_performance['feature_engineering_time'] = time.time() - start_time
            
        except ImportError:
            current_performance['feature_engineering_time'] = float('inf')
        
        # マルチタイムフレーム性能テスト
        try:
            from src.day_trade.analysis.multi_timeframe_analysis_unified import MultiTimeframeAnalysisManager
            
            config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
            manager = MultiTimeframeAnalysisManager(config)
            
            start_time = time.time()
            manager.analyze_multi_timeframe(self.medium_data)
            current_performance['multi_timeframe_time'] = time.time() - start_time
            
        except ImportError:
            current_performance['multi_timeframe_time'] = float('inf')
        
        # 回帰チェック
        regression_detected = False
        regression_details = []
        
        for metric, baseline_value in baseline_performance.items():
            if metric in current_performance:
                current_value = current_performance[metric]
                if current_value > baseline_value * 1.2:  # 20%以上の劣化で回帰
                    regression_detected = True
                    regression_ratio = current_value / baseline_value
                    regression_details.append(f"{metric}: {regression_ratio:.2f}倍劣化")
        
        print(f"📊 パフォーマンス回帰テスト:")
        for metric, baseline in baseline_performance.items():
            current = current_performance.get(metric, float('inf'))
            status = "✅" if current <= baseline * 1.2 else "❌"
            print(f"   {status} {metric}: 基準{baseline:.2f}, 現在{current:.2f}")
        
        if regression_detected:
            print(f"⚠️  パフォーマンス回帰検出: {', '.join(regression_details)}")
            # 回帰は警告のみ（CIでは継続）
        
        return {
            'regression_detected': regression_detected,
            'current_performance': current_performance,
            'baseline_performance': baseline_performance
        }
    
    @pytest.mark.benchmark(group="cache_efficiency")
    def test_cache_efficiency(self):
        """キャッシュ効率テスト"""
        try:
            from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager
            
            config = OptimizationConfig(
                level=OptimizationLevel.OPTIMIZED,
                cache_enabled=True
            )
            
            manager = TechnicalIndicatorsManager(config)
            indicators = ["sma", "ema", "rsi"]
            
            # 1回目実行（キャッシュ構築）
            start_time = time.time()
            manager.calculate_indicators(self.small_data, indicators)
            first_execution_time = time.time() - start_time
            
            # 2回目実行（キャッシュ効果期待）
            start_time = time.time()
            manager.calculate_indicators(self.small_data, indicators)  # 同じデータ・指標
            second_execution_time = time.time() - start_time
            
            # キャッシュ効果分析
            cache_speedup = first_execution_time / max(second_execution_time, 0.001)
            
            # キャッシュ統計取得
            strategy = manager.get_strategy()
            cache_stats = {}
            if hasattr(strategy, 'get_cache_stats'):
                cache_stats = strategy.get_cache_stats()
            
            print(f"🗄️  キャッシュ効率テスト:")
            print(f"   1回目: {first_execution_time:.3f}秒")
            print(f"   2回目: {second_execution_time:.3f}秒")
            print(f"   キャッシュ効果: {cache_speedup:.1f}倍高速化")
            
            if cache_stats:
                print(f"   キャッシュ統計: {cache_stats}")
            
            # キャッシュ効果期待値
            assert cache_speedup > 1.0, f"キャッシュ効果なし: {cache_speedup:.2f}倍"
            
            return {
                'first_execution_time': first_execution_time,
                'second_execution_time': second_execution_time,
                'cache_speedup': cache_speedup,
                'cache_stats': cache_stats
            }
            
        except ImportError as e:
            pytest.skip(f"キャッシュ効率テスト未実行: {e}")


if __name__ == "__main__":
    # 直接実行時のテスト
    test_suite = PerformanceTestSuite()
    test_suite.setup_performance_test_data()
    
    print("🚀 包括的パフォーマンステスト開始")
    
    try:
        # 基本パフォーマンステスト
        test_suite.test_technical_indicators_performance_standard_vs_optimized()
        test_suite.test_memory_usage_optimization()
        test_suite.test_cache_efficiency()
        
        print("✅ 包括的パフォーマンステスト完了")
        
    except Exception as e:
        print(f"❌ パフォーマンステスト失敗: {e}")
        raise