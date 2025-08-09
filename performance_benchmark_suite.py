#!/usr/bin/env python3
"""
パフォーマンス・ベンチマークスイート
Phase G: 本番運用最適化フェーズ

全システムの性能測定・ボトルネック発見・最適化提案
"""

import sys
import time
import json
import psutil
import threading
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# プロジェクトパス追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 主要モジュールのインポート
try:
    from src.day_trade.core.optimization_strategy import OptimizationConfig, OptimizationLevel, get_optimized_implementation
    from src.day_trade.data.stock_fetcher import StockFetcher
    from src.day_trade.ml import DeepLearningModelManager, DeepLearningConfig
    from src.day_trade.acceleration import GPUAccelerationEngine
    from src.day_trade.api import RealtimePredictionAPI
except ImportError as e:
    print(f"[CRITICAL] モジュールインポートエラー: {e}")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    test_name: str
    component: str
    optimization_level: str
    execution_time: float
    memory_usage: float  # MB
    throughput: float   # operations/sec
    success_rate: float # 0.0-1.0
    cpu_usage: float   # %
    latency_p50: float # ms
    latency_p95: float # ms
    latency_p99: float # ms
    error_count: int
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""
    timestamp: datetime
    system_info: Dict[str, Any]
    benchmark_results: List[BenchmarkResult]
    optimization_comparison: Dict[str, Dict[str, float]]
    bottleneck_analysis: Dict[str, Any]
    recommendations: List[str]
    overall_score: int


class SystemMonitor:
    """システム監視ユーティリティ"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """監視停止・結果取得"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {'cpu_avg': 0, 'memory_avg': 0, 'cpu_max': 0, 'memory_max': 0}
        
        cpu_values = [m['cpu'] for m in self.metrics]
        memory_values = [m['memory'] for m in self.metrics]
        
        return {
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_max': max(cpu_values),
            'cpu_p95': statistics.quantiles(cpu_values, n=20)[18] if len(cpu_values) > 10 else max(cpu_values),
            'memory_avg': statistics.mean(memory_values),
            'memory_max': max(memory_values),
            'memory_p95': statistics.quantiles(memory_values, n=20)[18] if len(memory_values) > 10 else max(memory_values)
        }
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu': cpu_percent,
                    'memory': memory_percent
                })
                
                time.sleep(0.5)
            except Exception:
                pass


class PerformanceBenchmarkSuite:
    """パフォーマンスベンチマークスイート"""
    
    def __init__(self):
        self.results = []
        self.system_info = self._collect_system_info()
        self.monitor = SystemMonitor()
        
        print("=" * 80)
        print("[BENCHMARK] Day Trade パフォーマンステストスイート")
        print("Phase G: 本番運用最適化フェーズ")
        print("=" * 80)
        print(f"テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"システム: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"CPU: {self.system_info['cpu_cores']}コア ({self.system_info['cpu_freq']:.0f}MHz)")
        print(f"RAM: {self.system_info['memory_total']:.1f}GB")
        print("=" * 80)
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """システム情報収集"""
        import platform
        
        cpu_freq = psutil.cpu_freq()
        
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'cpu_cores': psutil.cpu_count(logical=True),
            'cpu_physical_cores': psutil.cpu_count(logical=False),
            'cpu_freq': cpu_freq.current if cpu_freq else 0,
            'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
            'hostname': platform.node()
        }
    
    def _measure_latencies(self, execution_times: List[float]) -> Tuple[float, float, float]:
        """レイテンシ統計計算"""
        if not execution_times:
            return 0.0, 0.0, 0.0
        
        # ミリ秒に変換
        latencies_ms = [t * 1000 for t in execution_times]
        latencies_ms.sort()
        
        n = len(latencies_ms)
        p50 = latencies_ms[int(n * 0.50)]
        p95 = latencies_ms[int(n * 0.95)]
        p99 = latencies_ms[int(n * 0.99)]
        
        return p50, p95, p99
    
    def benchmark_data_fetching(self) -> List[BenchmarkResult]:
        """データ取得システムベンチマーク"""
        print("\n[DATA] データ取得システム ベンチマーク実行中...")
        
        results = []
        test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.OPTIMIZED]:
            print(f"  最適化レベル: {optimization_level.value}")
            
            config = OptimizationConfig(level=optimization_level)
            fetcher = StockFetcher(config)
            
            execution_times = []
            error_count = 0
            
            # システム監視開始
            self.monitor.start_monitoring()
            
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024**2  # MB
            
            # 複数シンボルでのテスト
            for symbol in test_symbols:
                try:
                    exec_start = time.time()
                    
                    # テストデータ取得（モック）
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=100)
                    
                    data = fetcher.get_stock_data(
                        symbol, 
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    exec_time = time.time() - exec_start
                    execution_times.append(exec_time)
                    
                    if data is None or len(data) == 0:
                        error_count += 1
                        
                except Exception:
                    error_count += 1
                    execution_times.append(0.0)
            
            total_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024**2  # MB
            memory_usage = end_memory - start_memory
            
            # 監視停止
            monitoring_stats = self.monitor.stop_monitoring()
            
            # 統計計算
            throughput = len(test_symbols) / total_time if total_time > 0 else 0
            success_rate = (len(test_symbols) - error_count) / len(test_symbols)
            p50, p95, p99 = self._measure_latencies(execution_times)
            
            result = BenchmarkResult(
                test_name="Data Fetching",
                component="stock_fetcher",
                optimization_level=optimization_level.value,
                execution_time=total_time,
                memory_usage=memory_usage,
                throughput=throughput,
                success_rate=success_rate,
                cpu_usage=monitoring_stats['cpu_avg'],
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99,
                error_count=error_count,
                metadata={
                    'symbols_tested': len(test_symbols),
                    'cache_enabled': hasattr(fetcher, 'cache') and fetcher.cache is not None
                }
            )
            
            results.append(result)
            
            print(f"    実行時間: {total_time:.3f}秒")
            print(f"    スループット: {throughput:.1f} symbols/sec")
            print(f"    成功率: {success_rate:.1%}")
            print(f"    P50レイテンシ: {p50:.1f}ms")
        
        return results
    
    def benchmark_technical_analysis(self) -> List[BenchmarkResult]:
        """テクニカル分析システムベンチマーク"""
        print("\n[ANALYSIS] テクニカル分析システム ベンチマーク実行中...")
        
        results = []
        
        # テストデータ生成
        test_data_sizes = [100, 500, 1000, 5000]
        indicators = ['sma', 'ema', 'rsi', 'bollinger_bands', 'macd']
        
        for optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.OPTIMIZED]:
            print(f"  最適化レベル: {optimization_level.value}")
            
            for data_size in test_data_sizes:
                # テストデータ生成
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', periods=data_size, freq='D')
                test_data = pd.DataFrame({
                    'Date': dates,
                    'Open': np.random.uniform(100, 200, data_size),
                    'High': np.random.uniform(150, 250, data_size),
                    'Low': np.random.uniform(50, 150, data_size),
                    'Close': np.random.uniform(100, 200, data_size),
                    'Volume': np.random.randint(100000, 1000000, data_size)
                }).set_index('Date')
                
                config = OptimizationConfig(level=optimization_level)
                technical_indicators = get_optimized_implementation("technical_indicators", config)
                
                execution_times = []
                error_count = 0
                
                # システム監視開始
                self.monitor.start_monitoring()
                
                start_time = time.time()
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024**2  # MB
                
                # 各指標での計算テスト
                for indicator in indicators:
                    try:
                        exec_start = time.time()
                        
                        if indicator == 'sma':
                            result = technical_indicators.calculate_sma(test_data, period=20)
                        elif indicator == 'ema':
                            result = technical_indicators.calculate_ema(test_data, period=20)
                        elif indicator == 'rsi':
                            result = technical_indicators.calculate_rsi(test_data, period=14)
                        elif indicator == 'bollinger_bands':
                            result = technical_indicators.calculate_bollinger_bands(test_data, period=20)
                        elif indicator == 'macd':
                            result = technical_indicators.calculate_macd(test_data)
                        
                        exec_time = time.time() - exec_start
                        execution_times.append(exec_time)
                        
                        if result is None:
                            error_count += 1
                            
                    except Exception:
                        error_count += 1
                        execution_times.append(0.0)
                
                total_time = time.time() - start_time
                end_memory = process.memory_info().rss / 1024**2  # MB
                memory_usage = end_memory - start_memory
                
                # 監視停止
                monitoring_stats = self.monitor.stop_monitoring()
                
                # 統計計算
                throughput = len(indicators) / total_time if total_time > 0 else 0
                success_rate = (len(indicators) - error_count) / len(indicators)
                p50, p95, p99 = self._measure_latencies(execution_times)
                
                result = BenchmarkResult(
                    test_name=f"Technical Analysis (n={data_size})",
                    component="technical_indicators",
                    optimization_level=optimization_level.value,
                    execution_time=total_time,
                    memory_usage=memory_usage,
                    throughput=throughput,
                    success_rate=success_rate,
                    cpu_usage=monitoring_stats['cpu_avg'],
                    latency_p50=p50,
                    latency_p95=p95,
                    latency_p99=p99,
                    error_count=error_count,
                    metadata={
                        'data_points': data_size,
                        'indicators_tested': len(indicators),
                        'memory_per_point': memory_usage / data_size if data_size > 0 else 0
                    }
                )
                
                results.append(result)
                
                print(f"    データサイズ {data_size}: {total_time:.3f}秒, {throughput:.1f} indicators/sec")
        
        return results
    
    def benchmark_gpu_acceleration(self) -> List[BenchmarkResult]:
        """GPU加速システムベンチマーク"""
        print("\n[GPU] GPU加速システム ベンチマーク実行中...")
        
        results = []
        
        # テストデータサイズ
        test_data_sizes = [1000, 5000, 10000]
        indicators = ['sma', 'ema', 'rsi']
        
        for data_size in test_data_sizes:
            # テストデータ生成
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=data_size, freq='D')
            test_data = pd.DataFrame({
                'Date': dates,
                'Open': np.random.uniform(100, 200, data_size),
                'High': np.random.uniform(150, 250, data_size),
                'Low': np.random.uniform(50, 150, data_size),
                'Close': np.random.uniform(100, 200, data_size),
                'Volume': np.random.randint(100000, 1000000, data_size)
            }).set_index('Date')
            
            gpu_engine = GPUAccelerationEngine()
            
            execution_times = []
            error_count = 0
            
            # システム監視開始
            self.monitor.start_monitoring()
            
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024**2  # MB
            
            try:
                exec_start = time.time()
                result = gpu_engine.accelerate_technical_indicators(test_data, indicators)
                exec_time = time.time() - exec_start
                execution_times.append(exec_time)
                
                if result is None:
                    error_count += 1
                    
            except Exception:
                error_count += 1
                execution_times.append(0.0)
            
            total_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024**2  # MB
            memory_usage = end_memory - start_memory
            
            # 監視停止
            monitoring_stats = self.monitor.stop_monitoring()
            
            # 統計計算
            throughput = data_size / total_time if total_time > 0 else 0
            success_rate = 1.0 - (error_count / max(1, len(indicators)))
            p50, p95, p99 = self._measure_latencies(execution_times)
            
            benchmark_result = BenchmarkResult(
                test_name=f"GPU Acceleration (n={data_size})",
                component="gpu_acceleration",
                optimization_level="gpu_accelerated",
                execution_time=total_time,
                memory_usage=memory_usage,
                throughput=throughput,
                success_rate=success_rate,
                cpu_usage=monitoring_stats['cpu_avg'],
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99,
                error_count=error_count,
                metadata={
                    'data_points': data_size,
                    'backend_used': gpu_engine.primary_backend.value,
                    'device_count': len(gpu_engine.devices)
                }
            )
            
            results.append(benchmark_result)
            
            print(f"    データサイズ {data_size}: {total_time:.3f}秒, {throughput:.0f} points/sec")
            print(f"    使用バックエンド: {gpu_engine.primary_backend.value}")
        
        return results
    
    def benchmark_deep_learning_system(self) -> List[BenchmarkResult]:
        """深層学習システムベンチマーク"""
        print("\n[ML] 深層学習システム ベンチマーク実行中...")
        
        results = []
        
        # 軽量テスト設定
        dl_config = DeepLearningConfig(
            sequence_length=30,
            epochs=3,  # 高速テスト用
            batch_size=16,
            use_pytorch=False  # 安定性のためNumPy使用
        )
        
        opt_config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)
        
        # テストデータサイズ
        test_data_sizes = [100, 500]
        
        for data_size in test_data_sizes:
            # テストデータ生成
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=data_size, freq='D')
            test_data = pd.DataFrame({
                'Date': dates,
                'Open': np.random.uniform(100, 200, data_size),
                'High': np.random.uniform(150, 250, data_size),
                'Low': np.random.uniform(50, 150, data_size),
                'Close': np.random.uniform(100, 200, data_size),
                'Volume': np.random.randint(100000, 1000000, data_size)
            }).set_index('Date')
            
            manager = DeepLearningModelManager(dl_config, opt_config)
            
            # モデル登録
            from src.day_trade.ml import TransformerModel, LSTMModel
            transformer = TransformerModel(dl_config)
            lstm = LSTMModel(dl_config)
            
            manager.register_model("transformer", transformer)
            manager.register_model("lstm", lstm)
            
            execution_times = []
            error_count = 0
            
            # システム監視開始
            self.monitor.start_monitoring()
            
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024**2  # MB
            
            try:
                # 訓練テスト
                exec_start = time.time()
                training_results = manager.train_ensemble(test_data)
                exec_time = time.time() - exec_start
                execution_times.append(exec_time)
                
                # 予測テスト
                exec_start = time.time()
                prediction_result = manager.predict_ensemble(test_data.tail(50))
                exec_time = time.time() - exec_start
                execution_times.append(exec_time)
                
                if training_results is None or prediction_result is None:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                execution_times.extend([0.0, 0.0])
                print(f"      エラー: {str(e)[:100]}...")
            
            total_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024**2  # MB
            memory_usage = end_memory - start_memory
            
            # 監視停止
            monitoring_stats = self.monitor.stop_monitoring()
            
            # 統計計算
            throughput = 1 / total_time if total_time > 0 else 0  # models/sec
            success_rate = 1.0 - (error_count / 2)  # 2つのテスト
            p50, p95, p99 = self._measure_latencies(execution_times)
            
            result = BenchmarkResult(
                test_name=f"Deep Learning (n={data_size})",
                component="deep_learning",
                optimization_level="optimized",
                execution_time=total_time,
                memory_usage=memory_usage,
                throughput=throughput,
                success_rate=success_rate,
                cpu_usage=monitoring_stats['cpu_avg'],
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99,
                error_count=error_count,
                metadata={
                    'data_points': data_size,
                    'models_trained': len(manager.models),
                    'pytorch_used': dl_config.use_pytorch
                }
            )
            
            results.append(result)
            
            print(f"    データサイズ {data_size}: {total_time:.3f}秒")
            print(f"    モデル数: {len(manager.models)}")
        
        return results
    
    def benchmark_api_performance(self) -> List[BenchmarkResult]:
        """API パフォーマンステスト"""
        print("\n[API] リアルタイム予測API ベンチマーク実行中...")
        
        results = []
        
        dl_config = DeepLearningConfig(use_pytorch=False, epochs=1)
        opt_config = OptimizationConfig(level=OptimizationLevel.STANDARD)
        
        api = RealtimePredictionAPI(dl_config, opt_config)
        
        # 並行リクエストテスト
        concurrent_levels = [1, 5, 10]
        
        for concurrent_requests in concurrent_levels:
            execution_times = []
            error_count = 0
            
            # システム監視開始
            self.monitor.start_monitoring()
            
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024**2  # MB
            
            # 並行処理テスト
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = []
                
                for i in range(concurrent_requests * 2):  # 各レベルで複数リクエスト
                    future = executor.submit(self._simulate_api_request, api)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        exec_time = future.result()
                        execution_times.append(exec_time)
                    except Exception:
                        error_count += 1
                        execution_times.append(0.0)
            
            total_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024**2  # MB
            memory_usage = end_memory - start_memory
            
            # 監視停止
            monitoring_stats = self.monitor.stop_monitoring()
            
            # 統計計算
            total_requests = concurrent_requests * 2
            throughput = total_requests / total_time if total_time > 0 else 0
            success_rate = (total_requests - error_count) / total_requests
            p50, p95, p99 = self._measure_latencies(execution_times)
            
            result = BenchmarkResult(
                test_name=f"API Performance (concurrent={concurrent_requests})",
                component="realtime_api",
                optimization_level="standard",
                execution_time=total_time,
                memory_usage=memory_usage,
                throughput=throughput,
                success_rate=success_rate,
                cpu_usage=monitoring_stats['cpu_avg'],
                latency_p50=p50,
                latency_p95=p95,
                latency_p99=p99,
                error_count=error_count,
                metadata={
                    'concurrent_requests': concurrent_requests,
                    'total_requests': total_requests
                }
            )
            
            results.append(result)
            
            print(f"    並行レベル {concurrent_requests}: {throughput:.1f} req/sec")
            print(f"    P95レイテンシ: {p95:.1f}ms")
        
        return results
    
    def _simulate_api_request(self, api: RealtimePredictionAPI) -> float:
        """API リクエスト シミュレーション"""
        from src.day_trade.api.realtime_prediction_api import PredictionRequest
        
        start_time = time.time()
        
        request = PredictionRequest(
            symbol="AAPL",
            prediction_horizon=3,
            confidence_level=0.95,
            include_uncertainty=False,
            model_ensemble=False
        )
        
        try:
            # 実際のAPIではなく直接メソッド呼び出し
            result = api._execute_prediction(request)
            return time.time() - start_time
        except Exception:
            return time.time() - start_time
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """ボトルネック分析"""
        print("\n[ANALYSIS] ボトルネック分析中...")
        
        if not self.results:
            return {}
        
        # コンポーネント別パフォーマンス
        component_stats = {}
        for result in self.results:
            component = result.component
            if component not in component_stats:
                component_stats[component] = {
                    'execution_times': [],
                    'memory_usage': [],
                    'throughput': [],
                    'cpu_usage': [],
                    'error_rates': []
                }
            
            component_stats[component]['execution_times'].append(result.execution_time)
            component_stats[component]['memory_usage'].append(result.memory_usage)
            component_stats[component]['throughput'].append(result.throughput)
            component_stats[component]['cpu_usage'].append(result.cpu_usage)
            component_stats[component]['error_rates'].append(1.0 - result.success_rate)
        
        # ボトルネック特定
        bottlenecks = {}
        
        # 最も遅いコンポーネント
        slowest_component = max(
            component_stats.keys(),
            key=lambda c: statistics.mean(component_stats[c]['execution_times'])
        )
        
        # 最もメモリを使用するコンポーネント
        memory_heaviest = max(
            component_stats.keys(),
            key=lambda c: statistics.mean(component_stats[c]['memory_usage'])
        )
        
        # 最もエラー率の高いコンポーネント
        error_prone = max(
            component_stats.keys(),
            key=lambda c: statistics.mean(component_stats[c]['error_rates'])
        )
        
        bottlenecks = {
            'performance_bottleneck': {
                'component': slowest_component,
                'avg_execution_time': statistics.mean(component_stats[slowest_component]['execution_times']),
                'impact': 'high'
            },
            'memory_bottleneck': {
                'component': memory_heaviest,
                'avg_memory_usage': statistics.mean(component_stats[memory_heaviest]['memory_usage']),
                'impact': 'medium'
            },
            'reliability_bottleneck': {
                'component': error_prone,
                'avg_error_rate': statistics.mean(component_stats[error_prone]['error_rates']),
                'impact': 'high' if statistics.mean(component_stats[error_prone]['error_rates']) > 0.1 else 'low'
            }
        }
        
        return bottlenecks
    
    def generate_optimization_comparison(self) -> Dict[str, Dict[str, float]]:
        """最適化レベル比較分析"""
        comparison = {}
        
        # コンポーネント別、最適化レベル別の統計
        for result in self.results:
            component = result.component
            opt_level = result.optimization_level
            
            if component not in comparison:
                comparison[component] = {}
            
            if opt_level not in comparison[component]:
                comparison[component][opt_level] = {
                    'execution_times': [],
                    'throughput': [],
                    'memory_usage': []
                }
            
            comparison[component][opt_level]['execution_times'].append(result.execution_time)
            comparison[component][opt_level]['throughput'].append(result.throughput)
            comparison[component][opt_level]['memory_usage'].append(result.memory_usage)
        
        # 平均値計算
        for component in comparison:
            for opt_level in comparison[component]:
                stats = comparison[component][opt_level]
                comparison[component][opt_level] = {
                    'avg_execution_time': statistics.mean(stats['execution_times']) if stats['execution_times'] else 0,
                    'avg_throughput': statistics.mean(stats['throughput']) if stats['throughput'] else 0,
                    'avg_memory_usage': statistics.mean(stats['memory_usage']) if stats['memory_usage'] else 0,
                    'sample_count': len(stats['execution_times'])
                }
        
        return comparison
    
    def generate_recommendations(self, bottlenecks: Dict[str, Any]) -> List[str]:
        """最適化推奨事項生成"""
        recommendations = []
        
        # パフォーマンスボトルネック対策
        if bottlenecks.get('performance_bottleneck', {}).get('impact') == 'high':
            component = bottlenecks['performance_bottleneck']['component']
            recommendations.append(f"{component}の実行時間最適化が最優先課題")
            
            if component == 'deep_learning':
                recommendations.append("PyTorch利用による深層学習モデル高速化")
                recommendations.append("バッチサイズ調整とモデル軽量化")
            elif component == 'technical_indicators':
                recommendations.append("ベクトル化計算の活用")
                recommendations.append("キャッシュ機能の有効化")
            elif component == 'gpu_acceleration':
                recommendations.append("GPU利用環境の構築")
                recommendations.append("CUDA/OpenCLライブラリのインストール")
        
        # メモリボトルネック対策
        if bottlenecks.get('memory_bottleneck', {}).get('avg_memory_usage', 0) > 500:  # 500MB超
            recommendations.append("メモリ使用量最適化")
            recommendations.append("データの段階的処理（ストリーミング）")
            recommendations.append("不要なデータのガベージコレクション強化")
        
        # 信頼性対策
        if bottlenecks.get('reliability_bottleneck', {}).get('impact') == 'high':
            recommendations.append("エラー処理とリトライ機構の強化")
            recommendations.append("入力データ検証の強化")
            recommendations.append("例外処理の詳細化")
        
        # 一般的な最適化提案
        recommendations.extend([
            "並列処理の活用によるスループット向上",
            "非同期処理の導入",
            "接続プールによるリソース効率化",
            "プロファイリングによる詳細なボトルネック分析"
        ])
        
        return recommendations
    
    def calculate_overall_score(self) -> int:
        """総合パフォーマンススコア計算"""
        if not self.results:
            return 0
        
        # 各指標のスコア計算 (0-100点)
        throughput_scores = []
        latency_scores = []
        success_rate_scores = []
        memory_scores = []
        
        for result in self.results:
            # スループットスコア (log スケール)
            if result.throughput > 0:
                throughput_score = min(100, 50 + 10 * np.log10(result.throughput))
            else:
                throughput_score = 0
            throughput_scores.append(throughput_score)
            
            # レイテンシスコア (逆数)
            if result.latency_p95 > 0:
                latency_score = min(100, max(0, 100 - result.latency_p95 / 10))
            else:
                latency_score = 100
            latency_scores.append(latency_score)
            
            # 成功率スコア
            success_rate_scores.append(result.success_rate * 100)
            
            # メモリスコア (使用量が少ないほど高スコア)
            if result.memory_usage > 0:
                memory_score = min(100, max(0, 100 - result.memory_usage / 10))
            else:
                memory_score = 100
            memory_scores.append(memory_score)
        
        # 重み付き平均 (成功率とレイテンシを重視)
        overall_score = (
            statistics.mean(success_rate_scores) * 0.3 +
            statistics.mean(latency_scores) * 0.3 +
            statistics.mean(throughput_scores) * 0.2 +
            statistics.mean(memory_scores) * 0.2
        )
        
        return int(overall_score)
    
    def run_comprehensive_benchmark(self) -> PerformanceReport:
        """包括的ベンチマーク実行"""
        print("\n[START] 包括的パフォーマンスベンチマーク開始...")
        
        # 各コンポーネントのベンチマーク実行
        all_results = []
        
        try:
            all_results.extend(self.benchmark_data_fetching())
        except Exception as e:
            print(f"[ERROR] データ取得ベンチマークエラー: {e}")
        
        try:
            all_results.extend(self.benchmark_technical_analysis())
        except Exception as e:
            print(f"[ERROR] テクニカル分析ベンチマークエラー: {e}")
        
        try:
            all_results.extend(self.benchmark_gpu_acceleration())
        except Exception as e:
            print(f"[ERROR] GPU加速ベンチマークエラー: {e}")
        
        try:
            all_results.extend(self.benchmark_deep_learning_system())
        except Exception as e:
            print(f"[ERROR] 深層学習ベンチマークエラー: {e}")
        
        try:
            all_results.extend(self.benchmark_api_performance())
        except Exception as e:
            print(f"[ERROR] APIベンチマークエラー: {e}")
        
        self.results = all_results
        
        # 分析実行
        bottlenecks = self.analyze_bottlenecks()
        optimization_comparison = self.generate_optimization_comparison()
        recommendations = self.generate_recommendations(bottlenecks)
        overall_score = self.calculate_overall_score()
        
        # レポート作成
        report = PerformanceReport(
            timestamp=datetime.now(),
            system_info=self.system_info,
            benchmark_results=all_results,
            optimization_comparison=optimization_comparison,
            bottleneck_analysis=bottlenecks,
            recommendations=recommendations,
            overall_score=overall_score
        )
        
        # 結果表示
        print("\n" + "=" * 80)
        print("[SUMMARY] ベンチマーク結果サマリー")
        print("=" * 80)
        print(f"総合パフォーマンススコア: {overall_score}/100")
        print(f"実行テスト数: {len(all_results)}")
        
        if bottlenecks.get('performance_bottleneck'):
            pb = bottlenecks['performance_bottleneck']
            print(f"主要ボトルネック: {pb['component']} ({pb['avg_execution_time']:.3f}秒)")
        
        print("\n主要推奨事項:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")
        
        return report
    
    def save_report(self, report: PerformanceReport, filename: str = None):
        """ベンチマークレポート保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_report_{timestamp}.json"
        
        # JSONシリアライズ対応
        report_dict = asdict(report)
        report_dict['timestamp'] = report.timestamp.isoformat()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n[REPORT] ベンチマークレポートを保存しました: {filename}")


def main():
    """メインベンチマーク実行"""
    try:
        benchmark_suite = PerformanceBenchmarkSuite()
        report = benchmark_suite.run_comprehensive_benchmark()
        benchmark_suite.save_report(report)
        
        # 終了コード決定
        if report.overall_score >= 80:
            exit_code = 0  # 良好
        elif report.overall_score >= 60:
            exit_code = 1  # 改善必要
        else:
            exit_code = 2  # 大幅改善必要
        
        print(f"\n[COMPLETE] パフォーマンスベンチマーク完了 (終了コード: {exit_code})")
        return exit_code
        
    except Exception as e:
        print(f"\n[ERROR] ベンチマークプロセスでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)