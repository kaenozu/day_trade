"""
統合予測・パフォーマンステストシステム

すべてのシステム（予測精度向上、パフォーマンス最適化、機械学習モデル、データ処理）
の統合テストとベンチマークを実行する。
"""

import asyncio
import time
import tracemalloc
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
import statistics

# 各システムのインポート
from ..ml.advanced_prediction_accuracy_system import (
    AdvancedPredictionAccuracySystem,
    PredictionResult
)
from ..optimization.advanced_performance_optimization_system import (
    AdvancedPerformanceOptimizationSystem,
    OptimizationResult
)
from ..ml.enhanced_ml_model_system import (
    EnhancedMLModelSystem,
    ModelPerformance
)
from ..data.advanced_data_processing_system import (
    AdvancedDataProcessingSystem
)


@dataclass
class IntegratedTestResult:
    """統合テスト結果"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # 予測精度メトリクス
    prediction_accuracy: float
    prediction_precision: float
    prediction_recall: float
    prediction_f1_score: float
    
    # パフォーマンスメトリクス
    cpu_usage_percent: float
    memory_usage_mb: float
    processing_speed_records_per_second: float
    cache_hit_rate: float
    
    # システム統合メトリクス
    overall_system_score: float
    component_scores: Dict[str, float]
    recommendations: List[str]
    
    success: bool
    error_message: Optional[str] = None


class IntegratedPredictionPerformanceTestSystem:
    """統合予測・パフォーマンステストシステム"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # システムコンポーネント
        self.prediction_system = AdvancedPredictionAccuracySystem()
        self.performance_system = AdvancedPerformanceOptimizationSystem()
        self.ml_system = EnhancedMLModelSystem()
        self.data_system = AdvancedDataProcessingSystem()
        
        # テスト結果
        self.test_results: List[IntegratedTestResult] = []
        
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def create_test_dataset(self, size: int = 10000) -> pd.DataFrame:
        """テスト用データセット作成"""
        np.random.seed(42)
        
        # 株式市場データのシミュレーション
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=size),
            periods=size,
            freq='1min'
        )
        
        # 基本価格データ
        base_price = 1000
        price_changes = np.random.normal(0, 0.01, size)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # 最小価格制限
        
        # OHLCV データ
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST'] * size,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': np.random.randint(1000, 100000, size),
            'market_cap': np.random.uniform(1e9, 1e12, size),
            'sector': np.random.choice(['tech', 'finance', 'healthcare'], size)
        })
        
        # ターゲット（次の価格変動）
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        data = data.dropna()
        
        return data
    
    async def run_prediction_accuracy_test(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """予測精度テスト"""
        self.logger.info("予測精度テスト開始")
        
        try:
            # データ分割
            split_idx = int(len(test_data) * 0.8)
            train_data = test_data[:split_idx]
            test_data_subset = test_data[split_idx:]
            
            # 予測システム初期化・訓練
            await self.prediction_system.initialize_system()
            
            # テスト予測
            predictions = []
            actuals = []
            
            for i in range(min(100, len(test_data_subset) - 10)):  # 性能のため制限
                row_data = test_data_subset.iloc[i:i+10]
                result = await self.prediction_system.predict(
                    symbol="TEST",
                    data=row_data,
                    market_context={'market_trend': 'bullish'}
                )
                
                predictions.append(1 if result.confidence > 0.5 else 0)
                actuals.append(int(test_data_subset.iloc[i]['target']))
            
            # メトリクス計算
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            accuracy = np.mean(predictions == actuals)
            
            # 精度、再現率、F1スコア
            tp = np.sum((predictions == 1) & (actuals == 1))
            fp = np.sum((predictions == 1) & (actuals == 0))
            fn = np.sum((predictions == 0) & (actuals == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            
        except Exception as e:
            self.logger.error(f"予測精度テストエラー: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    async def run_performance_test(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """パフォーマンステスト"""
        self.logger.info("パフォーマンステスト開始")
        
        # リソース監視開始
        tracemalloc.start()
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # データ処理テスト
            processing_results = []
            for i in range(0, len(test_data), 1000):  # 1000行ごとに処理
                chunk = test_data.iloc[i:i+1000]
                start_chunk = time.time()
                
                # データ処理システム使用
                result = await self.data_system.process_dataset(
                    chunk,
                    {'strategy': 'pandas', 'parallel': True}
                )
                
                processing_time = time.time() - start_chunk
                records_per_second = len(chunk) / processing_time
                processing_results.append(records_per_second)
                
                if len(processing_results) >= 5:  # 5チャンクでテスト終了
                    break
            
            # パフォーマンス最適化実行
            optimization_result = await self.performance_system.optimize_system_performance()
            
            # リソース使用量測定
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # キャッシュヒット率（データシステムから）
            cache_stats = self.data_system.cache.get_stats()
            cache_hit_rate = cache_stats.get('hit_rate', 0.0)
            
            return {
                'cpu_usage': (start_cpu + end_cpu) / 2,
                'memory_usage': (start_memory + end_memory) / 2,
                'processing_speed': statistics.mean(processing_results) if processing_results else 0.0,
                'cache_hit_rate': cache_hit_rate,
                'optimization_score': optimization_result.improvement_percentage if optimization_result else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"パフォーマンステストエラー: {e}")
            return {
                'cpu_usage': 100.0,
                'memory_usage': 1000.0,
                'processing_speed': 0.0,
                'cache_hit_rate': 0.0,
                'optimization_score': 0.0
            }
    
    def calculate_overall_score(self, prediction_metrics: Dict[str, float], 
                              performance_metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """総合スコア計算"""
        
        # コンポーネントスコア計算（0-100スケール）
        component_scores = {
            'prediction_accuracy': prediction_metrics['accuracy'] * 100,
            'prediction_quality': (
                prediction_metrics['precision'] + 
                prediction_metrics['recall'] + 
                prediction_metrics['f1_score']
            ) / 3 * 100,
            'performance_efficiency': min(100, performance_metrics['processing_speed'] / 100 * 100),
            'resource_optimization': max(0, 100 - performance_metrics['cpu_usage']),
            'cache_performance': performance_metrics['cache_hit_rate'] * 100,
            'system_optimization': performance_metrics.get('optimization_score', 0)
        }
        
        # 重み付き総合スコア
        weights = {
            'prediction_accuracy': 0.25,
            'prediction_quality': 0.25,
            'performance_efficiency': 0.20,
            'resource_optimization': 0.15,
            'cache_performance': 0.10,
            'system_optimization': 0.05
        }
        
        overall_score = sum(
            component_scores[component] * weights[component]
            for component in component_scores
        )
        
        return overall_score, component_scores
    
    def generate_recommendations(self, component_scores: Dict[str, float]) -> List[str]:
        """改善提案生成"""
        recommendations = []
        
        if component_scores['prediction_accuracy'] < 60:
            recommendations.append("予測精度が低いため、特徴量エンジニアリングとモデル調整を推奨")
        
        if component_scores['prediction_quality'] < 60:
            recommendations.append("予測品質向上のため、アンサンブル学習の強化を推奨")
        
        if component_scores['performance_efficiency'] < 60:
            recommendations.append("処理速度改善のため、並列処理とキャッシュ戦略の見直しを推奨")
        
        if component_scores['resource_optimization'] < 70:
            recommendations.append("リソース使用量削減のため、メモリ管理とCPU最適化を推奨")
        
        if component_scores['cache_performance'] < 50:
            recommendations.append("キャッシュ効率改善のため、キャッシュ戦略の調整を推奨")
        
        if not recommendations:
            recommendations.append("システムは良好に動作しています。継続的な監視を推奨")
        
        return recommendations
    
    async def run_integrated_test(self, test_name: str = "統合テスト") -> IntegratedTestResult:
        """統合テスト実行"""
        self.logger.info(f"統合テスト開始: {test_name}")
        start_time = datetime.now()
        
        try:
            # テストデータ作成
            self.logger.info("テストデータ作成中...")
            test_data = await self.create_test_dataset(5000)
            
            # 予測精度テスト
            self.logger.info("予測精度テスト実行中...")
            prediction_metrics = await self.run_prediction_accuracy_test(test_data)
            
            # パフォーマンステスト
            self.logger.info("パフォーマンステスト実行中...")
            performance_metrics = await self.run_performance_test(test_data)
            
            # 総合評価
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            overall_score, component_scores = self.calculate_overall_score(
                prediction_metrics, performance_metrics
            )
            
            recommendations = self.generate_recommendations(component_scores)
            
            # 結果作成
            result = IntegratedTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                prediction_accuracy=prediction_metrics['accuracy'],
                prediction_precision=prediction_metrics['precision'],
                prediction_recall=prediction_metrics['recall'],
                prediction_f1_score=prediction_metrics['f1_score'],
                cpu_usage_percent=performance_metrics['cpu_usage'],
                memory_usage_mb=performance_metrics['memory_usage'],
                processing_speed_records_per_second=performance_metrics['processing_speed'],
                cache_hit_rate=performance_metrics['cache_hit_rate'],
                overall_system_score=overall_score,
                component_scores=component_scores,
                recommendations=recommendations,
                success=True
            )
            
            self.test_results.append(result)
            self.logger.info(f"統合テスト完了: {test_name} (スコア: {overall_score:.1f})")
            
            return result
            
        except Exception as e:
            error_msg = f"統合テストエラー: {str(e)}"
            self.logger.error(error_msg)
            
            result = IntegratedTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                prediction_accuracy=0.0,
                prediction_precision=0.0,
                prediction_recall=0.0,
                prediction_f1_score=0.0,
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                processing_speed_records_per_second=0.0,
                cache_hit_rate=0.0,
                overall_system_score=0.0,
                component_scores={},
                recommendations=[],
                success=False,
                error_message=error_msg
            )
            
            return result
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """包括的テストスイート実行"""
        self.logger.info("包括的テストスイート開始")
        
        test_scenarios = [
            "基本統合テスト",
            "高負荷テスト",
            "精度最適化テスト"
        ]
        
        suite_results = []
        
        for scenario in test_scenarios:
            result = await self.run_integrated_test(scenario)
            suite_results.append(result)
        
        # 統計サマリー
        successful_tests = [r for r in suite_results if r.success]
        
        summary = {
            'total_tests': len(suite_results),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(suite_results) * 100,
            'average_score': statistics.mean([r.overall_system_score for r in successful_tests]) if successful_tests else 0,
            'average_prediction_accuracy': statistics.mean([r.prediction_accuracy for r in successful_tests]) if successful_tests else 0,
            'average_processing_speed': statistics.mean([r.processing_speed_records_per_second for r in successful_tests]) if successful_tests else 0,
            'test_results': [asdict(r) for r in suite_results],
            'overall_recommendations': self._compile_recommendations(suite_results)
        }
        
        self.logger.info(f"テストスイート完了 - 成功率: {summary['success_rate']:.1f}%")
        
        return summary
    
    def _compile_recommendations(self, results: List[IntegratedTestResult]) -> List[str]:
        """推奨事項統合"""
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # 重複除去と頻度順ソート
        from collections import Counter
        rec_counter = Counter(all_recommendations)
        
        return [rec for rec, count in rec_counter.most_common()]
    
    async def export_results(self, filepath: str = None) -> str:
        """結果エクスポート"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"integrated_test_results_{timestamp}.json"
        
        # 包括的テスト実行
        comprehensive_results = await self.run_comprehensive_test_suite()
        
        # ファイル保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"テスト結果エクスポート完了: {filepath}")
        return filepath


async def demo_integrated_testing():
    """統合テストデモ実行"""
    print("=== Day Trade 統合予測・パフォーマンステストシステム デモ ===")
    
    # テストシステム初期化
    test_system = IntegratedPredictionPerformanceTestSystem()
    
    try:
        # 単一テスト実行
        print("\n1. 単一統合テスト実行...")
        result = await test_system.run_integrated_test("デモテスト")
        
        print(f"\n--- テスト結果 ---")
        print(f"テスト名: {result.test_name}")
        print(f"実行時間: {result.duration_seconds:.2f}秒")
        print(f"成功: {'✓' if result.success else '✗'}")
        print(f"総合スコア: {result.overall_system_score:.1f}/100")
        
        print(f"\n--- 予測メトリクス ---")
        print(f"精度: {result.prediction_accuracy:.3f}")
        print(f"適合率: {result.prediction_precision:.3f}")
        print(f"再現率: {result.prediction_recall:.3f}")
        print(f"F1スコア: {result.prediction_f1_score:.3f}")
        
        print(f"\n--- パフォーマンスメトリクス ---")
        print(f"CPU使用率: {result.cpu_usage_percent:.1f}%")
        print(f"メモリ使用量: {result.memory_usage_mb:.1f}MB")
        print(f"処理速度: {result.processing_speed_records_per_second:.1f} レコード/秒")
        print(f"キャッシュヒット率: {result.cache_hit_rate:.3f}")
        
        print(f"\n--- コンポーネントスコア ---")
        for component, score in result.component_scores.items():
            print(f"{component}: {score:.1f}/100")
        
        print(f"\n--- 推奨事項 ---")
        for i, recommendation in enumerate(result.recommendations, 1):
            print(f"{i}. {recommendation}")
        
        # 結果エクスポート
        print(f"\n2. テスト結果エクスポート...")
        export_file = await test_system.export_results()
        print(f"エクスポート完了: {export_file}")
        
    except Exception as e:
        print(f"デモ実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_integrated_testing())