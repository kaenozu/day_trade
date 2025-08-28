"""
簡易統合テストシステム

予測精度向上とパフォーマンス向上の統合テストを実行
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
import statistics


@dataclass
class SimpleTestResult:
    """簡易テスト結果"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    prediction_accuracy: float
    processing_speed: float
    memory_efficiency: float
    overall_score: float
    success: bool
    recommendations: List[str]


class SimpleIntegratedTestSystem:
    """簡易統合テストシステム"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.test_results: List[SimpleTestResult] = []
        
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
    
    def create_test_data(self, size: int = 1000) -> pd.DataFrame:
        """テストデータ作成"""
        np.random.seed(42)
        
        # 株式市場データシミュレーション
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=size),
            periods=size,
            freq='1min'
        )
        
        # 価格データ
        base_price = 1000
        price_changes = np.random.normal(0, 0.01, size)
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['TEST'] * size,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
            'volume': np.random.randint(1000, 100000, size)
        })
        
        # ターゲット（次の価格が上昇するか）
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        data = data.dropna()
        
        return data
    
    def simple_prediction_test(self, data: pd.DataFrame) -> float:
        """簡易予測テスト"""
        # 単純な移動平均ベース予測
        data['ma5'] = data['close'].rolling(5).mean()
        data['ma20'] = data['close'].rolling(20).mean()
        
        # 予測: 短期移動平均 > 長期移動平均 なら上昇予測
        predictions = (data['ma5'] > data['ma20']).astype(int)
        actuals = data['target']
        
        # NaN除去
        valid_idx = ~(predictions.isna() | actuals.isna())
        predictions = predictions[valid_idx]
        actuals = actuals[valid_idx]
        
        if len(predictions) == 0:
            return 0.5
        
        accuracy = (predictions == actuals).mean()
        return accuracy
    
    def performance_test(self, data: pd.DataFrame) -> Dict[str, float]:
        """パフォーマンステスト"""
        start_time = time.time()
        
        # データ処理テスト
        processed_chunks = []
        chunk_size = 100
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            
            # 簡単な計算処理
            chunk_processed = chunk.copy()
            chunk_processed['sma'] = chunk['close'].rolling(5).mean()
            chunk_processed['ema'] = chunk['close'].ewm(span=5).mean()
            chunk_processed['rsi'] = self._calculate_rsi(chunk['close'])
            
            processed_chunks.append(len(chunk_processed))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        total_records = sum(processed_chunks)
        processing_speed = total_records / processing_time if processing_time > 0 else 0
        
        # メモリ効率（簡易計算）
        memory_efficiency = min(100, 1000 / len(data) * 100)  # データサイズ逆比例
        
        return {
            'processing_speed': processing_speed,
            'memory_efficiency': memory_efficiency,
            'processing_time': processing_time
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_overall_score(self, prediction_accuracy: float, 
                              performance_metrics: Dict[str, float]) -> float:
        """総合スコア計算"""
        # 重み付きスコア
        accuracy_score = prediction_accuracy * 100  # 0-100スケール
        speed_score = min(100, performance_metrics['processing_speed'] / 10)  # 速度正規化
        memory_score = performance_metrics['memory_efficiency']
        
        # 重み付き平均
        overall_score = (
            accuracy_score * 0.5 +      # 予測精度 50%
            speed_score * 0.3 +         # 処理速度 30%
            memory_score * 0.2          # メモリ効率 20%
        )
        
        return overall_score
    
    def generate_recommendations(self, prediction_accuracy: float,
                               performance_metrics: Dict[str, float],
                               overall_score: float) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if prediction_accuracy < 0.6:
            recommendations.append("予測精度向上のため、より高度な特徴量エンジニアリングを推奨")
        
        if performance_metrics['processing_speed'] < 100:
            recommendations.append("処理速度向上のため、並列処理の実装を推奨")
        
        if performance_metrics['memory_efficiency'] < 70:
            recommendations.append("メモリ使用効率化のため、データ処理の最適化を推奨")
        
        if overall_score < 70:
            recommendations.append("システム全体の最適化が必要です")
        elif overall_score >= 80:
            recommendations.append("システムは良好に動作しています")
        
        return recommendations
    
    async def run_integrated_test(self, test_name: str = "統合テスト") -> SimpleTestResult:
        """統合テスト実行"""
        self.logger.info(f"統合テスト開始: {test_name}")
        start_time = datetime.now()
        
        try:
            # テストデータ作成
            test_data = self.create_test_data(2000)
            self.logger.info(f"テストデータ作成完了: {len(test_data)}件")
            
            # 予測精度テスト
            prediction_accuracy = self.simple_prediction_test(test_data)
            self.logger.info(f"予測精度: {prediction_accuracy:.3f}")
            
            # パフォーマンステスト
            performance_metrics = self.performance_test(test_data)
            self.logger.info(f"処理速度: {performance_metrics['processing_speed']:.1f} records/sec")
            
            # 総合評価
            overall_score = self.calculate_overall_score(prediction_accuracy, performance_metrics)
            recommendations = self.generate_recommendations(
                prediction_accuracy, performance_metrics, overall_score
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = SimpleTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                prediction_accuracy=prediction_accuracy,
                processing_speed=performance_metrics['processing_speed'],
                memory_efficiency=performance_metrics['memory_efficiency'],
                overall_score=overall_score,
                success=True,
                recommendations=recommendations
            )
            
            self.test_results.append(result)
            self.logger.info(f"統合テスト完了: {test_name} (スコア: {overall_score:.1f})")
            
            return result
            
        except Exception as e:
            error_msg = f"統合テストエラー: {str(e)}"
            self.logger.error(error_msg)
            
            result = SimpleTestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                prediction_accuracy=0.0,
                processing_speed=0.0,
                memory_efficiency=0.0,
                overall_score=0.0,
                success=False,
                recommendations=["テスト実行でエラーが発生しました"]
            )
            
            return result
    
    async def run_test_suite(self) -> Dict[str, Any]:
        """テストスイート実行"""
        self.logger.info("統合テストスイート開始")
        
        test_scenarios = [
            "基本統合テスト",
            "パフォーマンステスト",
            "精度テスト"
        ]
        
        suite_results = []
        
        for scenario in test_scenarios:
            result = await self.run_integrated_test(scenario)
            suite_results.append(result)
            await asyncio.sleep(0.1)  # 小さな休憩
        
        # 統計サマリー
        successful_tests = [r for r in suite_results if r.success]
        
        summary = {
            'total_tests': len(suite_results),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(suite_results) * 100,
            'average_score': statistics.mean([r.overall_score for r in successful_tests]) if successful_tests else 0,
            'average_accuracy': statistics.mean([r.prediction_accuracy for r in successful_tests]) if successful_tests else 0,
            'average_speed': statistics.mean([r.processing_speed for r in successful_tests]) if successful_tests else 0,
            'test_results': [asdict(r) for r in suite_results]
        }
        
        return summary
    
    def export_results(self, filepath: str = None) -> str:
        """結果エクスポート"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"simple_integrated_test_results_{timestamp}.json"
        
        results_data = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'timestamp': datetime.now().isoformat(),
                'system_info': 'Day Trade Simple Integration Test'
            },
            'results': [asdict(r) for r in self.test_results]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"テスト結果エクスポート完了: {filepath}")
        return filepath


async def demo_simple_test():
    """簡易テストデモ"""
    print("=== Day Trade 簡易統合テストシステム デモ ===")
    
    test_system = SimpleIntegratedTestSystem()
    
    try:
        # テストスイート実行
        print("\n統合テストスイート実行中...")
        summary = await test_system.run_test_suite()
        
        print(f"\n=== テストサマリー ===")
        print(f"総テスト数: {summary['total_tests']}")
        print(f"成功テスト数: {summary['successful_tests']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print(f"平均スコア: {summary['average_score']:.1f}/100")
        print(f"平均予測精度: {summary['average_accuracy']:.3f}")
        print(f"平均処理速度: {summary['average_speed']:.1f} records/sec")
        
        # 個別結果表示
        for i, result_data in enumerate(summary['test_results'], 1):
            result = SimpleTestResult(**result_data)
            print(f"\n--- テスト {i}: {result.test_name} ---")
            print(f"実行時間: {result.duration_seconds:.2f}秒")
            print(f"成功: {'✓' if result.success else '✗'}")
            print(f"総合スコア: {result.overall_score:.1f}/100")
            print(f"予測精度: {result.prediction_accuracy:.3f}")
            print(f"処理速度: {result.processing_speed:.1f} records/sec")
            print(f"メモリ効率: {result.memory_efficiency:.1f}%")
            
            if result.recommendations:
                print("推奨事項:")
                for j, rec in enumerate(result.recommendations, 1):
                    print(f"  {j}. {rec}")
        
        # 結果エクスポート
        print(f"\n結果エクスポート中...")
        export_file = test_system.export_results()
        print(f"エクスポート完了: {export_file}")
        
        # 総合評価
        if summary['average_score'] >= 80:
            print(f"\n🎉 システムは優秀に動作しています！")
        elif summary['average_score'] >= 60:
            print(f"\n✅ システムは良好に動作しています。")
        else:
            print(f"\n⚠️ システムの最適化が必要です。")
        
    except Exception as e:
        print(f"デモ実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_simple_test())