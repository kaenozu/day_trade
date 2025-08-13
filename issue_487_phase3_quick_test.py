#!/usr/bin/env python3
"""
Issue #487 Phase 3 クイック性能テスト

完全自動化システムの基本性能確認:
- システム初期化時間
- 基本処理速度
- メモリ使用量
"""

import asyncio
import time
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.day_trade.automation.smart_symbol_selector import SmartSymbolSelector, SelectionCriteria
from src.day_trade.automation.adaptive_optimization_system import AdaptiveOptimizationSystem, OptimizationConfig
from src.day_trade.automation.dynamic_risk_management_system import DynamicRiskManagementSystem, RiskManagementConfig
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

class Issue487Phase3QuickTest:
    """Issue #487 Phase 3 クイック性能テストクラス"""

    def __init__(self):
        self.start_time = datetime.now()
        self.test_results = {}

    def generate_simple_market_data(self, n_days=100):
        """シンプルな市場データ生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [1000.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r))

        return pd.DataFrame({
            'Date': dates,
            'Close': prices[:n_days],
            'High': [p * 1.01 for p in prices[:n_days]],
            'Low': [p * 0.99 for p in prices[:n_days]],
            'Volume': np.random.lognormal(10, 0.5, n_days)
        })

    async def run_quick_test(self):
        """クイック性能テスト実行"""
        print("=" * 80)
        print("Issue #487 Phase 3: クイック性能テスト")
        print("=" * 80)
        print(f"開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("テスト目標: 基本性能・初期化速度確認")
        print()

        # テスト実行
        await self._test_1_initialization()
        await self._test_2_market_regime_detection()
        await self._test_3_risk_calculation()
        await self._test_4_system_integration()

        # 結果表示
        self._show_results()

    async def _test_1_initialization(self):
        """テスト1: 初期化性能"""
        print("テスト1: システム初期化性能")
        print("-" * 50)

        try:
            start_time = time.time()
            initial_memory = psutil.virtual_memory().percent

            # システム初期化
            smart_selector = SmartSymbolSelector()
            adaptive_optimizer = AdaptiveOptimizationSystem(
                OptimizationConfig(n_trials=5, timeout=30)
            )
            risk_manager = DynamicRiskManagementSystem()

            init_time = time.time() - start_time
            final_memory = psutil.virtual_memory().percent
            memory_usage = final_memory - initial_memory

            # 評価
            if init_time < 15:
                status = 'EXCELLENT'
                score = 100
            elif init_time < 30:
                status = 'SUCCESS'
                score = 80
            else:
                status = 'SLOW'
                score = 60

            self.test_results['initialization'] = {
                'status': status,
                'init_time': init_time,
                'memory_usage': memory_usage,
                'score': score
            }

            print(f"[{status}] 初期化時間: {init_time:.1f}秒")
            print(f"   メモリ使用量増加: {memory_usage:.1f}%")

            # 次のテストのため保存
            self.smart_selector = smart_selector
            self.adaptive_optimizer = adaptive_optimizer
            self.risk_manager = risk_manager

        except Exception as e:
            self.test_results['initialization'] = {
                'status': 'FAILED',
                'error': str(e),
                'score': 0
            }
            print(f"[FAILED] 初期化エラー: {e}")

        print()

    async def _test_2_market_regime_detection(self):
        """テスト2: 市場レジーム検出"""
        print("テスト2: 市場レジーム検出性能")
        print("-" * 50)

        try:
            if not hasattr(self, 'adaptive_optimizer'):
                raise Exception("適応的最適化システムが初期化されていません")

            # テストデータ生成
            market_data = self.generate_simple_market_data(n_days=100)

            # レジーム検出実行
            start_time = time.time()
            regime_metrics = self.adaptive_optimizer.detect_market_regime(market_data)
            detection_time = time.time() - start_time

            # 評価
            if detection_time < 1.0 and regime_metrics.regime.value != 'unknown':
                status = 'EXCELLENT'
                score = 100
            elif detection_time < 3.0 and regime_metrics.regime.value != 'unknown':
                status = 'SUCCESS'
                score = 80
            else:
                status = 'SLOW'
                score = 60

            self.test_results['regime_detection'] = {
                'status': status,
                'detection_time': detection_time,
                'regime': regime_metrics.regime.value,
                'confidence': regime_metrics.confidence,
                'score': score
            }

            print(f"[{status}] レジーム検出時間: {detection_time:.2f}秒")
            print(f"   検出レジーム: {regime_metrics.regime.value}")
            print(f"   信頼度: {regime_metrics.confidence:.2f}")

        except Exception as e:
            self.test_results['regime_detection'] = {
                'status': 'FAILED',
                'error': str(e),
                'score': 0
            }
            print(f"[FAILED] レジーム検出エラー: {e}")

        print()

    async def _test_3_risk_calculation(self):
        """テスト3: リスク計算"""
        print("テスト3: リスク計算性能")
        print("-" * 50)

        try:
            if not hasattr(self, 'risk_manager'):
                raise Exception("リスク管理システムが初期化されていません")

            # テストデータ
            market_data = self.generate_simple_market_data(n_days=100)

            # リスク計算実行
            start_time = time.time()
            risk_metrics = await self.risk_manager.calculate_risk_metrics("TEST_SYMBOL", market_data)
            risk_time = time.time() - start_time

            # ストップロス計算
            stop_start = time.time()
            stop_loss = await self.risk_manager.calculate_dynamic_stop_loss("TEST_SYMBOL", market_data, 1000.0)
            stop_time = time.time() - stop_start

            total_risk_time = risk_time + stop_time

            # 評価
            if total_risk_time < 2.0:
                status = 'EXCELLENT'
                score = 100
            elif total_risk_time < 5.0:
                status = 'SUCCESS'
                score = 80
            else:
                status = 'SLOW'
                score = 60

            self.test_results['risk_calculation'] = {
                'status': status,
                'risk_time': risk_time,
                'stop_time': stop_time,
                'total_time': total_risk_time,
                'var_95': risk_metrics.var_95,
                'stop_loss_pct': stop_loss.stop_loss_pct,
                'score': score
            }

            print(f"[{status}] リスク計算時間: {total_risk_time:.2f}秒")
            print(f"   VaR計算: {risk_time:.2f}秒")
            print(f"   ストップロス計算: {stop_time:.2f}秒")
            print(f"   VaR(95%): {risk_metrics.var_95:.4f}")
            print(f"   ストップロス: {stop_loss.stop_loss_pct*100:.2f}%")

        except Exception as e:
            self.test_results['risk_calculation'] = {
                'status': 'FAILED',
                'error': str(e),
                'score': 0
            }
            print(f"[FAILED] リスク計算エラー: {e}")

        print()

    async def _test_4_system_integration(self):
        """テスト4: システム統合"""
        print("テスト4: システム統合性能")
        print("-" * 50)

        try:
            # 統合フロー実行
            start_time = time.time()

            # 1. 銘柄選択
            criteria = SelectionCriteria(target_symbols=2)
            selected_symbols = await self.smart_selector.select_optimal_symbols(criteria)

            # 2. レジーム検出
            market_data = self.generate_simple_market_data()
            regime_metrics = self.adaptive_optimizer.detect_market_regime(market_data)

            # 3. リスク計算
            if selected_symbols:
                symbol = selected_symbols[0]
                risk_metrics = await self.risk_manager.calculate_risk_metrics(symbol, market_data)

            integration_time = time.time() - start_time

            # 評価
            success_factors = [
                selected_symbols and len(selected_symbols) > 0,
                regime_metrics.regime.value != 'unknown',
                'risk_metrics' in locals()
            ]

            success_rate = sum(success_factors) / len(success_factors) * 100

            if integration_time < 10 and success_rate >= 100:
                status = 'EXCELLENT'
                score = 100
            elif integration_time < 20 and success_rate >= 80:
                status = 'SUCCESS'
                score = 80
            else:
                status = 'PARTIAL'
                score = 60

            self.test_results['integration'] = {
                'status': status,
                'integration_time': integration_time,
                'success_rate': success_rate,
                'selected_count': len(selected_symbols) if selected_symbols else 0,
                'score': score
            }

            print(f"[{status}] 統合実行時間: {integration_time:.1f}秒")
            print(f"   成功率: {success_rate:.0f}%")
            print(f"   選定銘柄数: {len(selected_symbols) if selected_symbols else 0}")

        except Exception as e:
            self.test_results['integration'] = {
                'status': 'FAILED',
                'error': str(e),
                'score': 0
            }
            print(f"[FAILED] 統合テストエラー: {e}")

        print()

    def _show_results(self):
        """結果表示"""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()

        print("=" * 80)
        print("Issue #487 Phase 3 クイック性能テスト結果")
        print("=" * 80)

        # 個別結果
        print("個別テスト結果:")
        total_score = 0
        max_score = 0

        test_names = {
            'initialization': 'システム初期化',
            'regime_detection': '市場レジーム検出',
            'risk_calculation': 'リスク計算',
            'integration': 'システム統合'
        }

        for test_key, test_name in test_names.items():
            result = self.test_results.get(test_key, {'status': 'NOT_RUN', 'score': 0})
            score = result.get('score', 0)
            status = result.get('status', 'UNKNOWN')

            status_icon = {
                'EXCELLENT': '[EXCELLENT]',
                'SUCCESS': '[SUCCESS]',
                'PARTIAL': '[PARTIAL]',
                'SLOW': '[SLOW]',
                'FAILED': '[FAILED]',
                'NOT_RUN': '[NOT_RUN]'
            }.get(status, '[UNKNOWN]')

            print(f"  {status_icon} {test_name:20}: {score:3d}% ({status})")
            total_score += score
            max_score += 100

        # 総合評価
        overall_percentage = (total_score / max_score * 100) if max_score > 0 else 0

        print(f"\n総合結果:")
        print(f"   クイック性能スコア: {total_score}/{max_score} ({overall_percentage:.1f}%)")
        print(f"   総実行時間: {total_time:.1f}秒")
        print(f"   テスト日時: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 最終判定
        if overall_percentage >= 90:
            final_status = "EXCELLENT SUCCESS"
            final_message = "Issue #487 Phase 3 クイック性能テスト優秀！"
        elif overall_percentage >= 75:
            final_status = "SUCCESS"
            final_message = "Issue #487 Phase 3 基本性能良好"
        elif overall_percentage >= 60:
            final_status = "PARTIAL SUCCESS"
            final_message = "Issue #487 Phase 3 最適化の余地あり"
        else:
            final_status = "NEEDS IMPROVEMENT"
            final_message = "Issue #487 Phase 3 性能改善必要"

        print(f"\n{final_status}")
        print(f"{final_message}")

        # システムリソース情報
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()

        print(f"\n現在のシステム状態:")
        print(f"   CPU使用率: {cpu_percent:.1f}%")
        print(f"   メモリ使用率: {memory_info.percent:.1f}%")
        print(f"   利用可能メモリ: {memory_info.available / 1024**3:.1f}GB")

        print("=" * 80)

async def main():
    """メイン実行関数"""
    tester = Issue487Phase3QuickTest()
    await tester.run_quick_test()

if __name__ == "__main__":
    asyncio.run(main())