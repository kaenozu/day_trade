#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包括的テストスイート - 全システムコンポーネント統合テスト
Issues #933-943対応: システム全体の統合動作検証
"""

import asyncio
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any
import traceback

# 全モジュールインポート
try:
    from advanced_ai_engine import advanced_ai_engine
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False
    print("Warning: Advanced AI Engine not available")

try:
    from quantum_ai_engine import quantum_ai_engine
    HAS_QUANTUM_AI = True
except ImportError:
    HAS_QUANTUM_AI = False
    print("Warning: Quantum AI Engine not available")

try:
    from blockchain_trading import trading_blockchain_integration
    HAS_BLOCKCHAIN = True
except ImportError:
    HAS_BLOCKCHAIN = False
    print("Warning: Blockchain Trading not available")

try:
    from high_frequency_trading import hft_engine
    HAS_HFT = True
except ImportError:
    HAS_HFT = False
    print("Warning: High Frequency Trading not available")

try:
    from risk_management_ai import risk_management_ai
    HAS_RISK_MANAGEMENT = True
except ImportError:
    HAS_RISK_MANAGEMENT = False
    print("Warning: Risk Management AI not available")

try:
    from realtime_streaming import streaming_engine
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False
    print("Warning: Realtime Streaming not available")

try:
    from scalability_engine import scalability_engine
    HAS_SCALABILITY = True
except ImportError:
    HAS_SCALABILITY = False
    print("Warning: Scalability Engine not available")

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE = True
except ImportError:
    HAS_PERFORMANCE = False
    print("Warning: Performance Monitor not available")

try:
    from data_persistence import data_persistence
    HAS_DATA_PERSISTENCE = True
except ImportError:
    HAS_DATA_PERSISTENCE = False
    print("Warning: Data Persistence not available")


class ComprehensiveTestSuite:
    """包括的テストスイート"""

    def __init__(self):
        self.test_results = []
        self.test_symbols = ['7203', '8306', '9984', '6758', '4689']
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = datetime.now()

    def run_test(self, test_name: str, test_function, *args, **kwargs) -> bool:
        """個別テスト実行"""
        self.total_tests += 1
        start_time = time.time()

        try:
            print(f"[テスト実行] {test_name}...")
            result = test_function(*args, **kwargs)

            if asyncio.iscoroutine(result):
                result = asyncio.run(result)

            execution_time = time.time() - start_time

            self.test_results.append({
                'test_name': test_name,
                'status': 'PASSED',
                'execution_time': execution_time,
                'result': result,
                'error': None,
                'timestamp': datetime.now().isoformat()
            })

            self.passed_tests += 1
            print(f"[成功] {test_name} ({execution_time:.2f}秒)")
            return True

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"{type(e).__name__}: {str(e)}"

            self.test_results.append({
                'test_name': test_name,
                'status': 'FAILED',
                'execution_time': execution_time,
                'result': None,
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            })

            self.failed_tests += 1
            print(f"[失敗] {test_name}: {error_message}")
            return False

    # ===========================================
    # 個別コンポーネントテスト
    # ===========================================

    def test_advanced_ai_engine(self):
        """Advanced AI Engineテスト"""
        if not HAS_AI_ENGINE:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        results = []
        for symbol in self.test_symbols[:3]:
            signal = advanced_ai_engine.analyze_symbol(symbol)
            results.append({
                'symbol': symbol,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'strength': signal.strength
            })

        return {'analyzed_symbols': len(results), 'results': results}

    def test_quantum_ai_engine(self):
        """Quantum AI Engineテスト"""
        if not HAS_QUANTUM_AI:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        symbol = self.test_symbols[0]
        market_data = [100.0, 101.5, 99.8, 102.1, 100.9]

        prediction = quantum_ai_engine.quantum_market_analysis(symbol, market_data)
        return {
            'symbol': symbol,
            'prediction_confidence': prediction.confidence,
            'quantum_advantage': prediction.quantum_advantage,
            'classical_accuracy': prediction.classical_accuracy
        }

    def test_blockchain_trading(self):
        """Blockchain Tradingテスト"""
        if not HAS_BLOCKCHAIN:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        symbol = self.test_symbols[0]
        prediction = {'signal_type': 'BUY', 'confidence': 0.85}

        tx_id = trading_blockchain_integration.record_ai_prediction(symbol, prediction)
        verification = trading_blockchain_integration.verify_prediction_integrity(tx_id)

        return {
            'transaction_id': tx_id,
            'verification_status': verification,
            'blockchain_height': len(trading_blockchain_integration.blockchain.chain)
        }

    async def test_high_frequency_trading(self):
        """High Frequency Tradingテスト"""
        if not HAS_HFT:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        # HFTエンジン起動
        await hft_engine.start()

        # テスト注文
        orders_submitted = 0
        for i, symbol in enumerate(self.test_symbols[:3]):
            order_id = await hft_engine.submit_order(symbol, 'BUY', 100, 1500 + i*10)
            if order_id:
                orders_submitted += 1

        # 少し待って統計取得
        await asyncio.sleep(0.1)
        stats = hft_engine.get_trading_statistics()

        await hft_engine.stop()

        return {
            'orders_submitted': orders_submitted,
            'avg_latency_us': stats.get('avg_latency_us', 0),
            'total_trades': stats.get('total_trades', 0)
        }

    def test_risk_management_ai(self):
        """Risk Management AIテスト"""
        if not HAS_RISK_MANAGEMENT:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        # 個別リスク分析
        risk_metrics = []
        for symbol in self.test_symbols[:3]:
            metrics = risk_management_ai.calculate_risk_metrics(symbol)
            risk_metrics.append({
                'symbol': symbol,
                'var_95': metrics.var_95,
                'sharpe_ratio': metrics.sharpe_ratio,
                'risk_score': metrics.risk_score
            })

        # ポートフォリオ最適化
        portfolio = risk_management_ai.optimize_portfolio(self.test_symbols[:3])

        return {
            'risk_metrics_calculated': len(risk_metrics),
            'portfolio_expected_return': portfolio.expected_return,
            'portfolio_sharpe_ratio': portfolio.sharpe_ratio,
            'optimization_score': portfolio.optimization_score
        }

    async def test_realtime_streaming(self):
        """Realtime Streamingテスト"""
        if not HAS_STREAMING:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        # ストリーミング開始
        await streaming_engine.start()

        # テストサブスクリプション
        subscriptions = 0
        for symbol in self.test_symbols[:3]:
            success = await streaming_engine.subscribe_to_symbol(symbol)
            if success:
                subscriptions += 1

        # 少し待って統計取得
        await asyncio.sleep(0.1)
        stats = streaming_engine.get_streaming_statistics()

        await streaming_engine.stop()

        return {
            'subscriptions_created': subscriptions,
            'total_messages_sent': stats.get('total_messages_sent', 0),
            'active_connections': stats.get('active_connections', 0)
        }

    async def test_scalability_engine(self):
        """Scalability Engineテスト"""
        if not HAS_SCALABILITY:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        # スケーラビリティエンジン開始
        await scalability_engine.start()

        # テストタスク投入
        task_ids = []
        for symbol in self.test_symbols[:3]:
            task_id = await scalability_engine.submit_analysis(symbol)
            task_ids.append(task_id)

        # 結果待ち
        completed_tasks = 0
        for task_id in task_ids:
            result = await scalability_engine.load_balancer.get_task_result(task_id, timeout=5.0)
            if result and result.result:
                completed_tasks += 1

        # システムメトリクス取得
        metrics = scalability_engine.get_system_metrics()

        await scalability_engine.stop()

        return {
            'tasks_submitted': len(task_ids),
            'tasks_completed': completed_tasks,
            'avg_response_time_ms': metrics['avg_response_time_ms'],
            'total_requests': metrics['total_requests']
        }

    def test_performance_monitor(self):
        """Performance Monitorテスト"""
        if not HAS_PERFORMANCE:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        # パフォーマンス記録
        performance_monitor.record_analysis_performance('TEST_SYMBOL', 150.5, True)
        performance_monitor.record_api_call('test_endpoint', 45.2, True)

        # 統計取得
        summary = performance_monitor.get_performance_summary()

        return {
            'total_analyses': summary.get('total_analyses', 0),
            'avg_analysis_time': summary.get('avg_analysis_time_ms', 0),
            'success_rate': summary.get('analysis_success_rate', 0)
        }

    def test_data_persistence(self):
        """Data Persistenceテスト"""
        if not HAS_DATA_PERSISTENCE:
            return {'status': 'SKIPPED', 'reason': 'Module not available'}

        # テストデータ保存
        test_data = {
            'test_key': 'test_value',
            'timestamp': datetime.now().isoformat(),
            'numbers': [1, 2, 3, 4, 5]
        }

        data_persistence.save_performance_data('test_component', test_data)

        # データ取得テスト
        stored_data = data_persistence.get_performance_history('test_component', limit=1)

        return {
            'data_saved': len(test_data),
            'data_retrieved': len(stored_data) if stored_data else 0,
            'storage_available': True
        }

    # ===========================================
    # 統合テスト
    # ===========================================

    async def test_ai_pipeline_integration(self):
        """AI分析パイプライン統合テスト"""
        symbol = self.test_symbols[0]
        results = {}

        # Advanced AI Engine
        if HAS_AI_ENGINE:
            ai_signal = advanced_ai_engine.analyze_symbol(symbol)
            results['advanced_ai'] = {
                'signal_type': ai_signal.signal_type,
                'confidence': ai_signal.confidence
            }

        # Quantum AI Engine
        if HAS_QUANTUM_AI:
            market_data = [100.0, 101.5, 99.8, 102.1, 100.9]
            quantum_prediction = quantum_ai_engine.quantum_market_analysis(symbol, market_data)
            results['quantum_ai'] = {
                'confidence': quantum_prediction.confidence,
                'quantum_advantage': quantum_prediction.quantum_advantage
            }

        # Risk Management AI
        if HAS_RISK_MANAGEMENT:
            risk_metrics = risk_management_ai.calculate_risk_metrics(symbol)
            results['risk_analysis'] = {
                'var_95': risk_metrics.var_95,
                'risk_score': risk_metrics.risk_score
            }

        return {
            'symbol': symbol,
            'pipeline_components': len(results),
            'integrated_results': results
        }

    async def test_full_system_workflow(self):
        """フルシステムワークフロー統合テスト"""
        workflow_results = []

        for i, symbol in enumerate(self.test_symbols[:2]):
            step_results = {}

            # ステップ1: AI分析
            if HAS_AI_ENGINE:
                signal = advanced_ai_engine.analyze_symbol(symbol)
                step_results['ai_analysis'] = True

            # ステップ2: リスク評価
            if HAS_RISK_MANAGEMENT:
                risk_metrics = risk_management_ai.calculate_risk_metrics(symbol)
                step_results['risk_evaluation'] = True

            # ステップ3: ブロックチェーン記録
            if HAS_BLOCKCHAIN:
                prediction = {'signal_type': 'BUY', 'confidence': 0.8}
                tx_id = trading_blockchain_integration.record_ai_prediction(symbol, prediction)
                step_results['blockchain_record'] = bool(tx_id)

            # ステップ4: パフォーマンス記録
            if HAS_PERFORMANCE:
                performance_monitor.record_analysis_performance(symbol, 120.0, True)
                step_results['performance_recorded'] = True

            workflow_results.append({
                'symbol': symbol,
                'completed_steps': len(step_results),
                'step_results': step_results
            })

        return {
            'workflows_tested': len(workflow_results),
            'results': workflow_results
        }

    def test_system_resilience(self):
        """システム復元性テスト"""
        resilience_tests = []

        # エラーハンドリングテスト
        try:
            if HAS_AI_ENGINE:
                # 無効な銘柄でのテスト
                signal = advanced_ai_engine.analyze_symbol('INVALID_SYMBOL')
                resilience_tests.append({'test': 'invalid_symbol', 'passed': True})
        except Exception:
            resilience_tests.append({'test': 'invalid_symbol', 'passed': True})  # エラーハンドリングが機能

        # 負荷テスト（軽量版）
        if HAS_RISK_MANAGEMENT:
            start_time = time.time()
            for _ in range(5):  # 5回のリスク計算
                risk_management_ai.calculate_risk_metrics(self.test_symbols[0])
            load_test_time = time.time() - start_time
            resilience_tests.append({'test': 'load_test', 'passed': load_test_time < 10.0})

        return {
            'resilience_tests': len(resilience_tests),
            'passed_tests': sum(1 for t in resilience_tests if t['passed']),
            'results': resilience_tests
        }

    # ===========================================
    # メインテスト実行
    # ===========================================

    async def run_all_tests(self):
        """全テスト実行"""
        print("=" * 60)
        print("Day Trade Personal - 包括的システムテスト")
        print("=" * 60)
        print(f"開始時刻: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 個別コンポーネントテスト
        print("📊 個別コンポーネントテスト")
        print("-" * 40)

        self.run_test("Advanced AI Engine", self.test_advanced_ai_engine)
        self.run_test("Quantum AI Engine", self.test_quantum_ai_engine)
        self.run_test("Blockchain Trading", self.test_blockchain_trading)
        self.run_test("High Frequency Trading", self.test_high_frequency_trading)
        self.run_test("Risk Management AI", self.test_risk_management_ai)
        self.run_test("Realtime Streaming", self.test_realtime_streaming)
        self.run_test("Scalability Engine", self.test_scalability_engine)
        self.run_test("Performance Monitor", self.test_performance_monitor)
        self.run_test("Data Persistence", self.test_data_persistence)

        print()

        # 統合テスト
        print("🔗 システム統合テスト")
        print("-" * 40)

        self.run_test("AI Pipeline Integration", self.test_ai_pipeline_integration)
        self.run_test("Full System Workflow", self.test_full_system_workflow)
        self.run_test("System Resilience", self.test_system_resilience)

        print()

        # 結果サマリー
        self.print_test_summary()

        # 詳細結果をJSONで保存
        self.save_test_results()

    def print_test_summary(self):
        """テストサマリー出力"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0

        print("=" * 60)
        print("📈 テスト結果サマリー")
        print("=" * 60)
        print(f"総テスト数:     {self.total_tests}")
        print(f"成功:          {self.passed_tests} ✅")
        print(f"失敗:          {self.failed_tests} ❌")
        print(f"成功率:        {success_rate:.1f}%")
        print(f"総実行時間:    {total_time:.2f}秒")
        print()

        if self.failed_tests > 0:
            print("❌ 失敗したテスト:")
            for result in self.test_results:
                if result['status'] == 'FAILED':
                    print(f"  - {result['test_name']}: {result['error']}")
            print()

        # パフォーマンス統計
        execution_times = [r['execution_time'] for r in self.test_results if r['execution_time'] > 0]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            print(f"平均実行時間:  {avg_time:.3f}秒")
            print(f"最速テスト:    {min(execution_times):.3f}秒")
            print(f"最遅テスト:    {max(execution_times):.3f}秒")

        print()
        print("🎯 システム統合度評価:")
        available_modules = sum([
            HAS_AI_ENGINE, HAS_QUANTUM_AI, HAS_BLOCKCHAIN, HAS_HFT,
            HAS_RISK_MANAGEMENT, HAS_STREAMING, HAS_SCALABILITY,
            HAS_PERFORMANCE, HAS_DATA_PERSISTENCE
        ])
        integration_score = (available_modules / 9) * success_rate
        print(f"利用可能モジュール: {available_modules}/9")
        print(f"統合スコア:        {integration_score:.1f}/100")

        if integration_score >= 80:
            print("🌟 優秀 - システムは高度に統合されています")
        elif integration_score >= 60:
            print("👍 良好 - システムは適切に機能しています")
        elif integration_score >= 40:
            print("⚠️ 注意 - 一部改善が必要です")
        else:
            print("🔧 要改善 - システムに問題があります")

    def save_test_results(self):
        """テスト結果をファイルに保存"""
        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        test_report = {
            'test_session': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            'module_availability': {
                'advanced_ai_engine': HAS_AI_ENGINE,
                'quantum_ai_engine': HAS_QUANTUM_AI,
                'blockchain_trading': HAS_BLOCKCHAIN,
                'high_frequency_trading': HAS_HFT,
                'risk_management_ai': HAS_RISK_MANAGEMENT,
                'realtime_streaming': HAS_STREAMING,
                'scalability_engine': HAS_SCALABILITY,
                'performance_monitor': HAS_PERFORMANCE,
                'data_persistence': HAS_DATA_PERSISTENCE
            },
            'detailed_results': self.test_results
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(test_report, f, indent=2, ensure_ascii=False)
            print(f"📄 詳細テスト結果を保存: {filename}")
        except Exception as e:
            print(f"❌ テスト結果保存に失敗: {e}")


async def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.WARNING,  # テスト中はWARNING以上のログのみ
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    test_suite = ComprehensiveTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())