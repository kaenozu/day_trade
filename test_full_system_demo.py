#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - 完全統合システム動作検証・デモ
リアルタイムシステム全体の包括的な動作テスト

全コンポーネント統合運用による最終検証
"""

import asyncio
import time
import logging
import json
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# プロジェクト内インポート
from src.day_trade.utils.logging_config import get_context_logger
from src.day_trade.realtime.integration_manager import create_integration_manager, IntegrationConfig
from src.day_trade.realtime.websocket_stream import MarketTick, NewsItem, SocialPost

logger = get_context_logger(__name__)

class FullSystemValidator:
    """完全統合システム検証器"""

    def __init__(self):
        self.test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        self.validation_duration = 120  # 2分間の包括検証
        self.dashboard_port = 8080

        # 検証結果記録
        self.validation_results = {
            'start_time': None,
            'end_time': None,
            'total_duration_seconds': 0,
            'system_stability': 0.0,
            'predictions_generated': 0,
            'alerts_sent': 0,
            'dashboard_active': False,
            'components_status': {},
            'performance_metrics': {},
            'errors_encountered': [],
            'final_score': 0.0
        }

        logger.info("Full System Validator initialized")

    async def run_comprehensive_validation(self):
        """包括的システム検証"""

        print("=" * 60)
        print("Next-Gen AI Trading Engine - Full System Validation")
        print("=" * 60)
        print(f"Target Symbols: {self.test_symbols}")
        print(f"Validation Duration: {self.validation_duration} seconds")
        print(f"Dashboard URL: http://localhost:{self.dashboard_port}")
        print("=" * 60)

        self.validation_results['start_time'] = datetime.now()

        try:
            # Phase 1: システム初期化・起動
            print("\n[INIT] Phase 1: System Initialization")
            integration_manager = await self._initialize_complete_system()

            # Phase 2: 包括的運用テスト
            print("\n[TEST] Phase 2: Comprehensive Operation Test")
            await self._run_comprehensive_operation(integration_manager)

            # Phase 3: パフォーマンス検証
            print("\n[PERF] Phase 3: Performance Validation")
            await self._validate_system_performance(integration_manager)

            # Phase 4: システム停止・結果分析
            print("\n[DONE] Phase 4: System Shutdown & Analysis")
            await self._shutdown_and_analyze(integration_manager)

        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            self.validation_results['errors_encountered'].append(str(e))
            import traceback
            traceback.print_exc()

        finally:
            self.validation_results['end_time'] = datetime.now()
            if self.validation_results['start_time']:
                duration = self.validation_results['end_time'] - self.validation_results['start_time']
                self.validation_results['total_duration_seconds'] = duration.total_seconds()

        # 最終レポート生成
        self._generate_final_report()

        return self.validation_results

    async def _initialize_complete_system(self):
        """完全システム初期化"""

        print("  Creating integration manager...")

        # 統合管理システム設定
        config = IntegrationConfig(
            symbols=self.test_symbols,
            enable_streaming=True,
            enable_prediction=True,
            enable_monitoring=True,
            enable_alerts=True,
            enable_dashboard=True,
            dashboard_port=self.dashboard_port,
            update_interval=2.0,  # 2秒間隔
            detailed_logging=True
        )

        manager = create_integration_manager(
            symbols=self.test_symbols,
            dashboard_port=self.dashboard_port
        )

        print("  OK Integration manager created")
        print("  Initializing all components...")

        # システム初期化
        await manager.initialize_system()

        # コンポーネント状況確認
        components_status = {}

        if manager.stream_manager:
            components_status['streaming'] = 'initialized'
            print("  OK Streaming system ready")

        if manager.prediction_engine:
            components_status['prediction'] = 'initialized'
            print("  OK AI prediction engine ready")

        if manager.performance_monitor:
            components_status['monitoring'] = 'initialized'
            print("  OK Performance monitoring ready")

        if manager.alert_manager:
            components_status['alerts'] = 'initialized'
            print("  OK Alert system ready")

        if manager.dashboard_manager:
            components_status['dashboard'] = 'initialized'
            print("  OK Dashboard system ready")

        self.validation_results['components_status'] = components_status

        print(f"  System initialized with {len(components_status)} components")

        return manager

    async def _run_comprehensive_operation(self, manager):
        """包括的運用テスト"""

        print("  Starting complete system...")

        # システム開始（非同期）
        system_task = asyncio.create_task(manager.start_system())

        print(f"  Running comprehensive test for {self.validation_duration} seconds...")
        print("     - Real-time data streaming")
        print("     - AI prediction generation")
        print("     - Performance monitoring")
        print("     - Alert system")
        print("     - Dashboard serving")

        # 定期的な状況監視
        monitoring_interval = 10  # 10秒間隔
        cycles = self.validation_duration // monitoring_interval

        for cycle in range(cycles):
            await asyncio.sleep(monitoring_interval)

            try:
                # システム状況取得
                system_status = manager.get_system_status()
                predictions = manager.get_latest_predictions()
                market_data = manager.get_market_data_summary()

                # 進捗表示
                elapsed = (cycle + 1) * monitoring_interval
                progress = (elapsed / self.validation_duration) * 100

                print(f"  Progress: {progress:5.1f}% | "
                      f"Predictions: {len(predictions):2d} | "
                      f"Market Data: {len(market_data):2d} symbols | "
                      f"Uptime: {elapsed:3d}s")

                # 統計更新
                if predictions:
                    self.validation_results['predictions_generated'] = len(predictions)

                # 高信頼度予測の表示
                for symbol, prediction in predictions.items():
                    if prediction.action_confidence > 0.7:
                        print(f"     HIGH CONF {symbol}: {prediction.final_action} "
                              f"({prediction.action_confidence:.1%} confidence)")

            except Exception as e:
                logger.warning(f"Monitoring cycle {cycle} error: {e}")
                self.validation_results['errors_encountered'].append(f"Cycle {cycle}: {str(e)}")

        print("  OK Comprehensive operation test completed")

        # システム停止
        await manager.stop_system()
        system_task.cancel()

        try:
            await system_task
        except asyncio.CancelledError:
            pass

    async def _validate_system_performance(self, manager):
        """システムパフォーマンス検証"""

        print("  Analyzing system performance...")

        try:
            # 最終統計取得
            final_status = manager.get_system_status()
            final_predictions = manager.get_latest_predictions()

            # パフォーマンス指標計算
            if final_status:
                uptime = final_status.get('uptime_seconds', 0)
                stats = final_status.get('statistics', {})

                performance_metrics = {
                    'system_uptime_seconds': uptime,
                    'total_predictions': stats.get('total_predictions', 0),
                    'predictions_per_minute': (stats.get('total_predictions', 0) * 60 / max(uptime, 1)),
                    'total_alerts': stats.get('total_alerts', 0),
                    'system_errors': stats.get('system_errors', 0),
                    'active_symbols': len(final_predictions),
                    'error_rate': stats.get('system_errors', 0) / max(stats.get('total_predictions', 1), 1)
                }

                self.validation_results['performance_metrics'] = performance_metrics

                print(f"  OK System uptime: {uptime:.1f} seconds")
                print(f"  OK Total predictions: {performance_metrics['total_predictions']}")
                print(f"  OK Predictions/minute: {performance_metrics['predictions_per_minute']:.1f}")
                print(f"  OK Total alerts: {performance_metrics['total_alerts']}")
                print(f"  OK Error rate: {performance_metrics['error_rate']:.2%}")

                # システム安定性スコア計算
                stability_score = self._calculate_stability_score(performance_metrics)
                self.validation_results['system_stability'] = stability_score

                print(f"  System stability score: {stability_score:.1%}")

        except Exception as e:
            logger.error(f"Performance validation error: {e}")
            self.validation_results['errors_encountered'].append(f"Performance validation: {str(e)}")

    def _calculate_stability_score(self, metrics: Dict[str, Any]) -> float:
        """システム安定性スコア計算"""

        score = 1.0

        # エラー率ペナルティ
        error_rate = metrics.get('error_rate', 0)
        if error_rate > 0.1:  # 10%以上
            score *= 0.5
        elif error_rate > 0.05:  # 5%以上
            score *= 0.8

        # 予測頻度ボーナス
        pred_per_min = metrics.get('predictions_per_minute', 0)
        if pred_per_min >= 10:
            score *= 1.2
        elif pred_per_min >= 5:
            score *= 1.1

        # 稼働時間ボーナス
        uptime = metrics.get('system_uptime_seconds', 0)
        if uptime >= self.validation_duration * 0.9:  # 90%以上稼働
            score *= 1.1

        return min(score, 1.0)

    async def _shutdown_and_analyze(self, manager):
        """システム停止・分析"""

        print("  Performing final system shutdown...")

        try:
            # 最終状況記録
            if hasattr(manager, 'get_system_status'):
                final_status = manager.get_system_status()
                self.validation_results['components_status'].update({
                    'final_state': 'shutdown_completed'
                })

        except Exception as e:
            logger.warning(f"Final shutdown analysis error: {e}")

        print("  OK System shutdown completed")

    def _generate_final_report(self):
        """最終レポート生成"""

        print("\n" + "=" * 60)
        print("NEXT-GEN AI TRADING ENGINE - VALIDATION REPORT")
        print("=" * 60)

        # 基本情報
        start_time = self.validation_results['start_time']
        end_time = self.validation_results['end_time']
        duration = self.validation_results['total_duration_seconds']

        if start_time and end_time:
            print(f"Test Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%H:%M:%S')}")
            print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

        # コンポーネント状況
        print(f"\nCOMPONENTS STATUS:")
        components = self.validation_results['components_status']
        for component, status in components.items():
            status_mark = "OK" if status in ['initialized', 'active'] else "WARN"
            print(f"  [{status_mark}] {component}: {status}")

        # パフォーマンス指標
        print(f"\nPERFORMANCE METRICS:")
        metrics = self.validation_results['performance_metrics']
        if metrics:
            print(f"  Predictions Generated: {metrics.get('total_predictions', 0)}")
            print(f"  Predictions/Minute: {metrics.get('predictions_per_minute', 0):.1f}")
            print(f"  Alerts Sent: {metrics.get('total_alerts', 0)}")
            print(f"  Error Rate: {metrics.get('error_rate', 0):.2%}")
            print(f"  Active Symbols: {metrics.get('active_symbols', 0)}")

        # システム安定性
        stability = self.validation_results['system_stability']
        print(f"\nSYSTEM STABILITY: {stability:.1%}")

        # 最終スコア計算
        final_score = self._calculate_final_score()
        self.validation_results['final_score'] = final_score

        print(f"\nOVERALL SCORE: {final_score:.1%}")

        # 判定
        if final_score >= 0.9:
            grade = "EXCELLENT - Production Ready"
            mark = "EXCELLENT"
        elif final_score >= 0.8:
            grade = "GOOD - Minor Tuning Needed"
            mark = "GOOD"
        elif final_score >= 0.7:
            grade = "ACCEPTABLE - Some Issues"
            mark = "WARN"
        else:
            grade = "NEEDS IMPROVEMENT"
            mark = "FAIL"

        print(f"\nFINAL GRADE: [{mark}] {grade}")

        # エラーサマリー
        errors = self.validation_results['errors_encountered']
        if errors:
            print(f"\nISSUES ENCOUNTERED ({len(errors)}):")
            for i, error in enumerate(errors[:5], 1):  # 最大5件表示
                print(f"  {i}. {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more issues")
        else:
            print(f"\nNO CRITICAL ISSUES DETECTED")

        # 推奨事項
        print(f"\nRECOMMENDATIONS:")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            print(f"  - {rec}")

        print("\n" + "=" * 60)

        # 結果をJSONファイルに保存
        self._save_validation_report()

    def _calculate_final_score(self) -> float:
        """最終スコア計算"""

        score = 1.0

        # コンポーネント成功率
        components = self.validation_results['components_status']
        active_components = len([c for c in components.values() if c in ['initialized', 'active']])
        total_components = max(len(components), 1)
        component_score = active_components / total_components
        score *= component_score

        # システム安定性
        stability = self.validation_results['system_stability']
        score *= stability

        # パフォーマンス指標
        metrics = self.validation_results.get('performance_metrics', {})
        if metrics.get('total_predictions', 0) > 0:
            score *= 1.1  # 予測生成ボーナス

        # エラーペナルティ
        error_count = len(self.validation_results['errors_encountered'])
        if error_count > 5:
            score *= 0.7
        elif error_count > 2:
            score *= 0.9

        return min(score, 1.0)

    def _generate_recommendations(self) -> List[str]:
        """推奨事項生成"""

        recommendations = []

        metrics = self.validation_results.get('performance_metrics', {})
        errors = self.validation_results['errors_encountered']

        if metrics.get('error_rate', 0) > 0.05:
            recommendations.append("エラーハンドリングの改善を推奨")

        if metrics.get('predictions_per_minute', 0) < 5:
            recommendations.append("予測頻度の最適化を検討")

        if len(errors) > 0:
            recommendations.append("ログから特定されたエラーの修正")

        if not recommendations:
            recommendations.append("システムは良好に動作しています")
            recommendations.append("本番環境での運用が可能です")

        return recommendations

    def _save_validation_report(self):
        """検証レポート保存"""

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_validation_report_{timestamp}.json"

            # JSON形式で保存
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, default=str, ensure_ascii=False)

            print(f"\nValidation report saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

async def main():
    """メイン実行"""

    # ログレベル設定（重要な情報のみ表示）
    logging.getLogger().setLevel(logging.WARNING)

    try:
        # 完全システム検証実行
        validator = FullSystemValidator()
        results = await validator.run_comprehensive_validation()

        # 成功判定
        final_score = results.get('final_score', 0.0)

        if final_score >= 0.8:
            print("\nVALIDATION SUCCESSFUL!")
            print("System is ready for production deployment!")
            return 0
        else:
            print("\nVALIDATION COMPLETED WITH ISSUES")
            print("Please review the report and address identified issues.")
            return 1

    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        return 2
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    # 完全統合システム検証実行
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
