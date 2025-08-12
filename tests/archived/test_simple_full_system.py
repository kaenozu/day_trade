#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - 簡易完全統合システム検証
リアルタイムシステム全体の動作確認（シンプル版）
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from src.day_trade.realtime.integration_manager import (
    IntegrationConfig,
    create_integration_manager,
)

# プロジェクト内インポート
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SimpleSystemValidator:
    """簡易システム検証器"""

    def __init__(self):
        self.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        self.validation_duration = 30  # 30秒間の検証
        self.dashboard_port = 8080

        # 検証結果記録
        self.results = {
            "start_time": None,
            "end_time": None,
            "system_active": False,
            "predictions_count": 0,
            "components_initialized": 0,
            "errors": [],
        }

        logger.info("Simple System Validator initialized")

    async def run_validation(self):
        """システム検証実行"""

        print("=" * 60)
        print("Next-Gen AI Trading Engine - System Validation")
        print("=" * 60)
        print(f"Target Symbols: {self.test_symbols}")
        print(f"Validation Duration: {self.validation_duration} seconds")
        print(f"Dashboard URL: http://localhost:{self.dashboard_port}")
        print("=" * 60)

        self.results["start_time"] = datetime.now()

        try:
            # Phase 1: システム初期化
            print("\nPhase 1: System Initialization")
            manager = await self._initialize_system()

            # Phase 2: 運用テスト
            print("\nPhase 2: Operation Test")
            await self._run_operation_test(manager)

            # Phase 3: 結果分析
            print("\nPhase 3: Results Analysis")
            await self._analyze_results(manager)

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.results["errors"].append(str(e))
            import traceback

            traceback.print_exc()

        finally:
            self.results["end_time"] = datetime.now()

        # レポート生成
        self._generate_report()

        return self.results

    async def _initialize_system(self):
        """システム初期化"""

        print("  Creating integration manager...")

        # 統合管理システム設定
        config = IntegrationConfig(
            symbols=self.test_symbols,
            enable_streaming=True,
            enable_prediction=True,
            enable_monitoring=True,
            enable_alerts=True,
            enable_dashboard=False,  # 簡易版ではダッシュボード無効
            update_interval=3.0,  # 3秒間隔
            detailed_logging=False,
        )

        manager = create_integration_manager(
            symbols=self.test_symbols, dashboard_port=self.dashboard_port
        )

        print("  OK Integration manager created")

        # システム初期化
        print("  Initializing components...")
        await manager.initialize_system()

        # コンポーネント確認
        components_count = 0

        if manager.stream_manager:
            components_count += 1
            print("  OK Streaming system initialized")

        if manager.prediction_engine:
            components_count += 1
            print("  OK AI prediction engine initialized")

        if manager.performance_monitor:
            components_count += 1
            print("  OK Performance monitoring initialized")

        if manager.alert_manager:
            components_count += 1
            print("  OK Alert system initialized")

        self.results["components_initialized"] = components_count
        print(f"  RESULT: {components_count}/4 components initialized")

        return manager

    async def _run_operation_test(self, manager):
        """運用テスト"""

        print(f"  Starting system operation for {self.validation_duration} seconds...")

        # システム開始（非同期）
        system_task = asyncio.create_task(manager.start_system())

        # 定期監視
        monitoring_cycles = self.validation_duration // 5  # 5秒間隔

        for cycle in range(monitoring_cycles):
            await asyncio.sleep(5)

            try:
                # システム状況取得
                system_status = manager.get_system_status()
                predictions = manager.get_latest_predictions()

                if system_status and system_status.get("is_running"):
                    self.results["system_active"] = True
                    print(
                        f"  Cycle {cycle + 1}: System active, {len(predictions)} predictions"
                    )

                if predictions:
                    self.results["predictions_count"] = len(predictions)

                    # 予測結果表示
                    for symbol, prediction in predictions.items():
                        if prediction.action_confidence > 0.6:
                            print(
                                f"    {symbol}: {prediction.final_action} "
                                f"({prediction.action_confidence:.1%} confidence)"
                            )

            except Exception as e:
                logger.warning(f"Monitoring cycle {cycle} error: {e}")
                self.results["errors"].append(f"Cycle {cycle}: {str(e)}")

        print("  Operation test completed")

        # システム停止
        await manager.stop_system()
        system_task.cancel()

        try:
            await system_task
        except asyncio.CancelledError:
            pass

    async def _analyze_results(self, manager):
        """結果分析"""

        print("  Analyzing system performance...")

        try:
            # 最終統計取得
            final_status = manager.get_system_status()
            final_predictions = manager.get_latest_predictions()

            if final_status:
                uptime = final_status.get("uptime_seconds", 0)
                stats = final_status.get("statistics", {})

                print(f"  System uptime: {uptime:.1f} seconds")
                print(f"  Total predictions: {stats.get('total_predictions', 0)}")
                print(f"  Total alerts: {stats.get('total_alerts', 0)}")
                print(f"  System errors: {stats.get('system_errors', 0)}")
                print(f"  Active predictions: {len(final_predictions)}")

                # 結果記録
                self.results["predictions_count"] = max(
                    self.results["predictions_count"], stats.get("total_predictions", 0)
                )

        except Exception as e:
            logger.error(f"Results analysis error: {e}")
            self.results["errors"].append(f"Analysis: {str(e)}")

    def _generate_report(self):
        """レポート生成"""

        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        # 基本情報
        start_time = self.results["start_time"]
        end_time = self.results["end_time"]

        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
            print(f"Test Duration: {duration:.1f} seconds")

        # 結果サマリー
        components = self.results["components_initialized"]
        system_active = self.results["system_active"]
        predictions = self.results["predictions_count"]
        errors = len(self.results["errors"])

        print("\nRESULTS SUMMARY:")
        print(f"  Components Initialized: {components}/4")
        print(f"  System Active: {'Yes' if system_active else 'No'}")
        print(f"  Predictions Generated: {predictions}")
        print(f"  Errors Encountered: {errors}")

        # スコア計算
        score = 0.0

        if components >= 3:
            score += 0.3
        if system_active:
            score += 0.3
        if predictions > 0:
            score += 0.3
        if errors == 0:
            score += 0.1

        print(f"\nOVERALL SCORE: {score:.1%}")

        # 判定
        if score >= 0.8:
            grade = "EXCELLENT - System Ready"
        elif score >= 0.6:
            grade = "GOOD - Minor Issues"
        elif score >= 0.4:
            grade = "ACCEPTABLE - Needs Attention"
        else:
            grade = "POOR - Major Issues"

        print(f"GRADE: {grade}")

        # エラー詳細
        if errors > 0:
            print(f"\nISSUES DETECTED ({errors}):")
            for i, error in enumerate(self.results["errors"][:3], 1):
                print(f"  {i}. {error}")
            if errors > 3:
                print(f"  ... and {errors - 3} more issues")

        # 推奨事項
        print("\nRECOMMENDations:")
        if score >= 0.8:
            print("  - System is performing well")
            print("  - Ready for extended testing")
        elif score >= 0.6:
            print("  - Review and fix minor issues")
            print("  - Monitor system stability")
        else:
            print("  - Address critical issues before deployment")
            print("  - Check component initialization")

        print("=" * 60)

        # 結果保存
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_validation_{timestamp}.json"

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)

            print(f"\nReport saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")


async def main():
    """メイン実行"""

    # ログレベル設定
    logging.getLogger().setLevel(logging.WARNING)

    try:
        # システム検証実行
        validator = SimpleSystemValidator()
        results = await validator.run_validation()

        # 成功判定
        components = results["components_initialized"]
        system_active = results["system_active"]
        predictions = results["predictions_count"]
        errors = len(results["errors"])

        # スコア計算
        score = 0.0
        if components >= 3:
            score += 0.3
        if system_active:
            score += 0.3
        if predictions > 0:
            score += 0.3
        if errors == 0:
            score += 0.1

        if score >= 0.7:
            print("\nVALIDATION SUCCESSFUL!")
            print("System is ready for production!")
            return 0
        else:
            print("\nVALIDATION COMPLETED WITH ISSUES")
            print("Please review and address the issues.")
            return 1

    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return 2
    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback

        traceback.print_exc()
        return 3


if __name__ == "__main__":
    # システム検証実行
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
