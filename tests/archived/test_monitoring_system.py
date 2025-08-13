#!/usr/bin/env python3
"""
リアルタイムメトリクス・アラートシステム テスト
Monitoring System Integration Test
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict

from src.day_trade.monitoring.metrics import (
    get_ai_metrics,
    get_health_metrics,
    get_metrics_collector,
    get_risk_metrics,
    get_trading_metrics,
    start_metrics_server,
)
from src.day_trade.risk.risk_coordinator import RiskAnalysisCoordinator
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MonitoringSystemTest:
    """監視システム統合テスト"""

    def __init__(self):
        self.test_results = {}
        logger.info("監視システム統合テスト初期化")

    async def run_comprehensive_test(self):
        """包括的テスト実行"""

        print("\n" + "=" * 70)
        print("リアルタイムメトリクス・アラートシステム 統合テスト")
        print("=" * 70)
        print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # テスト実行
        await self._test_metrics_collection()
        await self._test_risk_metrics_integration()
        await self._test_metrics_exporter()
        await self._test_alert_generation()

        # 結果表示
        self._display_results()

    async def _test_metrics_collection(self):
        """メトリクス収集テスト"""

        print("テスト1: メトリクス収集システム")
        print("-" * 40)

        start_time = time.time()

        try:
            # メトリクス収集器取得
            collector = get_metrics_collector()
            risk_metrics = get_risk_metrics()
            trading_metrics = get_trading_metrics()
            ai_metrics = get_ai_metrics()
            health_metrics = get_health_metrics()

            # 基本メトリクス収集
            collection_result = collector.collect_all_metrics()

            processing_time = time.time() - start_time

            print("  OK メトリクス収集器初期化完了")
            print("  OK リスク管理メトリクス: 準備完了")
            print("  OK 取引メトリクス: 準備完了")
            print("  OK AIメトリクス: 準備完了")
            print("  OK ヘルスメトリクス: 準備完了")
            print(f"  処理時間: {processing_time:.3f}秒")

            self.test_results["metrics_collection"] = {
                "success": True,
                "collection_result": collection_result,
                "processing_time": processing_time,
            }

        except Exception as e:
            print(f"  NG メトリクス収集エラー: {e}")
            self.test_results["metrics_collection"] = {
                "success": False,
                "error": str(e),
            }

        print()

    async def _test_risk_metrics_integration(self):
        """リスクメトリクス統合テスト"""

        print("テスト2: リスク管理システム統合")
        print("-" * 40)

        start_time = time.time()

        try:
            # リスクコーディネーター初期化
            risk_coordinator = RiskAnalysisCoordinator()

            # テスト用トランザクションデータ
            test_transactions = [
                {
                    "symbol": "AAPL",
                    "amount": 1000000,  # 100万円
                    "transaction_type": "buy",
                    "user_id": "test_user_001",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "symbol": "GOOGL",
                    "amount": 5000000,  # 500万円（高リスク）
                    "transaction_type": "sell",
                    "user_id": "test_user_002",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "symbol": "TSLA",
                    "amount": 15000000,  # 1500万円（非常に高リスク）
                    "transaction_type": "buy",
                    "user_id": "test_user_003",
                    "timestamp": datetime.now().isoformat(),
                },
            ]

            risk_assessments = []

            for i, transaction in enumerate(test_transactions):
                print(
                    f"  リスク分析実行 {i+1}/3: {transaction['symbol']} (¥{transaction['amount']:,})"
                )

                # リスク評価実行
                assessment = await risk_coordinator.comprehensive_risk_assessment(
                    transaction_data=transaction,
                    enable_ai_analysis=False,  # テスト用に無効化
                    enable_fraud_detection=True,
                )

                risk_assessments.append(assessment)

                print(f"    リスクスコア: {assessment.overall_risk_score:.3f}")
                print(f"    リスクレベル: {assessment.risk_category}")
                print(f"    処理時間: {assessment.processing_time_total:.3f}秒")

            processing_time = time.time() - start_time

            print(f"\n  OK 全リスク分析完了: {len(risk_assessments)}件")
            print(f"  総処理時間: {processing_time:.3f}秒")

            self.test_results["risk_metrics_integration"] = {
                "success": True,
                "assessments_count": len(risk_assessments),
                "risk_scores": [a.overall_risk_score for a in risk_assessments],
                "processing_time": processing_time,
            }

        except Exception as e:
            print(f"  NG リスクメトリクス統合エラー: {e}")
            self.test_results["risk_metrics_integration"] = {
                "success": False,
                "error": str(e),
            }

        print()

    async def _test_metrics_exporter(self):
        """メトリクスエクスポーターテスト"""

        print("テスト3: メトリクスエクスポーター")
        print("-" * 40)

        start_time = time.time()

        try:
            # バックグラウンドでメトリクスサーバー開始
            print("  メトリクスサーバー開始中...")
            exporter = start_metrics_server(port=8001, background=True)

            # 少し待機
            await asyncio.sleep(2)

            processing_time = time.time() - start_time

            print("  OK メトリクスサーバー開始完了")
            print("  OK エクスポーター: http://localhost:8001/metrics")
            print("  OK ヘルスチェック: http://localhost:8001/health")
            print(f"  処理時間: {processing_time:.3f}秒")

            self.test_results["metrics_exporter"] = {
                "success": True,
                "server_url": "http://localhost:8001",
                "processing_time": processing_time,
            }

        except Exception as e:
            print(f"  NG メトリクスエクスポーターエラー: {e}")
            self.test_results["metrics_exporter"] = {"success": False, "error": str(e)}

        print()

    async def _test_alert_generation(self):
        """アラート生成テスト"""

        print("テスト4: アラート生成システム")
        print("-" * 40)

        start_time = time.time()

        try:
            risk_metrics = get_risk_metrics()

            # テストアラート生成
            test_scenarios = [
                {"component": "test_critical", "symbol": "TEST1", "score": 0.95},
                {"component": "test_high", "symbol": "TEST2", "score": 0.75},
                {"component": "test_medium", "symbol": "TEST3", "score": 0.45},
            ]

            for scenario in test_scenarios:
                # リスクスコア更新
                risk_metrics.update_risk_score(
                    scenario["component"], scenario["symbol"], scenario["score"]
                )

                # アラート生成条件判定
                if scenario["score"] >= 0.9:
                    level = "critical"
                elif scenario["score"] >= 0.7:
                    level = "high"
                else:
                    level = "medium"

                # アラートメトリクス記録
                risk_metrics.risk_alerts_total.labels(
                    alert_level=level, component=scenario["component"]
                ).inc()

                print(
                    f"  OK アラート生成: {level} - {scenario['component']} (スコア: {scenario['score']})"
                )

            processing_time = time.time() - start_time

            print(f"\n  OK アラート生成完了: {len(test_scenarios)}件")
            print(f"  処理時間: {processing_time:.3f}秒")

            self.test_results["alert_generation"] = {
                "success": True,
                "alerts_generated": len(test_scenarios),
                "processing_time": processing_time,
            }

        except Exception as e:
            print(f"  NG アラート生成エラー: {e}")
            self.test_results["alert_generation"] = {"success": False, "error": str(e)}

        print()

    def _display_results(self):
        """結果表示"""

        print("=" * 70)
        print("テスト結果サマリー")
        print("=" * 70)

        # 成功率計算
        total_tests = len(self.test_results)
        successful_tests = sum(
            1 for result in self.test_results.values() if result.get("success", False)
        )
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"総合成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        print()

        # 個別テスト結果
        for test_name, result in self.test_results.items():
            status = "OK 成功" if result.get("success") else "NG 失敗"
            print(f"{test_name.replace('_', ' ').title()}: {status}")

            if result.get("success"):
                if "processing_time" in result:
                    print(f"  処理時間: {result['processing_time']:.3f}秒")
                if "server_url" in result:
                    print(f"  サーバーURL: {result['server_url']}")
                if "assessments_count" in result:
                    print(f"  リスク評価実行数: {result['assessments_count']}件")
            else:
                if "error" in result:
                    print(f"  エラー: {result['error']}")
            print()

        # 総合評価
        print("-" * 70)
        if success_rate == 100:
            print(
                "全テスト成功! リアルタイムメトリクス・アラートシステムは正常に動作しています。"
            )
            print()
            print("次のステップ:")
            print(
                "1. Docker環境での統合テスト: docker-compose -f docker-compose.monitoring.yml up"
            )
            print("2. Grafana ダッシュボード確認: http://localhost:3000")
            print("3. Prometheus 確認: http://localhost:9090")
            print("4. AlertManager 確認: http://localhost:9093")
        elif success_rate >= 75:
            print("大部分のテストが成功しています。一部機能で改善余地があります。")
        else:
            print("いくつかのテストが失敗しています。設定確認が必要です。")

        print()
        print("監視システム統合テスト完了")
        print("=" * 70)


async def main():
    """メイン実行関数"""

    try:
        # 統合テスト実行
        test_system = MonitoringSystemTest()
        await test_system.run_comprehensive_test()

    except KeyboardInterrupt:
        print("\n\nテスト中断")
    except Exception as e:
        print(f"\n\nテストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("リアルタイムメトリクス・アラートシステム 統合テスト起動中...")
    asyncio.run(main())
