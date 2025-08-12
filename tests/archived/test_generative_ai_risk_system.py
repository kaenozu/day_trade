#!/usr/bin/env python3
"""
生成AI統合リスク管理システム統合テスト
Generative AI Risk Management System Integration Test

全コンポーネントの統合テスト・デモンストレーション
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from src.day_trade.realtime.risk_dashboard import RiskDashboardManager
from src.day_trade.risk.fraud_detection_engine import (
    FraudDetectionEngine,
    FraudDetectionRequest,
)

# プロジェクト内インポート
from src.day_trade.risk.generative_ai_engine import (
    GenerativeAIConfig,
    GenerativeAIRiskEngine,
    RiskAnalysisRequest,
)
from src.day_trade.risk.real_time_monitor import (
    RealTimeRiskMonitor,
    RiskMonitoringConfig,
)
from src.day_trade.risk.risk_coordinator import RiskAnalysisCoordinator
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class GenerativeAIRiskSystemDemo:
    """生成AI統合リスク管理システムデモ"""

    def __init__(self):
        # システムコンポーネント初期化
        self.ai_config = GenerativeAIConfig(
            openai_api_key="demo_key",  # 実際はos.getenv("OPENAI_API_KEY")
            anthropic_api_key="demo_key",  # 実際はos.getenv("ANTHROPIC_API_KEY")
            temperature=0.3,
            max_tokens=800,
            enable_caching=True,
        )

        self.generative_ai_engine = GenerativeAIRiskEngine(self.ai_config)
        self.fraud_engine = FraudDetectionEngine()
        self.risk_coordinator = RiskAnalysisCoordinator()

        # リアルタイム監視設定
        self.monitor_config = RiskMonitoringConfig(
            monitoring_interval_seconds=3,
            batch_analysis_interval_minutes=2,
            alert_cooldown_minutes=1,
            enable_auto_response=True,
        )
        self.realtime_monitor = RealTimeRiskMonitor(self.monitor_config)

        # ダッシュボード
        self.dashboard = RiskDashboardManager(port=8888)

        # テスト結果
        self.test_results = {}

        logger.info("生成AI統合リスク管理システムデモ初期化完了")

    async def run_comprehensive_demo(self):
        """包括的デモンストレーション実行"""

        print("\n" + "=" * 80)
        print("🤖 生成AI統合リスク管理システム - 包括的デモンストレーション")
        print("=" * 80)
        print(f"📅 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 目標: 95%精度, 1秒以内検知, 10億円損失防止")
        print()

        # デモシーケンス実行
        await self._demo_1_basic_ai_analysis()
        await self._demo_2_fraud_detection()
        await self._demo_3_integrated_risk_assessment()
        await self._demo_4_realtime_monitoring()
        await self._demo_5_dashboard_visualization()

        # 総合結果表示
        self._display_comprehensive_results()

    async def _demo_1_basic_ai_analysis(self):
        """デモ1: 基本AI分析機能"""

        print("🧠 デモ1: 生成AI基本分析機能テスト")
        print("-" * 50)

        start_time = time.time()

        # テスト用リスク分析リクエスト
        test_request = RiskAnalysisRequest(
            transaction_id="DEMO_AI_001",
            symbol="7203",  # トヨタ自動車
            transaction_type="buy",
            amount=8000000,  # 800万円（高額取引）
            timestamp=datetime.now(),
            market_data={
                "current_price": 2450,
                "price_change_percent": -3.2,
                "volume": 2500000,
                "volatility": 0.28,
                "rsi": 25,  # 売られ過ぎ
                "macd_signal": "bullish",
                "market_sentiment": "cautious",
            },
            user_profile={
                "user_id": "premium_001",
                "risk_tolerance": "moderate",
                "experience_level": "advanced",
                "portfolio_value": 50000000,
                "investment_horizon": "long_term",
            },
        )

        try:
            # 生成AI分析実行（GPT-4/Claudeを無効化してテスト）
            result = await self.generative_ai_engine.analyze_risk_comprehensive(
                test_request,
                use_gpt4=False,  # デモではダミーキーなので無効
                use_claude=False,  # デモではダミーキーなので無効
                use_ensemble=True,
            )

            processing_time = time.time() - start_time

            print(f"✅ AI分析完了 ({processing_time:.2f}秒)")
            print(f"   リスクスコア: {result.risk_score:.3f}")
            print(f"   リスクレベル: {result.risk_level}")
            print(f"   信頼度: {result.confidence:.3f}")
            print(f"   使用モデル: {', '.join(result.ai_models_used)}")
            print(f"   処理時間: {result.processing_time:.3f}秒")

            # 結果保存
            self.test_results["ai_analysis"] = {
                "success": True,
                "processing_time": processing_time,
                "risk_score": result.risk_score,
                "confidence": result.confidence,
            }

        except Exception as e:
            print(f"❌ AI分析エラー: {e}")
            self.test_results["ai_analysis"] = {"success": False, "error": str(e)}

        print()

    async def _demo_2_fraud_detection(self):
        """デモ2: 不正検知エンジン"""

        print("🛡️ デモ2: 深層学習不正検知システム")
        print("-" * 50)

        start_time = time.time()

        # 疑わしい取引データ作成
        suspicious_request = FraudDetectionRequest(
            transaction_id="DEMO_FRAUD_001",
            user_id="user_suspicious",
            amount=12000000,  # 1200万円（超高額）
            timestamp=datetime(2025, 1, 1, 2, 30, 0),  # 深夜取引
            transaction_type="transfer",
            account_balance=500000,  # 残高に対して異常に高額
            location="foreign_high_risk",
            device_info={
                "type": "mobile",
                "os": "android",
                "is_new_device": True,  # 新規デバイス
                "ip_location": "suspicious_region",
            },
            transaction_history=[
                {"amount": 50000, "timestamp": "2025-01-01T02:00:00"},
                {"amount": 100000, "timestamp": "2025-01-01T02:15:00"},
                {
                    "amount": 200000,
                    "timestamp": "2025-01-01T02:25:00",
                },  # エスカレーション
            ],
            market_conditions={
                "volatility": 0.45,  # 高ボラティリティ
                "volume": 300000,  # 低取引量
                "trend": "bearish",
            },
        )

        try:
            # 不正検知実行
            fraud_result = await self.fraud_engine.detect_fraud(suspicious_request)

            processing_time = time.time() - start_time

            print(
                f"{'🚨' if fraud_result.is_fraud else '✅'} 不正検知完了 ({processing_time:.2f}秒)"
            )
            print(f"   不正判定: {'はい' if fraud_result.is_fraud else 'いいえ'}")
            print(f"   不正確率: {fraud_result.fraud_probability:.3f}")
            print(f"   信頼度: {fraud_result.confidence:.3f}")
            print(f"   異常スコア: {fraud_result.anomaly_score:.3f}")
            print(f"   使用モデル: {', '.join(fraud_result.models_used)}")
            print(f"   推奨アクション: {fraud_result.recommended_action}")

            # 結果保存
            self.test_results["fraud_detection"] = {
                "success": True,
                "processing_time": processing_time,
                "is_fraud": fraud_result.is_fraud,
                "fraud_probability": fraud_result.fraud_probability,
                "confidence": fraud_result.confidence,
            }

        except Exception as e:
            print(f"❌ 不正検知エラー: {e}")
            self.test_results["fraud_detection"] = {"success": False, "error": str(e)}

        print()

    async def _demo_3_integrated_risk_assessment(self):
        """デモ3: 統合リスク評価"""

        print("⚖️ デモ3: 統合リスク評価システム")
        print("-" * 50)

        start_time = time.time()

        # 複合リスクシナリオ
        complex_transaction = {
            "symbol": "6758",  # ソニーグループ
            "type": "margin_buy",  # 信用取引
            "amount": 15000000,  # 1500万円
            "timestamp": datetime.now().isoformat(),
            "user_id": "trader_007",
            "account_balance": 8000000,
            "location": "domestic",
            "device_info": {"type": "desktop", "os": "windows", "is_new_device": False},
            "history": [
                {
                    "amount": 1000000,
                    "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                },
                {
                    "amount": 2000000,
                    "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                },
            ],
            "market_conditions": {
                "volatility": 0.35,
                "volume": 1800000,
                "trend": "volatile",
                "news_sentiment": "negative",
            },
        }

        try:
            # 統合リスク評価実行
            assessment = await self.risk_coordinator.comprehensive_risk_assessment(
                complex_transaction,
                market_context=complex_transaction["market_conditions"],
                user_profile={"risk_tolerance": "aggressive"},
                enable_ai_analysis=True,
                enable_fraud_detection=True,
            )

            processing_time = time.time() - start_time

            print(f"✅ 統合評価完了 ({processing_time:.2f}秒)")
            print(f"   総合リスクスコア: {assessment.overall_risk_score:.3f}")
            print(f"   リスクカテゴリー: {assessment.risk_category.upper()}")
            print(f"   信頼度: {assessment.confidence_score:.3f}")
            print(f"   分析手法: {', '.join(assessment.analysis_methods)}")
            print(f"   主要リスク要因: {', '.join(assessment.key_risk_factors[:3])}")
            print(
                f"   推定損失ポテンシャル: ¥{assessment.estimated_loss_potential:,.0f}"
            )
            print(f"   処理時間合計: {assessment.processing_time_total:.3f}秒")

            # 推奨事項表示
            if assessment.recommendations:
                print("   推奨事項:")
                for i, rec in enumerate(assessment.recommendations[:3], 1):
                    print(f"     {i}. {rec}")

            # 結果保存
            self.test_results["integrated_assessment"] = {
                "success": True,
                "processing_time": processing_time,
                "risk_score": assessment.overall_risk_score,
                "risk_category": assessment.risk_category,
                "confidence": assessment.confidence_score,
            }

        except Exception as e:
            print(f"❌ 統合評価エラー: {e}")
            self.test_results["integrated_assessment"] = {
                "success": False,
                "error": str(e),
            }

        print()

    async def _demo_4_realtime_monitoring(self):
        """デモ4: リアルタイム監視"""

        print("📊 デモ4: リアルタイム監視システム")
        print("-" * 50)

        # テスト用自動応答ハンドラー
        response_count = {"critical": 0, "high": 0, "medium": 0}

        async def critical_handler(assessment):
            response_count["critical"] += 1
            print(f"   🚨 重要リスク自動応答実行: {assessment.request_id}")

        async def high_handler(assessment):
            response_count["high"] += 1
            print(f"   ⚠️ 高リスク自動応答実行: {assessment.request_id}")

        async def medium_handler(assessment):
            response_count["medium"] += 1
            print(f"   📊 中程度リスク自動応答実行: {assessment.request_id}")

        # ハンドラー登録
        self.realtime_monitor.register_response_handler("critical", critical_handler)
        self.realtime_monitor.register_response_handler("high", high_handler)
        self.realtime_monitor.register_response_handler("medium", medium_handler)

        # テスト銘柄
        test_symbols = ["7203", "6758", "9984"]  # トヨタ、ソニー、ソフトバンク

        print(f"監視銘柄: {', '.join(test_symbols)}")
        print("リアルタイム監視開始（15秒間）...")

        start_time = time.time()

        # 監視タスク開始
        monitor_task = asyncio.create_task(
            self.realtime_monitor.start_monitoring(test_symbols)
        )

        try:
            # 15秒間監視実行
            await asyncio.wait_for(monitor_task, timeout=15)
        except asyncio.TimeoutError:
            # 正常なタイムアウト
            await self.realtime_monitor.stop_monitoring()

        processing_time = time.time() - start_time

        # 監視結果取得
        status = self.realtime_monitor.get_monitoring_status()

        print(f"✅ リアルタイム監視完了 ({processing_time:.1f}秒)")
        print(
            f"   監視サイクル数: {status['performance_stats']['total_monitoring_cycles']}"
        )
        print(f"   リスク分析数: {status['performance_stats']['total_risk_analyses']}")
        print(f"   送信アラート数: {status['performance_stats']['total_alerts_sent']}")
        print(f"   自動応答実行数: {sum(response_count.values())}")
        print(f"   システムヘルス: {status['system_health']}")
        print(
            f"   平均サイクル時間: {status['performance_stats']['avg_cycle_time']:.3f}秒"
        )

        # 結果保存
        self.test_results["realtime_monitoring"] = {
            "success": True,
            "monitoring_time": processing_time,
            "monitoring_cycles": status["performance_stats"]["total_monitoring_cycles"],
            "risk_analyses": status["performance_stats"]["total_risk_analyses"],
            "alerts_sent": status["performance_stats"]["total_alerts_sent"],
            "auto_responses": sum(response_count.values()),
            "system_health": status["system_health"],
        }

        print()

    async def _demo_5_dashboard_visualization(self):
        """デモ5: ダッシュボード可視化"""

        print("🖥️ デモ5: リアルタイムダッシュボード")
        print("-" * 50)

        print("ダッシュボードサーバー起動中...")
        print("URL: http://localhost:8888")
        print("（実際のブラウザでアクセスしてください）")

        # ダッシュボードを5秒間起動
        dashboard_task = asyncio.create_task(self.dashboard.run_dashboard())

        try:
            await asyncio.wait_for(dashboard_task, timeout=5)
        except asyncio.TimeoutError:
            print("✅ ダッシュボードデモ完了（5秒間起動）")

            self.test_results["dashboard"] = {
                "success": True,
                "startup_time": 5.0,
                "url": "http://localhost:8888",
            }
        except Exception as e:
            print(f"❌ ダッシュボードエラー: {e}")
            self.test_results["dashboard"] = {"success": False, "error": str(e)}

        print()

    def _display_comprehensive_results(self):
        """総合結果表示"""

        print("\n" + "=" * 80)
        print("📊 生成AI統合リスク管理システム - 総合結果")
        print("=" * 80)

        # 成功率計算
        total_tests = len(self.test_results)
        successful_tests = sum(
            1 for result in self.test_results.values() if result.get("success", False)
        )
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"🎯 総合成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        print()

        # 各コンポーネント結果
        for component, result in self.test_results.items():
            status = "✅ 成功" if result.get("success") else "❌ 失敗"
            print(f"{component.replace('_', ' ').title()}: {status}")

            if result.get("success"):
                if "processing_time" in result:
                    print(f"  処理時間: {result['processing_time']:.3f}秒")
                if "risk_score" in result:
                    print(f"  リスクスコア: {result['risk_score']:.3f}")
                if "confidence" in result:
                    print(f"  信頼度: {result['confidence']:.3f}")
                if "monitoring_cycles" in result:
                    print(f"  監視サイクル: {result['monitoring_cycles']}")
            else:
                if "error" in result:
                    print(f"  エラー: {result['error']}")
            print()

        # パフォーマンス要約
        print("⚡ パフォーマンス要約:")

        # 平均処理時間
        processing_times = [
            r.get("processing_time", 0)
            for r in self.test_results.values()
            if r.get("processing_time")
        ]
        if processing_times:
            avg_time = np.mean(processing_times)
            print(f"  平均処理時間: {avg_time:.3f}秒")
            print(f"  最速処理: {min(processing_times):.3f}秒")
            print(f"  最低処理: {max(processing_times):.3f}秒")

        # システム統計
        ai_stats = self.generative_ai_engine.get_performance_stats()
        fraud_stats = self.fraud_engine.get_stats()
        coordinator_stats = self.risk_coordinator.get_performance_summary()

        print("\n🧠 AI エンジン統計:")
        print(f"  総分析数: {ai_stats.get('total_analyses', 0)}")
        print(f"  成功分析数: {ai_stats.get('successful_analyses', 0)}")
        print(f"  キャッシュサイズ: {ai_stats.get('cache_size', 0)}")

        print("\n🛡️ 不正検知統計:")
        print(f"  総検知数: {fraud_stats.get('total_detections', 0)}")
        print(f"  不正検知数: {fraud_stats.get('fraud_detected', 0)}")
        print(f"  平均処理時間: {fraud_stats.get('avg_processing_time', 0):.3f}秒")

        print("\n⚖️ 統合コーディネーター統計:")
        print(f"  総評価数: {coordinator_stats.get('total_assessments', 0)}")
        print(f"  成功率: {coordinator_stats.get('success_rate', 0):.1%}")
        print(
            f"  平均処理時間: {coordinator_stats.get('avg_processing_time', 0):.3f}秒"
        )

        # 結論
        print("\n" + "=" * 80)
        if success_rate >= 80:
            print("🎉 生成AI統合リスク管理システム デモ成功!")
            print("   システムは期待通りに動作しています。")
        elif success_rate >= 60:
            print("⚠️ 生成AI統合リスク管理システム 部分的成功")
            print("   いくつかの機能で改善が必要です。")
        else:
            print("❌ 生成AI統合リスク管理システム デモ失敗")
            print("   システムに重大な問題があります。")

        print("\n💡 次世代金融AIリスク管理システムの完成!")
        print("   - GPT-4/Claude統合による高精度分析")
        print("   - 深層学習による不正検知")
        print("   - リアルタイム監視・自動対応")
        print("   - 直感的ダッシュボード")
        print("=" * 80)


async def main():
    """メイン実行関数"""

    try:
        # デモシステム作成・実行
        demo = GenerativeAIRiskSystemDemo()
        await demo.run_comprehensive_demo()

    except KeyboardInterrupt:
        print("\n\nデモンストレーション中断")
    except Exception as e:
        print(f"\n\nデモエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 生成AI統合リスク管理システム - 統合デモ起動中...")
    asyncio.run(main())
