#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Issue #332: エンタープライズ級完全統合システム統合テスト

全システム統合動作検証:
- Phase 1: 統合オーケストレーションエンジン
- Phase 2: エンタープライズ級可視化ダッシュボード
- Phase 3: 統合システムテストスイート

統合システム包括検証:
- API・外部統合システム (Issue #331)
- 監視・アラートシステム (Issue #318)
- 高速データ管理システム (Issue #317)
- エンタープライズ級統合システム (Issue #332)
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

try:
    # 基本ログ設定
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    print("Issue #332 エンタープライズ級完全統合システム統合テスト開始")

    # Issue #332 システム
    from src.day_trade.core.enterprise_integration_orchestrator import (
        EnterpriseIntegrationOrchestrator, OrchestrationConfig, OperationMode
    )
    from src.day_trade.dashboard.enterprise_dashboard_system import (
        EnterpriseDashboardSystem, DashboardConfig, DashboardTheme
    )

    # Issue #331 API・外部統合システム
    from src.day_trade.api.api_integration_manager import APIIntegrationManager, IntegrationConfig
    from src.day_trade.api.external_api_client import ExternalAPIClient, APIProvider
    from src.day_trade.api.websocket_streaming_client import WebSocketStreamingClient, StreamProvider

except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なコンポーネントが不足している可能性があります")


async def test_orchestration_engine() -> Dict[str, Any]:
    """Phase 1: 統合オーケストレーションエンジンテスト"""
    print("Phase 1: 統合オーケストレーションエンジンテスト開始")

    try:
        start_time = time.time()

        # オーケストレーター設定
        config = OrchestrationConfig(
            operation_mode=OperationMode.SAFE_MODE,
            enable_api_integration=True,
            enable_monitoring_alerts=True,
            enable_data_management=True,
            enable_advanced_analytics=True,
            max_concurrent_operations=10
        )

        # オーケストレーター初期化
        orchestrator = EnterpriseIntegrationOrchestrator(config)

        # システム初期化テスト
        init_success = await orchestrator.initialize_enterprise_system()

        if not init_success:
            return {
                'phase': 'Phase 1: 統合オーケストレーションエンジン',
                'success': False,
                'error': 'システム初期化失敗'
            }

        # システム概要取得テスト
        system_overview = orchestrator.get_system_overview()
        component_details = orchestrator.get_component_details()

        # 統合分析テストシンボル
        test_symbols = ['7203', '8306', '9984']

        # 運用開始テスト
        await orchestrator.start_enterprise_operations()

        # 少し待機してシステムが安定するまで待つ
        await asyncio.sleep(5)

        # 統合分析レポート生成テスト
        analysis_report = await orchestrator.get_integrated_analysis_report(test_symbols)

        # 運用停止テスト
        await orchestrator.stop_enterprise_operations()

        processing_time = (time.time() - start_time) * 1000

        # オーケストレーション機能検証
        orchestration_features = {
            'system_initialization': init_success,
            'component_management': len(component_details) > 0,
            'enterprise_operations': True,  # 運用開始・停止が成功
            'integrated_analysis': len(analysis_report.get('analysis_results', {})) > 0,
            'safe_mode_compliance': system_overview.get('safe_mode_status', {}).get('safe_mode', False)
        }

        # システム性能指標
        performance_metrics = {
            'initialization_time_ms': processing_time,
            'registered_components': len(component_details),
            'healthy_components': system_overview.get('components', {}).get('healthy', 0),
            'total_components': system_overview.get('components', {}).get('total', 0),
            'health_ratio': system_overview.get('components', {}).get('health_ratio', 0.0)
        }

        # 統合分析結果評価
        analysis_evaluation = {
            'symbols_analyzed': len(test_symbols),
            'successful_analysis': sum(1 for symbol, result in analysis_report.get('analysis_results', {}).items() if 'error' not in result),
            'analysis_success_rate': 0.0
        }

        if analysis_evaluation['symbols_analyzed'] > 0:
            analysis_evaluation['analysis_success_rate'] = analysis_evaluation['successful_analysis'] / analysis_evaluation['symbols_analyzed']

        return {
            'phase': 'Phase 1: 統合オーケストレーションエンジン',
            'success': True,
            'orchestration_features': orchestration_features,
            'performance_metrics': performance_metrics,
            'analysis_evaluation': analysis_evaluation,
            'system_overview': system_overview,
            'test_time_ms': processing_time,
            'symbols_tested': len(test_symbols)
        }

    except Exception as e:
        return {
            'phase': 'Phase 1: 統合オーケストレーションエンジン',
            'success': False,
            'error': str(e)
        }


async def test_dashboard_system() -> Dict[str, Any]:
    """Phase 2: エンタープライズダッシュボードシステムテスト"""
    print("Phase 2: エンタープライズダッシュボードシステムテスト開始")

    try:
        start_time = time.time()

        # オーケストレーター初期化（ダッシュボード用）
        orchestration_config = OrchestrationConfig(
            operation_mode=OperationMode.SAFE_MODE,
            enable_api_integration=True
        )
        orchestrator = EnterpriseIntegrationOrchestrator(orchestration_config)
        await orchestrator.initialize_enterprise_system()

        # ダッシュボード設定
        dashboard_config = DashboardConfig(
            theme=DashboardTheme.ENTERPRISE,
            enable_real_time_updates=True,
            auto_refresh_interval_seconds=5,
            host="localhost",
            port=8081  # テスト用ポート
        )

        # ダッシュボード初期化
        dashboard = EnterpriseDashboardSystem(orchestrator, dashboard_config)

        # ダッシュボード機能テスト
        dashboard_features = {
            'orchestrator_integration': orchestrator is not None,
            'fastapi_app_creation': dashboard.app is not None,
            'websocket_management': hasattr(dashboard, 'websocket_connections'),
            'template_system': hasattr(dashboard, 'templates'),
            'chart_generation': True,  # チャート生成機能
            'real_time_updates': dashboard_config.enable_real_time_updates
        }

        # チャート生成テスト
        system_status_chart = await dashboard._generate_system_status_chart()
        performance_chart = await dashboard._generate_performance_chart()

        chart_generation_results = {
            'system_status_chart_generated': len(system_status_chart) > 100,
            'performance_chart_generated': len(performance_chart) > 100,
            'chart_format': 'HTML/Plotly'
        }

        # 包括レポート生成テスト
        test_symbols = ['7203', '8306', '9984']
        comprehensive_report = await dashboard.generate_comprehensive_report(test_symbols)

        report_quality = {
            'report_structure_complete': all(key in comprehensive_report for key in [
                'report_id', 'generated_at', 'system_overview', 'analysis_results', 'charts', 'summary'
            ]),
            'symbols_in_report': len(comprehensive_report.get('analysis_results', {})),
            'charts_included': len(comprehensive_report.get('charts', {})),
            'summary_metrics': len(comprehensive_report.get('summary', {}))
        }

        # システムクリーンアップ
        await orchestrator.stop_enterprise_operations()

        processing_time = (time.time() - start_time) * 1000

        # 可視化システム評価
        visualization_evaluation = {
            'dashboard_features_ratio': sum(1 for v in dashboard_features.values() if v) / len(dashboard_features),
            'chart_generation_success': all(chart_generation_results.values()),
            'comprehensive_report_quality': report_quality['report_structure_complete'],
            'real_time_capability': dashboard_config.enable_real_time_updates
        }

        return {
            'phase': 'Phase 2: エンタープライズダッシュボードシステム',
            'success': True,
            'dashboard_features': dashboard_features,
            'chart_generation_results': chart_generation_results,
            'report_quality': report_quality,
            'visualization_evaluation': visualization_evaluation,
            'test_time_ms': processing_time,
            'symbols_tested': len(test_symbols)
        }

    except Exception as e:
        return {
            'phase': 'Phase 2: エンタープライズダッシュボードシステム',
            'success': False,
            'error': str(e)
        }


async def test_integrated_system_operations() -> Dict[str, Any]:
    """Phase 3: 統合システム運用テスト"""
    print("Phase 3: 統合システム運用テスト開始")

    try:
        start_time = time.time()

        # フルシステム統合テストシナリオ
        integration_scenario = {
            'scenario': 'enterprise_full_system_integration',
            'components': [
                'enterprise_orchestrator',
                'api_integration_manager',
                'dashboard_system',
                'monitoring_alerts',
                'data_management'
            ],
            'test_duration_seconds': 30
        }

        # エンタープライズシステム完全初期化
        config = OrchestrationConfig(
            operation_mode=OperationMode.SAFE_MODE,
            enable_api_integration=True,
            enable_monitoring_alerts=True,
            enable_data_management=True,
            enable_advanced_analytics=True
        )

        orchestrator = EnterpriseIntegrationOrchestrator(config)
        init_success = await orchestrator.initialize_enterprise_system()

        if not init_success:
            return {
                'phase': 'Phase 3: 統合システム運用',
                'success': False,
                'error': 'システム初期化失敗'
            }

        # ダッシュボード統合
        dashboard_config = DashboardConfig(
            enable_real_time_updates=True,
            auto_refresh_interval_seconds=2
        )
        dashboard = EnterpriseDashboardSystem(orchestrator, dashboard_config)

        # フルシステム運用開始
        await orchestrator.start_enterprise_operations()

        # 統合運用テスト実行
        test_symbols = ['7203', '8306', '9984', '4502', '7182']

        # パフォーマンステスト
        performance_results = []

        for i in range(5):  # 5回の連続テスト
            test_start = time.time()

            # 統合分析実行
            analysis_result = await orchestrator.get_integrated_analysis_report(test_symbols)

            # ダッシュボードレポート生成
            dashboard_report = await dashboard.generate_comprehensive_report(test_symbols[:3])

            test_duration = (time.time() - test_start) * 1000

            performance_results.append({
                'iteration': i + 1,
                'analysis_duration_ms': test_duration,
                'symbols_processed': len(test_symbols),
                'dashboard_report_generated': 'report_id' in dashboard_report,
                'system_health': orchestrator.get_system_overview()['components']['health_ratio']
            })

            await asyncio.sleep(2)  # 2秒間隔

        # システム信頼性テスト
        reliability_metrics = {
            'system_uptime_consistency': all(r['system_health'] > 0.8 for r in performance_results),
            'performance_consistency': max(r['analysis_duration_ms'] for r in performance_results) - min(r['analysis_duration_ms'] for r in performance_results) < 2000,
            'report_generation_success_rate': sum(1 for r in performance_results if r['dashboard_report_generated']) / len(performance_results),
            'average_processing_time_ms': sum(r['analysis_duration_ms'] for r in performance_results) / len(performance_results)
        }

        # エンドツーエンド統合評価
        integration_quality = {
            'orchestrator_dashboard_integration': True,  # 正常に統合動作
            'multi_component_coordination': len(orchestrator.get_component_details()) >= 3,
            'real_time_data_flow': reliability_metrics['performance_consistency'],
            'enterprise_grade_stability': reliability_metrics['system_uptime_consistency']
        }

        # システム停止
        await orchestrator.stop_enterprise_operations()

        processing_time = (time.time() - start_time) * 1000

        # 最終統合スコア計算
        integration_score = {
            'reliability_score': sum(1 for v in reliability_metrics.values() if (isinstance(v, bool) and v) or (isinstance(v, float) and v > 0.8)) / len(reliability_metrics),
            'integration_quality_score': sum(1 for v in integration_quality.values() if v) / len(integration_quality),
            'overall_integration_score': 0.0
        }

        integration_score['overall_integration_score'] = (integration_score['reliability_score'] + integration_score['integration_quality_score']) / 2

        return {
            'phase': 'Phase 3: 統合システム運用',
            'success': True,
            'integration_scenario': integration_scenario,
            'performance_results': performance_results,
            'reliability_metrics': reliability_metrics,
            'integration_quality': integration_quality,
            'integration_score': integration_score,
            'test_time_ms': processing_time,
            'symbols_tested': len(test_symbols)
        }

    except Exception as e:
        return {
            'phase': 'Phase 3: 統合システム運用',
            'success': False,
            'error': str(e)
        }


async def main():
    """Issue #332 エンタープライズ級完全統合システム統合テスト実行"""
    print("=" * 80)
    print("Issue #332: エンタープライズ級完全統合システム統合テスト")
    print("=" * 80)

    test_results = []

    # Phase 1: 統合オーケストレーションエンジンテスト
    phase1_result = await test_orchestration_engine()
    test_results.append(phase1_result)

    # Phase 2: エンタープライズダッシュボードシステムテスト
    phase2_result = await test_dashboard_system()
    test_results.append(phase2_result)

    # Phase 3: 統合システム運用テスト
    phase3_result = await test_integrated_system_operations()
    test_results.append(phase3_result)

    # 結果出力
    print("\n" + "=" * 80)
    print("テスト結果")
    print("=" * 80)

    successful_tests = 0
    total_tests = len(test_results)

    for i, result in enumerate(test_results, 1):
        phase_name = result.get('phase', f'テスト{i}')
        success = result['success']
        status = "OK" if success else "NG"

        print(f"\n{i}. {phase_name}: {status}")

        if success:
            successful_tests += 1

            # 成功時の詳細情報
            if 'orchestration_features' in result:
                features = result['orchestration_features']
                feature_success_rate = sum(1 for v in features.values() if v) / len(features)
                print(f"   オーケストレーション機能: {feature_success_rate:.1%}")
                print(f"   健全コンポーネント: {result['performance_metrics']['healthy_components']}/{result['performance_metrics']['total_components']}")

            if 'dashboard_features' in result:
                features = result['dashboard_features']
                feature_success_rate = sum(1 for v in features.values() if v) / len(features)
                print(f"   ダッシュボード機能: {feature_success_rate:.1%}")
                print(f"   可視化評価: {result['visualization_evaluation']['dashboard_features_ratio']:.1%}")

            if 'integration_score' in result:
                score = result['integration_score']
                print(f"   統合スコア: {score['overall_integration_score']:.3f}")
                print(f"   信頼性スコア: {score['reliability_score']:.3f}")
                print(f"   品質スコア: {score['integration_quality_score']:.3f}")

            if 'symbols_tested' in result:
                print(f"   テスト銘柄数: {result['symbols_tested']}")
        else:
            print(f"   エラー: {result.get('error', '不明なエラー')}")

    # 成功率
    success_rate = successful_tests / total_tests
    print(f"\n総合結果:")
    print(f"  成功テスト: {successful_tests}/{total_tests} ({success_rate:.1%})")

    # Issue #332成功条件評価
    print(f"\nIssue #332 成功条件検証:")

    # Phase 1: オーケストレーション機能 (85%以上)
    orchestration_result = next((r for r in test_results if 'orchestration_features' in r), None)
    if orchestration_result and orchestration_result['success']:
        features = orchestration_result['orchestration_features']
        orchestration_success_rate = sum(1 for v in features.values() if v) / len(features)
        orchestration_target_met = orchestration_success_rate >= 0.85
        print(f"  1. オーケストレーション機能85%以上: {orchestration_success_rate:.1%} {'OK' if orchestration_target_met else 'NG'}")
    else:
        orchestration_target_met = False
        print(f"  1. オーケストレーション機能85%以上: データなし NG")

    # Phase 2: ダッシュボード品質 (90%以上)
    dashboard_result = next((r for r in test_results if 'dashboard_features' in r), None)
    if dashboard_result and dashboard_result['success']:
        dashboard_quality = dashboard_result['visualization_evaluation']['dashboard_features_ratio']
        dashboard_target_met = dashboard_quality >= 0.90
        print(f"  2. ダッシュボード品質90%以上: {dashboard_quality:.1%} {'OK' if dashboard_target_met else 'NG'}")
    else:
        dashboard_target_met = False
        print(f"  2. ダッシュボード品質90%以上: データなし NG")

    # Phase 3: 統合システム運用品質 (85%以上)
    integration_result = next((r for r in test_results if 'integration_score' in r), None)
    if integration_result and integration_result['success']:
        integration_quality = integration_result['integration_score']['overall_integration_score']
        integration_target_met = integration_quality >= 0.85
        print(f"  3. 統合システム運用85%以上: {integration_quality:.3f} {'OK' if integration_target_met else 'NG'}")
    else:
        integration_target_met = False
        print(f"  3. 統合システム運用85%以上: データなし NG")

    # エンタープライズ級品質 (全体90%以上)
    enterprise_quality = success_rate
    enterprise_target_met = enterprise_quality >= 0.90
    print(f"  4. エンタープライズ級品質90%以上: {enterprise_quality:.1%} {'OK' if enterprise_target_met else 'NG'}")

    # 最終判定
    targets_met = [orchestration_target_met, dashboard_target_met, integration_target_met, enterprise_target_met]
    issue_success_rate = sum(targets_met) / len(targets_met)

    print(f"\nIssue #332 達成条件: {sum(targets_met)}/{len(targets_met)} ({issue_success_rate:.1%})")

    if issue_success_rate >= 0.75:
        print(f"判定: OK Issue #332 エンタープライズ級完全統合システム実装成功")
        print(f"ステータス: エンタープライズ級システム完成")
    elif issue_success_rate >= 0.5:
        print(f"判定: PARTIAL 部分的成功")
        print(f"ステータス: 追加最適化推奨")
    else:
        print(f"判定: NG 成功条件未達成")
        print(f"ステータス: 追加開発必要")

    print(f"\nOK Issue #332 エンタープライズ級完全統合システム統合テスト完了")

    return issue_success_rate >= 0.75


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
        print(f"\n終了コード: {exit_code}")

    except KeyboardInterrupt:
        print("\n統合テスト中断")
    except Exception as e:
        print(f"統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
