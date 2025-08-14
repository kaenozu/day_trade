#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Readiness Quick Test - 実運用準備クイックテスト

Issue #803対応：軽量版の実運用準備検証
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import time

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class QuickValidationResult:
    """クイック検証結果"""
    def __init__(self, test_name, status, score, details):
        self.test_name = test_name
        self.status = status  # "PASS", "WARNING", "FAIL"
        self.score = score  # 0-100
        self.details = details

async def test_data_provider():
    """データプロバイダーテスト"""
    print("📊 データプロバイダーテスト...")

    try:
        from real_data_provider_v2 import real_data_provider

        # 複数銘柄でテスト
        test_symbols = ["7203", "8306"]
        successful_fetches = 0

        for symbol in test_symbols:
            try:
                data = await real_data_provider.get_stock_data(symbol, "5d")
                if data is not None and len(data) > 0:
                    successful_fetches += 1
            except Exception as e:
                print(f"  ❌ {symbol}: {e}")
                continue

        success_rate = successful_fetches / len(test_symbols) * 100

        if success_rate >= 80:
            status = "PASS"
        elif success_rate >= 50:
            status = "WARNING"
        else:
            status = "FAIL"

        return QuickValidationResult(
            "データプロバイダー",
            status,
            success_rate,
            f"{successful_fetches}/{len(test_symbols)} 銘柄成功"
        )

    except Exception as e:
        return QuickValidationResult(
            "データプロバイダー",
            "FAIL",
            0,
            f"エラー: {e}"
        )

async def test_prediction_system():
    """予測システムテスト"""
    print("🤖 予測システムテスト...")

    try:
        from optimized_prediction_system import optimized_prediction_system

        # 1銘柄でテスト
        symbol = "7203"
        prediction = await optimized_prediction_system.predict_with_optimized_models(symbol)

        if prediction and prediction.confidence > 0.3:
            score = prediction.confidence * 100
            status = "PASS" if score >= 60 else "WARNING" if score >= 40 else "FAIL"
            return QuickValidationResult(
                "予測システム",
                status,
                score,
                f"信頼度: {prediction.confidence:.2f}, 予測: {prediction.prediction}"
            )
        else:
            return QuickValidationResult(
                "予測システム",
                "FAIL",
                0,
                "予測結果が不正または信頼度が低い"
            )

    except Exception as e:
        return QuickValidationResult(
            "予測システム",
            "FAIL",
            0,
            f"エラー: {e}"
        )

async def test_data_quality():
    """データ品質テスト"""
    print("📈 データ品質テスト...")

    try:
        from data_quality_manager import data_quality_manager

        symbol = "7203"
        quality_result = await data_quality_manager.evaluate_data_quality(symbol)

        if quality_result and 'overall_score' in quality_result:
            score = quality_result['overall_score']
            status = "PASS" if score >= 85 else "WARNING" if score >= 70 else "FAIL"
            return QuickValidationResult(
                "データ品質",
                status,
                score,
                f"総合品質スコア: {score:.1f}"
            )
        else:
            return QuickValidationResult(
                "データ品質",
                "WARNING",
                50,
                "品質評価結果が不完全"
            )

    except Exception as e:
        return QuickValidationResult(
            "データ品質",
            "FAIL",
            0,
            f"エラー: {e}"
        )

async def test_performance_optimizer():
    """パフォーマンス最適化テスト"""
    print("⚡ パフォーマンス最適化テスト...")

    try:
        from realtime_performance_optimizer import realtime_performance_optimizer

        # データ取得テスト
        symbol = "7203"
        start_time = time.time()
        data = await realtime_performance_optimizer.optimize_data_retrieval(symbol)
        response_time = time.time() - start_time

        if data is not None:
            score = min(100, max(0, 100 - (response_time * 20)))  # 5秒で0点
            status = "PASS" if score >= 70 else "WARNING" if score >= 40 else "FAIL"
            return QuickValidationResult(
                "パフォーマンス最適化",
                status,
                score,
                f"応答時間: {response_time:.2f}秒"
            )
        else:
            return QuickValidationResult(
                "パフォーマンス最適化",
                "FAIL",
                0,
                "データ取得失敗"
            )

    except Exception as e:
        return QuickValidationResult(
            "パフォーマンス最適化",
            "FAIL",
            0,
            f"エラー: {e}"
        )

async def test_alert_system():
    """アラートシステムテスト"""
    print("🔔 アラートシステムテスト...")

    try:
        from realtime_alert_notification_system import realtime_alert_system

        # テストデータでアラートトリガー
        symbol = "TEST"
        realtime_alert_system.update_market_data(symbol, {
            'current_price': 3000,
            'current_change': 0.08,  # 8%変動でアラート発火
            'volume_ratio': 2.0,
            'signal_strength': 50,
            'risk_score': 30,
            'volatility': 0.2,
            'prediction_confidence': 0.6,
            'signal_consensus': 0.5,
            'error_count': 0
        })

        realtime_alert_system.start_monitoring()
        await realtime_alert_system.check_alert_conditions(symbol)

        # 少し待機してアラート処理
        await asyncio.sleep(1)

        active_alerts = realtime_alert_system.get_active_alerts()
        realtime_alert_system.stop_monitoring()

        if len(active_alerts) > 0:
            return QuickValidationResult(
                "アラートシステム",
                "PASS",
                100,
                f"アラート {len(active_alerts)} 件発火"
            )
        else:
            return QuickValidationResult(
                "アラートシステム",
                "WARNING",
                50,
                "アラート発火なし（設定要確認）"
            )

    except Exception as e:
        return QuickValidationResult(
            "アラートシステム",
            "FAIL",
            0,
            f"エラー: {e}"
        )

async def run_quick_validation():
    """クイック検証実行"""

    print("=== 🚀 実運用準備クイック検証 ===")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 各テストを実行
    tests = [
        test_data_provider(),
        test_prediction_system(),
        test_data_quality(),
        test_performance_optimizer(),
        test_alert_system()
    ]

    print("\n📋 検証実行中...")
    results = []

    for test in tests:
        try:
            result = await asyncio.wait_for(test, timeout=30)
            results.append(result)
        except asyncio.TimeoutError:
            results.append(QuickValidationResult(
                "タイムアウト",
                "FAIL",
                0,
                "30秒でタイムアウト"
            ))
        except Exception as e:
            results.append(QuickValidationResult(
                "エラー",
                "FAIL",
                0,
                f"実行エラー: {e}"
            ))

    # 結果表示
    print("\n" + "="*80)
    print("🎯 実運用準備検証レポート")
    print("="*80)

    total_score = 0
    pass_count = 0
    warning_count = 0
    fail_count = 0

    print("\n📊 個別テスト結果:")

    for result in results:
        status_emoji = {
            "PASS": "✅",
            "WARNING": "⚠️",
            "FAIL": "❌"
        }

        emoji = status_emoji.get(result.status, "❓")
        print(f"  {emoji} {result.test_name}: {result.score:.1f}/100 - {result.details}")

        total_score += result.score

        if result.status == "PASS":
            pass_count += 1
        elif result.status == "WARNING":
            warning_count += 1
        else:
            fail_count += 1

    # 総合評価
    avg_score = total_score / len(results)

    print(f"\n📈 総合結果:")
    print(f"  平均スコア: {avg_score:.1f}/100")
    print(f"  ✅ 合格: {pass_count}")
    print(f"  ⚠️ 警告: {warning_count}")
    print(f"  ❌ 失敗: {fail_count}")

    # 準備状況判定
    if fail_count == 0 and warning_count <= 1:
        readiness = "🎉 実運用準備完了"
        desc = "システムは実運用に向けて準備が整っています"
    elif fail_count <= 1 and avg_score >= 60:
        readiness = "⚠️ 部分的準備完了"
        desc = "軽微な調整後に実運用可能です"
    else:
        readiness = "🔧 追加改善必要"
        desc = "失敗項目の改善が必要です"

    print(f"\n{readiness}")
    print(f"{desc}")

    print("\n💡 推奨事項:")
    if fail_count == 0:
        print("  • 段階的な実運用開始を推奨")
        print("  • 初期は少額取引から開始")
        print("  • システム監視を継続")
    else:
        print("  • 失敗項目の詳細調査と修正")
        print("  • 再検証実施")
        print("  • エラーログの確認")

    print("="*80)

    return {
        'avg_score': avg_score,
        'pass_count': pass_count,
        'warning_count': warning_count,
        'fail_count': fail_count,
        'readiness': readiness
    }

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_quick_validation())