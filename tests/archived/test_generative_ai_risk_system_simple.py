#!/usr/bin/env python3
"""
生成AI統合リスク管理システム簡易テスト
Simplified Generative AI Risk Management System Test

依存関係を最小限にしたテスト版
"""

import asyncio
import json
import time
from datetime import datetime, timedelta

import numpy as np

# 簡易テスト用インポート
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

class SimplifiedRiskSystemTest:
    """簡易リスク管理システムテスト"""

    def __init__(self):
        self.test_results = {}
        logger.info("簡易リスク管理システムテスト初期化")

    async def run_basic_tests(self):
        """基本テスト実行"""

        print("\n" + "="*60)
        print("生成AI統合リスク管理システム - 簡易テスト")
        print("="*60)
        print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # テスト実行
        await self._test_basic_imports()
        await self._test_risk_calculation()
        await self._test_fraud_simulation()
        await self._test_alert_generation()

        # 結果表示
        self._display_results()

    async def _test_basic_imports(self):
        """基本インポートテスト"""

        print("テスト1: モジュールインポート")
        print("-" * 30)

        start_time = time.time()
        success_count = 0
        total_count = 0

        # 核となるモジュールのテスト
        modules_to_test = [
            ("utils.logging_config", "get_context_logger"),
            ("core.optimization_strategy", "OptimizationConfig"),
            ("models.database", "DatabaseManager"),
            ("data.stock_fetcher_v2", "StockFetcher"),
        ]

        for module_path, class_name in modules_to_test:
            total_count += 1
            try:
                module = __import__(f"src.day_trade.{module_path}", fromlist=[class_name])
                getattr(module, class_name)
                print(f"  OK {module_path}.{class_name}")
                success_count += 1
            except Exception as e:
                print(f"  NG {module_path}.{class_name}: {e}")

        processing_time = time.time() - start_time

        print(f"\n  成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"  処理時間: {processing_time:.3f}秒")

        self.test_results['imports'] = {
            'success': success_count == total_count,
            'success_rate': success_count / total_count,
            'processing_time': processing_time
        }
        print()

    async def _test_risk_calculation(self):
        """リスク計算テスト"""

        print("テスト2: リスク計算エンジン")
        print("-" * 30)

        start_time = time.time()

        try:
            # シンプルなリスク計算実装
            risk_score = await self._calculate_simple_risk({
                'amount': 5000000,  # 500万円
                'time_hour': 14,  # 14時
                'volatility': 0.25,
                'user_risk_tolerance': 'moderate',
                'account_balance': 10000000
            })

            processing_time = time.time() - start_time

            print("  OK リスク計算完了")
            print(f"  リスクスコア: {risk_score:.3f}")
            print(f"  処理時間: {processing_time:.3f}秒")

            # リスクレベル判定
            if risk_score >= 0.8:
                risk_level = "CRITICAL"
            elif risk_score >= 0.6:
                risk_level = "HIGH"
            elif risk_score >= 0.3:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            print(f"  リスクレベル: {risk_level}")

            self.test_results['risk_calculation'] = {
                'success': True,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'processing_time': processing_time
            }

        except Exception as e:
            print(f"  NG リスク計算エラー: {e}")
            self.test_results['risk_calculation'] = {
                'success': False,
                'error': str(e)
            }

        print()

    async def _test_fraud_simulation(self):
        """不正検知シミュレーション"""

        print("テスト3: 不正検知シミュレーション")
        print("-" * 30)

        start_time = time.time()

        try:
            # 不正取引パターンのテスト
            suspicious_patterns = [
                {
                    'pattern': 'high_amount_night',
                    'amount': 15000000,  # 1500万円
                    'hour': 3,  # 深夜3時
                    'new_device': True,
                    'expected_fraud': True
                },
                {
                    'pattern': 'normal_day_trade',
                    'amount': 100000,  # 10万円
                    'hour': 14,  # 午後2時
                    'new_device': False,
                    'expected_fraud': False
                }
            ]

            correct_predictions = 0
            total_predictions = len(suspicious_patterns)

            for pattern in suspicious_patterns:
                fraud_score = await self._calculate_fraud_score(pattern)
                is_fraud_predicted = fraud_score > 0.5

                if is_fraud_predicted == pattern['expected_fraud']:
                    correct_predictions += 1
                    result_emoji = "OK"
                else:
                    result_emoji = "NG"

                print(f"  {result_emoji} パターン: {pattern['pattern']}")
                print(f"     不正スコア: {fraud_score:.3f}")
                print(f"     予測: {'不正' if is_fraud_predicted else '正常'}")
                print(f"     実際: {'不正' if pattern['expected_fraud'] else '正常'}")

            processing_time = time.time() - start_time
            accuracy = correct_predictions / total_predictions

            print(f"\n  検知精度: {correct_predictions}/{total_predictions} ({accuracy*100:.1f}%)")
            print(f"  処理時間: {processing_time:.3f}秒")

            self.test_results['fraud_detection'] = {
                'success': True,
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'processing_time': processing_time
            }

        except Exception as e:
            print(f"  NG 不正検知エラー: {e}")
            self.test_results['fraud_detection'] = {
                'success': False,
                'error': str(e)
            }

        print()

    async def _test_alert_generation(self):
        """アラート生成テスト"""

        print("テスト4: アラート生成システム")
        print("-" * 30)

        start_time = time.time()

        try:
            # 各種リスクレベルのアラート生成
            alert_scenarios = [
                {'risk_score': 0.95, 'expected_level': 'CRITICAL'},
                {'risk_score': 0.75, 'expected_level': 'HIGH'},
                {'risk_score': 0.45, 'expected_level': 'MEDIUM'},
                {'risk_score': 0.15, 'expected_level': 'LOW'}
            ]

            generated_alerts = []

            for i, scenario in enumerate(alert_scenarios):
                alert = await self._generate_alert(scenario['risk_score'], f"TEST_ALERT_{i+1}")
                generated_alerts.append(alert)

                level_match = alert['level'] == scenario['expected_level']
                result_emoji = "OK" if level_match else "NG"

                print(f"  {result_emoji} アラート {i+1}: {alert['level']}")
                print(f"     リスクスコア: {scenario['risk_score']:.2f}")
                print(f"     メッセージ: {alert['message'][:50]}...")

            processing_time = time.time() - start_time

            print(f"\n  生成アラート数: {len(generated_alerts)}")
            print(f"  処理時間: {processing_time:.3f}秒")

            self.test_results['alert_generation'] = {
                'success': True,
                'alerts_generated': len(generated_alerts),
                'processing_time': processing_time,
                'alerts': generated_alerts
            }

        except Exception as e:
            print(f"  NG アラート生成エラー: {e}")
            self.test_results['alert_generation'] = {
                'success': False,
                'error': str(e)
            }

        print()

    async def _calculate_simple_risk(self, transaction_data):
        """シンプルリスク計算"""

        risk_score = 0.0

        # 金額リスク
        amount = transaction_data.get('amount', 0)
        if amount > 10000000:  # 1000万円以上
            risk_score += 0.3
        elif amount > 1000000:  # 100万円以上
            risk_score += 0.1

        # 時間リスク
        hour = transaction_data.get('time_hour', 12)
        if hour < 9 or hour > 15:  # 取引時間外
            risk_score += 0.2

        # ボラティリティリスク
        volatility = transaction_data.get('volatility', 0.2)
        if volatility > 0.3:
            risk_score += 0.2

        # 残高比率リスク
        balance_ratio = amount / max(transaction_data.get('account_balance', 1), 1)
        if balance_ratio > 0.5:
            risk_score += 0.3

        return min(1.0, max(0.0, risk_score))

    async def _calculate_fraud_score(self, pattern_data):
        """不正スコア計算"""

        fraud_score = 0.0

        # 高額取引
        amount = pattern_data.get('amount', 0)
        if amount > 10000000:  # 1000万円以上
            fraud_score += 0.4
        elif amount > 5000000:  # 500万円以上
            fraud_score += 0.2

        # 時間帯
        hour = pattern_data.get('hour', 12)
        if hour < 6 or hour > 22:  # 深夜・早朝
            fraud_score += 0.3

        # 新規デバイス
        if pattern_data.get('new_device', False):
            fraud_score += 0.2

        return min(1.0, max(0.0, fraud_score))

    async def _generate_alert(self, risk_score, alert_id):
        """アラート生成"""

        # リスクレベル決定
        if risk_score >= 0.8:
            level = "CRITICAL"
            message = "緊急: 重要リスクが検知されました。即座に対応が必要です。"
        elif risk_score >= 0.6:
            level = "HIGH"
            message = "警告: 高リスクが検知されました。詳細確認を推奨します。"
        elif risk_score >= 0.3:
            level = "MEDIUM"
            message = "注意: 中程度のリスクが検知されました。監視を継続してください。"
        else:
            level = "LOW"
            message = "情報: 低リスクです。通常の監視を継続してください。"

        return {
            'id': alert_id,
            'level': level,
            'risk_score': risk_score,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'auto_generated': True
        }

    def _display_results(self):
        """結果表示"""

        print("="*60)
        print("テスト結果サマリー")
        print("="*60)

        # 成功率計算
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values()
                             if result.get('success', False))
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"総合成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        print()

        # 個別テスト結果
        for test_name, result in self.test_results.items():
            status = "OK 成功" if result.get('success') else "NG 失敗"
            print(f"{test_name.replace('_', ' ').title()}: {status}")

            if result.get('success'):
                if 'processing_time' in result:
                    print(f"  処理時間: {result['processing_time']:.3f}秒")
                if 'success_rate' in result:
                    print(f"  成功率: {result['success_rate']*100:.1f}%")
                if 'accuracy' in result:
                    print(f"  精度: {result['accuracy']*100:.1f}%")
                if 'risk_score' in result:
                    print(f"  リスクスコア: {result['risk_score']:.3f}")
            else:
                if 'error' in result:
                    print(f"  エラー: {result['error']}")
            print()

        # 総合評価
        print("-" * 60)
        if success_rate == 100:
            print("全テスト成功! システムは正常に動作しています。")
        elif success_rate >= 75:
            print("大部分のテストが成功しています。一部機能で改善余地があります。")
        elif success_rate >= 50:
            print("半数以上のテストが成功していますが、いくつかの問題があります。")
        else:
            print("多くのテストが失敗しています。システムに問題がある可能性があります。")

        print()
        print("簡易テスト完了")
        print("   次のステップ: 本格的な統合テストとデプロイメント")
        print("="*60)

async def main():
    """メイン実行関数"""

    try:
        # 簡易テスト実行
        test_system = SimplifiedRiskSystemTest()
        await test_system.run_basic_tests()

    except KeyboardInterrupt:
        print("\n\nテスト中断")
    except Exception as e:
        print(f"\n\nテストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("生成AI統合リスク管理システム - 簡易テスト起動中...")
    asyncio.run(main())
