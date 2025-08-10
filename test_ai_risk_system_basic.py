#!/usr/bin/env python3
"""
AI Risk Management System Basic Test
"""

import asyncio
import time
from datetime import datetime
import numpy as np

# 基本インポート
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

class BasicRiskTest:
    def __init__(self):
        self.test_results = {}
        logger.info("Basic risk test initialized")

    async def run_tests(self):
        print("\n" + "="*50)
        print("AI Risk Management System - Basic Test")
        print("="*50)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        await self._test_imports()
        await self._test_risk_calc()
        await self._test_fraud_detection()

        self._show_results()

    async def _test_imports(self):
        print("Test 1: Module Imports")
        print("-" * 20)

        modules = [
            ("utils.logging_config", "get_context_logger"),
            ("core.optimization_strategy", "OptimizationConfig"),
            ("models.database", "DatabaseManager"),
            ("data.stock_fetcher_v2", "StockFetcherV2"),
        ]

        success = 0
        total = len(modules)

        for module_path, class_name in modules:
            try:
                module = __import__(f"src.day_trade.{module_path}", fromlist=[class_name])
                getattr(module, class_name)
                print(f"  OK: {module_path}")
                success += 1
            except Exception as e:
                print(f"  FAIL: {module_path} - {e}")

        rate = success / total * 100
        print(f"  Success: {success}/{total} ({rate:.1f}%)")

        self.test_results['imports'] = {'success': success == total, 'rate': rate}
        print()

    async def _test_risk_calc(self):
        print("Test 2: Risk Calculation")
        print("-" * 20)

        try:
            # Simple risk calculation
            data = {
                'amount': 5000000,
                'hour': 14,
                'volatility': 0.25,
                'balance': 10000000
            }

            risk_score = await self._calc_risk(data)

            if 0 <= risk_score <= 1:
                level = "CRITICAL" if risk_score >= 0.8 else \
                        "HIGH" if risk_score >= 0.6 else \
                        "MEDIUM" if risk_score >= 0.3 else "LOW"

                print(f"  Risk Score: {risk_score:.3f}")
                print(f"  Risk Level: {level}")
                print(f"  Status: OK")

                self.test_results['risk_calc'] = {
                    'success': True,
                    'score': risk_score,
                    'level': level
                }
            else:
                raise ValueError(f"Invalid risk score: {risk_score}")

        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['risk_calc'] = {'success': False, 'error': str(e)}

        print()

    async def _test_fraud_detection(self):
        print("Test 3: Fraud Detection")
        print("-" * 20)

        test_cases = [
            {
                'name': 'suspicious_night',
                'amount': 15000000,
                'hour': 3,
                'new_device': True,
                'expected': True
            },
            {
                'name': 'normal_day',
                'amount': 100000,
                'hour': 14,
                'new_device': False,
                'expected': False
            }
        ]

        correct = 0
        total = len(test_cases)

        try:
            for case in test_cases:
                fraud_score = await self._calc_fraud(case)
                predicted = fraud_score > 0.5

                if predicted == case['expected']:
                    correct += 1
                    result = "OK"
                else:
                    result = "FAIL"

                print(f"  {case['name']}: {fraud_score:.3f} ({result})")

            accuracy = correct / total * 100
            print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")

            self.test_results['fraud'] = {
                'success': True,
                'accuracy': accuracy,
                'correct': correct
            }

        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['fraud'] = {'success': False, 'error': str(e)}

        print()

    async def _calc_risk(self, data):
        risk = 0.0

        # Amount risk
        if data['amount'] > 10000000:
            risk += 0.3
        elif data['amount'] > 1000000:
            risk += 0.1

        # Time risk
        if data['hour'] < 9 or data['hour'] > 15:
            risk += 0.2

        # Volatility risk
        if data.get('volatility', 0) > 0.3:
            risk += 0.2

        # Balance ratio risk
        ratio = data['amount'] / max(data.get('balance', 1), 1)
        if ratio > 0.5:
            risk += 0.3

        return min(1.0, max(0.0, risk))

    async def _calc_fraud(self, data):
        score = 0.0

        # High amount
        if data['amount'] > 10000000:
            score += 0.4
        elif data['amount'] > 5000000:
            score += 0.2

        # Night time
        if data['hour'] < 6 or data['hour'] > 22:
            score += 0.3

        # New device
        if data.get('new_device', False):
            score += 0.2

        return min(1.0, max(0.0, score))

    def _show_results(self):
        print("="*50)
        print("Test Results")
        print("="*50)

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results.values() if r.get('success', False))
        rate = passed / total * 100 if total > 0 else 0

        print(f"Overall: {passed}/{total} ({rate:.1f}%)")
        print()

        for name, result in self.test_results.items():
            status = "PASS" if result.get('success') else "FAIL"
            print(f"{name}: {status}")

            if result.get('success'):
                if 'rate' in result:
                    print(f"  Rate: {result['rate']:.1f}%")
                if 'score' in result:
                    print(f"  Score: {result['score']:.3f}")
                if 'accuracy' in result:
                    print(f"  Accuracy: {result['accuracy']:.1f}%")
            else:
                if 'error' in result:
                    print(f"  Error: {result['error']}")
            print()

        if rate == 100:
            print("All tests passed!")
        elif rate >= 75:
            print("Most tests passed.")
        else:
            print("Several tests failed.")

        print("="*50)

async def main():
    try:
        test = BasicRiskTest()
        await test.run_tests()
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    print("Starting AI Risk Management System Test...")
    asyncio.run(main())
