#!/usr/bin/env python3
"""
PPO System Simple Test
Windows互換・依存関係最小化テスト
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_ppo_imports():
    """PPOモジュールインポートテスト"""
    print("PPOモジュールインポートテスト開始...")

    try:
        from src.day_trade.rl.ppo_agent import PPOConfig

        config = PPOConfig()
        print("PPOConfig作成成功")
        return True
    except ImportError as e:
        print(f"PPOインポートエラー: {e}")
        return False


def test_trading_environment():
    """取引環境テスト"""
    print("取引環境テスト開始...")

    try:
        from src.day_trade.rl.trading_environment import TradingAction

        action = TradingAction(
            position_size=0.5, asset_allocation={"A": 0.6, "B": 0.4}, risk_level=0.3
        )
        print("TradingAction作成成功")
        return True
    except ImportError as e:
        print(f"環境インポートエラー: {e}")
        return False


def test_ml_engine():
    """MLエンジンテスト"""
    print("MLエンジンテスト開始...")

    try:
        from src.day_trade.data.advanced_ml_engine import ModelConfig

        config = ModelConfig()
        print("ModelConfig作成成功")
        return True
    except Exception as e:
        print(f"MLエンジンエラー: {e}")
        return False


def test_data_pipeline():
    """データパイプラインテスト"""
    print("データパイプラインテスト開始...")

    try:
        from src.day_trade.data.batch_data_fetcher import DataRequest

        request = DataRequest(symbol="TEST", period="30d")
        print("DataRequest作成成功")
        return True
    except Exception as e:
        print(f"データパイプラインエラー: {e}")
        return False


def main():
    """メインテスト"""
    print("=== Next-Gen AI Trading PPO System テスト ===")
    print()

    tests = [
        ("PPOインポート", test_ppo_imports),
        ("取引環境", test_trading_environment),
        ("MLエンジン", test_ml_engine),
        ("データパイプライン", test_data_pipeline),
    ]

    results = []

    for name, test_func in tests:
        print(f"[実行] {name}")
        try:
            result = test_func()
            results.append(result)
            status = "成功" if result else "失敗"
            print(f"[{status}] {name}")
        except Exception as e:
            print(f"[エラー] {name}: {e}")
            results.append(False)
        print()

    # サマリー
    passed = sum(results)
    total = len(results)

    print("=== テスト結果 ===")
    print(f"成功: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")

    if passed == total:
        print("全テスト成功!")
    else:
        print(f"{total-passed}件のテスト失敗")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
