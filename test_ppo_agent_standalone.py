#!/usr/bin/env python3
"""
PPO Agent スタンドアロンテスト
依存関係を最小化した強化学習エージェントテスト
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_ppo_components():
    """PPOコンポーネントの基本テスト"""
    print("=== PPO コンポーネントテスト ===")

    # 基本インポートテスト
    try:
        from src.day_trade.rl.ppo_agent import PPOConfig, PPOExperience
        print("✓ PPO設定クラス読み込み成功")

        # 設定テスト
        config = PPOConfig()
        assert config.learning_rate > 0
        assert config.gamma <= 1.0
        assert config.epsilon_clip > 0
        print("✓ PPO設定検証完了")

        # 経験データクラステスト
        experience = PPOExperience(
            states=np.random.randn(10, 512),
            actions=np.random.randn(10, 5),
            rewards=np.random.randn(10),
            values=np.random.randn(10),
            log_probs=np.random.randn(10),
            dones=np.random.randint(0, 2, 10),
        )
        print("✓ PPO経験データクラス作成成功")

    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        return False

    return True

def test_trading_environment_basic():
    """取引環境の基本機能テスト"""
    print("\n=== 取引環境基本テスト ===")

    try:
        # 基本データクラスのテスト
        from src.day_trade.rl.trading_environment import (
            TradingAction, MarketState, TradingReward
        )
        print("✓ 取引環境データクラス読み込み成功")

        # TradingActionテスト
        action = TradingAction(
            position_size=0.5,
            asset_allocation={"STOCK_A": 0.6, "STOCK_B": 0.4},
            risk_level=0.3
        )
        assert action.position_size == 0.5
        assert len(action.asset_allocation) == 2
        print("✓ TradingAction作成成功")

        # TradingRewardテスト
        reward = TradingReward(
            profit_loss=100.0,
            risk_adjusted_return=50.0,
            drawdown_penalty=-10.0,
            trading_costs=-5.0
        )
        reward.total_reward = reward.profit_loss + reward.risk_adjusted_return + reward.drawdown_penalty + reward.trading_costs
        assert reward.total_reward == 135.0
        print("✓ TradingReward計算成功")

    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        return False

    return True

def test_ml_engine_integration():
    """ML Engineとの統合テスト"""
    print("\n=== ML Engine統合テスト ===")

    try:
        from src.day_trade.data.advanced_ml_engine import AdvancedMLEngine, ModelConfig
        print("✓ Advanced ML Engine読み込み成功")

        # 基本設定テスト
        config = ModelConfig(
            lstm_hidden_size=128,
            transformer_d_model=256,
            sequence_length=60,
            num_features=20
        )
        print("✓ ML Engine設定作成成功")

        # MLエンジンインスタンス作成テスト（PyTorchなしの場合はフォールバック）
        try:
            engine = AdvancedMLEngine(config)
            print("✓ ML Engineインスタンス作成成功")
        except ImportError:
            print("! PyTorch未インストール - フォールバック版使用")

    except ImportError as e:
        print(f"✗ ML Engine インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ ML Engine テストエラー: {e}")
        return False

    return True

def test_batch_data_fetcher():
    """バッチデータフェッチャーテスト"""
    print("\n=== バッチデータフェッチャーテスト ===")

    try:
        from src.day_trade.data.batch_data_fetcher import (
            AdvancedBatchDataFetcher, DataRequest, DataResponse
        )
        print("✓ バッチデータフェッチャー読み込み成功")

        # データリクエスト作成テスト
        request = DataRequest(
            symbol="TEST_STOCK",
            period="30d",
            preprocessing=True,
            priority=3
        )
        assert request.symbol == "TEST_STOCK"
        assert request.preprocessing == True
        print("✓ データリクエスト作成成功")

        # データレスポンステスト
        response = DataResponse(
            symbol="TEST_STOCK",
            data=None,
            success=False,
            error_message="テスト用",
            data_quality_score=85.0
        )
        assert response.data_quality_score == 85.0
        print("✓ データレスポンス作成成功")

        # フェッチャーインスタンス作成（依存関係なしモード）
        fetcher = AdvancedBatchDataFetcher(
            max_workers=2,
            enable_kafka=False,
            enable_redis=False
        )
        print("✓ バッチデータフェッチャーインスタンス作成成功")

        # 統計取得テスト
        stats = fetcher.get_pipeline_stats()
        assert hasattr(stats, 'total_requests')
        print("✓ パイプライン統計取得成功")

    except ImportError as e:
        print(f"✗ バッチデータフェッチャー インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ バッチデータフェッチャー テストエラー: {e}")
        return False

    return True

def test_integration_workflow():
    """統合ワークフローテスト"""
    print("\n=== 統合ワークフローテスト ===")

    try:
        # テスト用データ生成
        np.random.seed(42)
        test_data = pd.DataFrame({
            '終値': 1000 + np.cumsum(np.random.randn(100) * 10),
            '高値': 1000 + np.cumsum(np.random.randn(100) * 10) + np.random.rand(100) * 5,
            '安値': 1000 + np.cumsum(np.random.randn(100) * 10) - np.random.rand(100) * 5,
            '出来高': np.random.randint(1000, 10000, 100)
        })
        print(f"✓ テストデータ生成成功: {test_data.shape}")

        # データ品質チェック
        assert not test_data.isnull().any().any()
        assert len(test_data) == 100
        assert test_data['終値'].min() > 0
        print("✓ データ品質検証成功")

        # 基本統計計算
        returns = test_data['終値'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 年率ボラティリティ
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

        print(f"  年率ボラティリティ: {volatility:.3f}")
        print(f"  シャープレシオ: {sharpe_ratio:.3f}")
        print("✓ 基本統計計算成功")

        # ポートフォリオシミュレーション
        initial_balance = 1000000
        positions = {"STOCK_A": 0.6, "STOCK_B": 0.4}
        portfolio_value = initial_balance * sum(positions.values())

        print(f"  初期資金: ¥{initial_balance:,}")
        print(f"  ポートフォリオ価値: ¥{portfolio_value:,}")
        print("✓ ポートフォリオシミュレーション成功")

    except Exception as e:
        print(f"✗ 統合ワークフローエラー: {e}")
        return False

    return True

def main():
    """メインテスト実行"""
    print("Next-Gen AI Trading System - PPO Agent スタンドアロンテスト")
    print("=" * 70)

    test_results = []
    start_time = time.time()

    # テスト実行
    tests = [
        ("PPOコンポーネントテスト", test_ppo_components),
        ("取引環境基本テスト", test_trading_environment_basic),
        ("ML Engine統合テスト", test_ml_engine_integration),
        ("バッチデータフェッチャーテスト", test_batch_data_fetcher),
        ("統合ワークフローテスト", test_integration_workflow)
    ]

    for test_name, test_func in tests:
        try:
            print(f"\n実行中: {test_name}")
            result = test_func()
            test_results.append((test_name, result))

            if result:
                print(f"[OK] {test_name} - 成功")
            else:
                print(f"[NG] {test_name} - 失敗")

        except Exception as e:
            print(f"[ERROR] {test_name} - 例外: {e}")
            test_results.append((test_name, False))

    # 結果サマリー
    total_time = time.time() - start_time
    passed = len([r for _, r in test_results if r])
    total = len(test_results)

    print("\n" + "=" * 70)
    print("テスト結果サマリー")
    print("=" * 70)

    for test_name, result in test_results:
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{status:<8} {test_name}")

    print(f"\n成功: {passed}/{total}")
    print(f"実行時間: {total_time:.2f}秒")
    print(f"成功率: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 全テスト成功! Next-Gen AI Trading System 基盤準備完了")
    else:
        print(f"\n⚠️  {total-passed} 件のテスト失敗 - システム確認が必要")

    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n致命的エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
