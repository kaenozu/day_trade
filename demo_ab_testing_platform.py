#!/usr/bin/env python3
"""
A/Bテスト・デプロイメント統合プラットフォーム デモンストレーション

Issue #733対応: MLモデルA/Bテストフレームワークの完全デモ
実際の株式取引システムでのモデル改善プロセスをシミュレート
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.ml.ml_experimentation_platform import (
    MLExperimentationPlatform, ModelVersion
)
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockPredictionSimulator:
    """株価予測シミュレータ"""

    def __init__(self):
        """初期化"""
        self.symbols = ["7203", "8306", "9984", "6758", "4689", "1605", "2914", "8058", "3382", "9437"]
        self.base_prices = {symbol: 1000 + hash(symbol) % 500 for symbol in self.symbols}
        self.market_volatility = 0.02  # 2%のボラティリティ

        # モデル性能設定
        self.model_performance = {
            "RandomForest_v1": {"accuracy": 0.75, "bias": 0.0, "latency": 65.0},
            "RandomForest_v2": {"accuracy": 0.78, "bias": 0.5, "latency": 55.0},
            "XGBoost_v1": {"accuracy": 0.80, "bias": 1.0, "latency": 45.0},
            "LSTM_v1": {"accuracy": 0.82, "bias": 1.5, "latency": 85.0},
        }

    def generate_market_data(self, symbol: str, timestamp: datetime) -> dict:
        """市場データの生成"""
        base_price = self.base_prices[symbol]

        # 時間ベースの価格変動
        time_factor = np.sin((timestamp.hour + timestamp.minute / 60) * np.pi / 12)
        daily_trend = np.random.normal(0, self.market_volatility)

        current_price = base_price * (1 + daily_trend + time_factor * 0.01)

        return {
            "symbol": symbol,
            "current_price": current_price,
            "timestamp": timestamp,
            "volatility": abs(daily_trend),
            "volume": np.random.uniform(10000, 100000)
        }

    def simulate_model_prediction(self, model_name: str, market_data: dict) -> dict:
        """モデル予測のシミュレート"""
        if model_name not in self.model_performance:
            model_name = "RandomForest_v1"  # デフォルト

        perf = self.model_performance[model_name]
        current_price = market_data["current_price"]

        # 実際の価格変動（未来の値）
        true_change = np.random.normal(0, 0.01)  # 1%の標準的な変動
        actual_future_price = current_price * (1 + true_change)

        # モデル予測（精度とバイアスを考慮）
        prediction_noise = np.random.normal(0, 1 - perf["accuracy"])
        prediction_change = true_change + perf["bias"] * 0.001 + prediction_noise * 0.02
        predicted_price = current_price * (1 + prediction_change)

        # レイテンシのシミュレート
        latency = np.random.normal(perf["latency"], 10)

        return {
            "prediction": predicted_price,
            "actual_value": actual_future_price,
            "latency_ms": max(10, latency),
            "confidence": perf["accuracy"] + np.random.uniform(-0.05, 0.05),
            "model_name": model_name
        }


async def run_comprehensive_ab_test_demo():
    """包括的A/Bテストデモンストレーション"""

    print("=" * 80)
    print("🚀 ML A/Bテスト・デプロイメントプラットフォーム 包括的デモ")
    print("   Issue #733: MLモデル用A/Bテストフレームワーク")
    print("=" * 80)

    # 1. プラットフォーム初期化
    print("\n📋 1. プラットフォーム初期化")
    print("-" * 50)

    platform = MLExperimentationPlatform("data/ab_testing_demo")
    simulator = StockPredictionSimulator()

    print("✅ ML実験プラットフォーム初期化完了")
    print("✅ 株価予測シミュレータ初期化完了")

    # 2. モデル比較実験の設定
    print("\n🧪 2. モデル比較実験の設定")
    print("-" * 50)

    experiments = [
        {
            "name": "RandomForest v1 vs v2 Performance Comparison",
            "control_model": {"model_type": "RandomForest_v1", "version": "1.0", "n_estimators": 100},
            "test_model": {"model_type": "RandomForest_v2", "version": "2.0", "n_estimators": 120},
            "traffic_split": 0.3,
            "duration_hours": 0.1  # デモ用に短時間
        },
        {
            "name": "RandomForest vs XGBoost Model Comparison",
            "control_model": {"model_type": "RandomForest_v2", "version": "2.0", "n_estimators": 120},
            "test_model": {"model_type": "XGBoost_v1", "version": "1.0", "max_depth": 6},
            "traffic_split": 0.4,
            "duration_hours": 0.1
        },
        {
            "name": "Traditional ML vs Deep Learning Comparison",
            "control_model": {"model_type": "XGBoost_v1", "version": "1.0", "max_depth": 6},
            "test_model": {"model_type": "LSTM_v1", "version": "1.0", "hidden_units": 128},
            "traffic_split": 0.2,  # 保守的なトラフィック配分
            "duration_hours": 0.1
        }
    ]

    experiment_ids = []

    for exp_config in experiments:
        experiment_id = await platform.create_model_comparison_experiment(
            experiment_name=exp_config["name"],
            control_model=exp_config["control_model"],
            test_model=exp_config["test_model"],
            traffic_split=exp_config["traffic_split"],
            experiment_duration_hours=exp_config["duration_hours"]
        )

        if experiment_id:
            experiment_ids.append(experiment_id)
            print(f"✅ 実験作成成功: {exp_config['name'][:50]}...")
            print(f"   - 実験ID: {experiment_id}")
            print(f"   - トラフィック分割: {exp_config['traffic_split']*100:.1f}%")
        else:
            print(f"❌ 実験作成失敗: {exp_config['name']}")

    print(f"\n📊 作成された実験数: {len(experiment_ids)}")

    # 3. 実験データの生成・記録
    print("\n📈 3. 実験データの生成・記録")
    print("-" * 50)

    prediction_count = 0
    start_time = time.time()

    # 各実験で予測データを生成
    for i, experiment_id in enumerate(experiment_ids):
        exp_config = experiments[i]
        control_model = exp_config["control_model"]["model_type"]
        test_model = exp_config["test_model"]["model_type"]

        print(f"\n🔬 実験 {i+1}: {control_model} vs {test_model}")

        # シミュレート予測の実行
        for prediction_round in range(100):  # 各実験100回の予測
            symbol = np.random.choice(simulator.symbols)
            current_time = datetime.now() + timedelta(minutes=prediction_round)

            # 市場データの生成
            market_data = simulator.generate_market_data(symbol, current_time)

            # 実験グループの割り当てを取得するために一時的な予測
            temp_prediction = simulator.simulate_model_prediction(control_model, market_data)

            # プラットフォームを通じて予測を記録
            success = await platform.record_model_prediction(
                experiment_id=experiment_id,
                symbol=symbol,
                prediction=temp_prediction["prediction"],
                actual_value=temp_prediction["actual_value"],
                latency_ms=temp_prediction["latency_ms"],
                metadata={
                    "confidence": temp_prediction["confidence"],
                    "model_name": temp_prediction["model_name"],
                    "market_volatility": market_data["volatility"],
                    "volume": market_data["volume"]
                }
            )

            if success:
                prediction_count += 1

            # 進捗表示（10回ごと）
            if prediction_round % 25 == 0 and prediction_round > 0:
                print(f"   📊 {prediction_round} 予測完了...")

        print(f"   ✅ 実験 {i+1} データ生成完了: 100 予測")

    elapsed_time = time.time() - start_time
    print(f"\n📊 データ生成サマリー:")
    print(f"   - 総予測回数: {prediction_count}")
    print(f"   - 実行時間: {elapsed_time:.2f}秒")
    print(f"   - 平均予測速度: {prediction_count/elapsed_time:.1f} 予測/秒")

    # 4. 実験結果の分析
    print("\n🔍 4. 実験結果の分析")
    print("-" * 50)

    analysis_results = []

    for i, experiment_id in enumerate(experiment_ids):
        exp_config = experiments[i]

        print(f"\n📈 実験分析 {i+1}: {exp_config['name'][:60]}...")

        # パフォーマンス比較の実行
        performance_comparison = await platform.analyze_experiment_performance(experiment_id)

        if performance_comparison:
            analysis_results.append(performance_comparison)

            print(f"   ✅ 分析完了")
            print(f"   📊 推奨事項: {performance_comparison.recommendation}")
            print(f"   🎯 信頼度: {performance_comparison.confidence_score:.3f}")

            # 詳細メトリクス表示
            if performance_comparison.metric_comparisons:
                print("   📋 メトリクス比較:")
                for metric, comparison in performance_comparison.metric_comparisons.items():
                    change_pct = comparison.get("improvement_percentage", 0)
                    if abs(change_pct) > 1:  # 1%以上の変化のみ表示
                        direction = "↗️" if change_pct > 0 else "↘️"
                        print(f"      {metric}: {direction} {change_pct:+.1f}%")
        else:
            print(f"   ❌ 分析失敗")

    print(f"\n📊 分析完了: {len(analysis_results)} / {len(experiment_ids)} 実験")

    # 5. デプロイメント推奨事項
    print("\n🚀 5. デプロイメント推奨事項")
    print("-" * 50)

    deployment_decisions = []

    for i, experiment_id in enumerate(experiment_ids):
        exp_config = experiments[i]

        print(f"\n🎯 デプロイメント判定 {i+1}: {exp_config['name'][:50]}...")

        # デプロイメントコンテキストの設定
        deployment_context = {
            'daily_request_volume': np.random.uniform(10000, 500000),
            'recent_deployments_count': np.random.randint(0, 3),
            'model_complexity_score': 0.3 + i * 0.2  # 実験順で複雑さが増加
        }

        # デプロイメント推奨の取得
        deployment_decision = await platform.get_deployment_recommendation(
            experiment_id, deployment_context
        )

        if deployment_decision:
            deployment_decisions.append(deployment_decision)

            print(f"   📋 判定結果: {'🟢 デプロイ推奨' if deployment_decision.deploy_recommended else '🟡 デプロイ見送り'}")
            print(f"   🎯 信頼度: {deployment_decision.confidence_level:.3f}")
            print(f"   🚀 推奨戦略: {deployment_decision.deployment_strategy.value}")
            print(f"   📝 判定理由:")
            for reason in deployment_decision.reasons[:3]:  # 最初の3つのみ表示
                print(f"      • {reason}")

            # リスク評価の表示
            overall_risk = deployment_decision.risk_assessment.get('overall_risk', 0.5)
            risk_level = "🟢 低" if overall_risk < 0.3 else "🟡 中" if overall_risk < 0.7 else "🔴 高"
            print(f"   ⚠️  全体リスク: {risk_level} ({overall_risk:.3f})")
        else:
            print(f"   ❌ 判定失敗")

    # 6. 自動デプロイメントのシミュレート
    print("\n🤖 6. 自動デプロイメント実行")
    print("-" * 50)

    deployed_count = 0

    for i, decision in enumerate(deployment_decisions):
        if decision.deploy_recommended:
            exp_config = experiments[i]
            test_model = exp_config["test_model"]

            print(f"\n🚀 自動デプロイメント実行 {deployed_count + 1}:")
            print(f"   📦 モデル: {test_model['model_type']} v{test_model.get('version', '1.0')}")

            # デプロイ用モデルバージョンの作成
            model_version = ModelVersion(
                version_id=f"{test_model['model_type']}_v{test_model.get('version', '1.0')}",
                name=f"{test_model['model_type']} Version {test_model.get('version', '1.0')}",
                description=f"A/Bテストで選択された{test_model['model_type']}モデル",
                model_path=f"/models/{test_model['model_type'].lower()}_v{test_model.get('version', '1.0')}.joblib",
                config=test_model,
                performance_metrics={
                    "accuracy": 0.8 + np.random.uniform(-0.05, 0.05),
                    "latency_ms": 50 + np.random.uniform(-15, 15),
                    "throughput": 1000 + np.random.uniform(-200, 200)
                }
            )

            # 自動デプロイメントの実行
            deployment_id = await platform.execute_auto_deployment(decision, model_version)

            if deployment_id:
                print(f"   ✅ デプロイメント開始: {deployment_id}")
                deployed_count += 1

                # 短時間待機してデプロイメント状態を確認
                await asyncio.sleep(1)

                deployment_state = platform.deployment_manager.get_deployment_status(deployment_id)
                if deployment_state:
                    print(f"   📊 デプロイメント状態: {deployment_state.status.value}")
                    print(f"   🎯 現在のトラフィック: {deployment_state.current_traffic_percentage*100:.1f}%")
            else:
                print(f"   ❌ デプロイメント失敗")
        else:
            print(f"\n⏸️  デプロイメントスキップ {i+1}: 推奨されていません")

    print(f"\n📊 デプロイメントサマリー: {deployed_count} / {len(deployment_decisions)} デプロイメント実行")

    # 7. プラットフォーム状態の最終確認
    print("\n📋 7. プラットフォーム状態サマリー")
    print("-" * 50)

    active_experiments = platform.get_active_experiments()
    active_deployments = platform.get_active_deployments()
    experiment_pairs = platform.get_experiment_deployment_pairs()

    print(f"📊 最終統計:")
    print(f"   🧪 作成実験数: {len(experiment_ids)}")
    print(f"   📈 分析完了数: {len(analysis_results)}")
    print(f"   🎯 デプロイメント判定数: {len(deployment_decisions)}")
    print(f"   🚀 実行デプロイメント数: {deployed_count}")
    print(f"   🔗 実験-デプロイペア数: {len(experiment_pairs)}")

    # 成功率の計算
    experiment_success_rate = len(experiment_ids) / len(experiments) * 100
    analysis_success_rate = len(analysis_results) / len(experiment_ids) * 100 if experiment_ids else 0
    deployment_rate = deployed_count / len(deployment_decisions) * 100 if deployment_decisions else 0

    print(f"\n✅ 成功率:")
    print(f"   🧪 実験作成成功率: {experiment_success_rate:.1f}%")
    print(f"   📈 分析成功率: {analysis_success_rate:.1f}%")
    print(f"   🚀 デプロイメント実行率: {deployment_rate:.1f}%")

    # 推奨事項の要約
    print(f"\n🎯 推奨事項サマリー:")
    deploy_recommended_count = sum(1 for d in deployment_decisions if d.deploy_recommended)
    print(f"   ✅ デプロイ推奨: {deploy_recommended_count} / {len(deployment_decisions)}")

    if deployment_decisions:
        avg_confidence = np.mean([d.confidence_level for d in deployment_decisions])
        print(f"   📊 平均信頼度: {avg_confidence:.3f}")

        strategy_counts = {}
        for d in deployment_decisions:
            strategy = d.deployment_strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        print(f"   🚀 推奨戦略分布:")
        for strategy, count in strategy_counts.items():
            print(f"      {strategy}: {count} 回")

    print("\n" + "=" * 80)
    print("🎉 ML A/Bテスト・デプロイメントプラットフォーム デモ完了!")
    print("   Issue #733実装により、データ駆動型MLモデル改善が実現されました。")
    print("=" * 80)


async def main():
    """メインエントリーポイント"""
    try:
        await run_comprehensive_ab_test_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  ユーザーによってデモが中断されました")
    except Exception as e:
        print(f"\n\n❌ デモ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())