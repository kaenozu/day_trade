# 統合最適化システム ユーザーガイド

## 概要

Day Trade統合最適化システムは、Strategy Patternを活用した高性能・高拡張性のトレーディングシステムです。標準実装と最適化実装を統一インターフェースで提供し、設定ベースで動作レベルを選択できます。

## 主要機能

### 🎯 コア機能

1. **Strategy Pattern統合アーキテクチャ**
   - 標準実装・最適化実装・適応的選択の統一インターフェース
   - コンポーネント別最適化レベル設定
   - 自動フォールバック機能

2. **統合コンポーネント**
   - テクニカル指標分析（ML強化・Numba高速化）
   - 特徴量エンジニアリング（並列処理・チャンク処理）
   - データベース管理（クエリキャッシュ・バルク最適化）
   - 機械学習モデル（並列訓練・予測キャッシュ）
   - マルチタイムフレーム分析（並列・非同期処理）

3. **包括的監視・管理機能**
   - リアルタイムパフォーマンス監視
   - 適応的リソース管理
   - CLI管理ツール

## インストールと初期設定

### 必要要件

```bash
# Python 3.11以上
python --version

# 必須パッケージ
pip install pandas numpy scikit-learn sqlalchemy psutil

# オプショナル最適化パッケージ
pip install numba xgboost lightgbm joblib
```

### 初期セットアップ

1. **設定ファイル作成**
```bash
python -m src.day_trade.core.optimization_cli config template
```

2. **システム情報確認**
```bash
python -m src.day_trade.core.optimization_cli system
```

3. **コンポーネント一覧確認**
```bash
python -m src.day_trade.core.optimization_cli component list
```

## 基本的な使用方法

### 1. 設定ベース初期化

```python
from src.day_trade.core import (
    OptimizationConfig,
    OptimizationLevel,
    get_optimized_implementation
)

# 基本設定
config = OptimizationConfig(
    level=OptimizationLevel.OPTIMIZED,
    auto_fallback=True,
    performance_monitoring=True,
    cache_enabled=True
)

# コンポーネント取得
strategy = get_optimized_implementation("technical_indicators", config)
```

### 2. テクニカル指標分析

```python
from src.day_trade.analysis.technical_indicators_unified import TechnicalIndicatorsManager
import pandas as pd

# データ準備
data = pd.read_csv("stock_data.csv", index_col='Date', parse_dates=True)

# マネージャー初期化
manager = TechnicalIndicatorsManager(config)

# 指標計算
indicators = ["sma", "bollinger_bands", "rsi", "macd", "ichimoku"]
results = manager.calculate_indicators(data, indicators, period=20)

# 結果確認
for indicator, result in results.items():
    print(f"{indicator}: {result.calculation_time:.3f}秒")
    print(f"戦略: {result.strategy_used}")
```

### 3. 特徴量エンジニアリング

```python
from src.day_trade.analysis.feature_engineering_unified import (
    FeatureEngineeringManager,
    FeatureConfig
)

# 特徴量設定
feature_config = FeatureConfig(
    lookback_periods=[5, 10, 20, 50],
    volatility_windows=[10, 20, 50],
    momentum_periods=[5, 10, 20],
    enable_parallel=True,
    max_workers=4
)

# マネージャー初期化・実行
manager = FeatureEngineeringManager(config)
result = manager.generate_features(data, feature_config)

print(f"生成特徴量: {len(result.feature_names)}個")
print(f"処理時間: {result.generation_time:.3f}秒")
print(f"戦略: {result.strategy_used}")
```

### 4. 機械学習モデル

```python
from src.day_trade.analysis.ml_models_unified import (
    MLModelsManager,
    ModelConfig
)

# モデル設定
model_config = ModelConfig(
    model_type="xgboost",  # random_forest, gradient_boosting, xgboost, lightgbm
    n_estimators=100,
    max_depth=6,
    enable_parallel=True
)

# マネージャー初期化
ml_manager = MLModelsManager(config)

# モデル訓練
training_result = ml_manager.train_model(X_train, y_train, model_config)
print(f"訓練スコア: {training_result.training_score:.4f}")
print(f"検証スコア: {training_result.validation_score:.4f}")

# 予測実行
prediction = ml_manager.predict(X_test)
print(f"予測値: {prediction.prediction}")
print(f"信頼度: {prediction.confidence:.3f}")
```

### 5. マルチタイムフレーム分析

```python
from src.day_trade.analysis.multi_timeframe_analysis_unified import (
    MultiTimeframeAnalysisManager
)

# マネージャー初期化・分析実行
mtf_manager = MultiTimeframeAnalysisManager(config)
result = mtf_manager.analyze_multi_timeframe(data)

print(f"統合トレンド: {result.integrated_trend}")
print(f"信頼度: {result.confidence_score:.3f}")
print(f"一貫性: {result.trend_consistency:.3f}")

# タイムフレーム別結果
for tf_name, tf_result in result.timeframe_results.items():
    print(f"{tf_name}: {tf_result['trend']} (強度: {tf_result['trend_strength']:.3f})")
```

## 設定管理

### 設定ファイル形式

```json
{
  "level": "adaptive",
  "auto_fallback": true,
  "performance_monitoring": true,
  "cache_enabled": true,
  "parallel_processing": true,
  "batch_size": 1000,
  "timeout_seconds": 60,
  "memory_limit_mb": 1024,

  "component_specific": {
    "technical_indicators": {
      "level": "optimized",
      "enable_numba": true,
      "cache_ttl_seconds": 300
    },
    "feature_engineering": {
      "level": "optimized",
      "chunk_size": 10000,
      "max_workers": 6
    },
    "ml_models": {
      "level": "optimized",
      "enable_caching": true,
      "cache_size": 2000
    }
  },

  "system_thresholds": {
    "auto_fallback_triggers": {
      "memory_over": 90,
      "cpu_over": 90,
      "error_rate_over": 0.10
    }
  }
}
```

### 環境変数設定

```bash
# 基本設定
export DAYTRADE_OPTIMIZATION_LEVEL=optimized
export DAYTRADE_AUTO_FALLBACK=true
export DAYTRADE_PERF_MONITORING=true
export DAYTRADE_CACHE_ENABLED=true
export DAYTRADE_PARALLEL=true

# パフォーマンス設定
export DAYTRADE_BATCH_SIZE=2000
export DAYTRADE_TIMEOUT=120
export DAYTRADE_MEMORY_LIMIT=2048

# 設定ファイルパス
export DAYTRADE_CONFIG_PATH=/path/to/optimization_config.json
```

## CLI管理ツール

### 基本コマンド

```bash
# 設定表示
python -m src.day_trade.core.optimization_cli config show

# 設定テンプレート作成
python -m src.day_trade.core.optimization_cli config template --output my_config.json

# コンポーネント一覧
python -m src.day_trade.core.optimization_cli component list

# コンポーネントテスト
python -m src.day_trade.core.optimization_cli component test technical_indicators --level optimized

# ベンチマーク実行
python -m src.day_trade.core.optimization_cli benchmark

# システム情報
python -m src.day_trade.core.optimization_cli system
```

## 最適化レベル詳細

### Standard（標準）
- **特徴**: 安定性・互換性重視
- **用途**: 開発環境、初期導入時
- **パフォーマンス**: 基本レベル
- **メモリ使用量**: 標準

### Optimized（最適化）  
- **特徴**: 高性能・高スループット
- **用途**: 本番環境、大容量処理
- **パフォーマンス**: 100倍高速化（並列処理・キャッシュ・Numba）
- **メモリ使用量**: 98%削減（最適化キャッシュ）

### Adaptive（適応的）
- **特徴**: 動的レベル選択・自動最適化
- **用途**: 環境変動が大きいシステム
- **パフォーマンス**: 状況に応じて自動調整
- **制御**: CPU・メモリ使用率に基づく自動切り替え

### Debug（デバッグ）
- **特徴**: 詳細ログ・診断機能
- **用途**: 問題調査・パフォーマンス分析
- **パフォーマンス**: やや低下（詳細監視のため）
- **情報量**: 最大

## パフォーマンス監視

### メトリクス確認

```python
# パフォーマンス概要
manager = TechnicalIndicatorsManager(config)
metrics = manager.get_performance_summary()

print(f"実行回数: {metrics['execution_count']}")
print(f"成功率: {metrics.get('success_rate', 0):.2%}")
print(f"平均実行時間: {metrics['average_time']:.3f}秒")

# キャッシュ統計（最適化実装のみ）
if hasattr(manager.get_strategy(), 'get_cache_stats'):
    cache_stats = manager.get_strategy().get_cache_stats()
    print(f"キャッシュヒット率: {cache_stats.get('hit_rate', 0):.2%}")
```

### リアルタイム監視

```python
import psutil
import time

def monitor_system_resources():
    """システムリソース監視"""
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        print(f"CPU: {cpu_percent}%, Memory: {memory.percent}%")

        # アラート閾値チェック
        if cpu_percent > 90:
            print("⚠️  高CPU使用率検出")
        if memory.percent > 85:
            print("⚠️  高メモリ使用率検出")

        time.sleep(10)

# バックグラウンド実行
import threading
monitoring_thread = threading.Thread(target=monitor_system_resources)
monitoring_thread.daemon = True
monitoring_thread.start()
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. ImportError: 統合コンポーネントが見つからない

```bash
# Pythonパス確認
export PYTHONPATH="${PYTHONPATH}:/path/to/day_trade/src"

# または
python -c "import sys; sys.path.insert(0, 'src')"
```

#### 2. 最適化パッケージが利用できない

```bash
# Numba高速化
pip install numba

# XGBoost/LightGBM
pip install xgboost lightgbm

# システム監視
pip install psutil
```

#### 3. メモリ使用量が多い

```python
# 設定でメモリ制限
config = OptimizationConfig(
    memory_limit_mb=512,  # 512MBに制限
    cache_enabled=False,  # キャッシュ無効化
)

# またはガベージコレクション強制実行
import gc
gc.collect()
```

#### 4. パフォーマンスが期待より低い

```python
# 並列処理有効化
config = OptimizationConfig(
    level=OptimizationLevel.OPTIMIZED,
    parallel_processing=True,
    batch_size=2000  # バッチサイズ増加
)

# Numba有効化確認
from src.day_trade.analysis.feature_engineering_unified import NUMBA_AVAILABLE
print(f"Numba利用可能: {NUMBA_AVAILABLE}")
```

#### 5. 設定が反映されない

```bash
# 環境変数確認
env | grep DAYTRADE

# 設定ファイルパス確認
python -m src.day_trade.core.optimization_cli config show
```

## プロダクション環境デプロイ

### 自動セットアップ

```bash
# プロダクション環境の完全セットアップ
python deployment/production_setup.py --action setup --environment production

# デプロイメントパッケージ作成
python deployment/production_setup.py --action package

# システム検証
python deployment/production_setup.py --action validate
```

### 手動セットアップ

1. **環境変数設定**
```bash
source .env.production
```

2. **設定ファイル配置**
```bash
cp config/optimization_config.json /etc/daytrade/optimization_config.json
```

3. **サービス登録**
```bash
sudo cp deployment/daytrade.service /etc/systemd/system/
sudo systemctl enable daytrade
sudo systemctl start daytrade
```

4. **ログローテーション**
```bash
sudo cp deployment/logrotate.conf /etc/logrotate.d/daytrade
```

### 監視設定

```bash
# 監視スクリプト設定
chmod +x deployment/monitoring.sh

# Cronジョブ追加
echo "*/5 * * * * /path/to/deployment/monitoring.sh" | crontab -
```

## 統合テスト実行

```bash
# 包括的統合テスト
python test_unified_optimization_system.py

# 特定コンポーネントテスト
python -m src.day_trade.core.optimization_cli component test technical_indicators

# パフォーマンスベンチマーク
python -m src.day_trade.core.optimization_cli benchmark
```

## FAQ

### Q: どの最適化レベルを選ぶべきですか？
A:
- **開発・テスト**: Standard
- **本番・高負荷**: Optimized  
- **環境変動大**: Adaptive
- **問題調査**: Debug

### Q: メモリ使用量を削減するには？
A:
- `cache_enabled=false` でキャッシュ無効化
- `memory_limit_mb` で制限設定
- `batch_size` を小さく設定

### Q: 処理を高速化するには？
A:
- `level=OptimizationLevel.OPTIMIZED` に設定
- `parallel_processing=true` で並列処理有効化
- Numba・XGBoost・LightGBMをインストール
- `batch_size` を最適化

### Q: エラーが発生した場合の対処は？
A:
- `auto_fallback=true` で自動フォールバック有効化
- ログレベルを DEBUG に設定して詳細確認
- CLIツールでコンポーネント個別テスト実行

### Q: カスタムコンポーネントを追加するには？
A:
```python
from src.day_trade.core.optimization_strategy import optimization_strategy, OptimizationLevel

@optimization_strategy("my_component", OptimizationLevel.STANDARD)
class MyComponent(OptimizationStrategy):
    def get_strategy_name(self) -> str:
        return "カスタムコンポーネント"

    def execute(self, *args, **kwargs):
        # 実装
        pass
```

## サポート

- **ドキュメント**: `/docs` ディレクトリ内の詳細ガイド
- **CLI ヘルプ**: `python -m src.day_trade.core.optimization_cli --help`
- **システム情報**: `python -m src.day_trade.core.optimization_cli system`
- **ベンチマーク**: `python -m src.day_trade.core.optimization_cli benchmark`

統合最適化システムにより、従来のトレーディングシステムが現代的な高性能・高拡張性アーキテクチャに生まれ変わります。
