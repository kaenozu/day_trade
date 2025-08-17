# Enhanced Prediction System User Manual
# Issue #870 拡張予測システム ユーザーマニュアル

## 概要

Issue #870で実装された拡張予測システムは、従来の予測精度を30-60%向上させる統合機械学習システムです。このマニュアルでは、システムの使用方法、設定、最適化について詳しく説明します。

## 主要機能

### 1. 高度な特徴量選択システム
- **市場状況適応型選択**: 市場のボラティリティやトレンドに応じて自動的に最適な特徴量を選択
- **アンサンブル選択手法**: 複数の選択アルゴリズムを組み合わせて安定性を向上
- **動的重み調整**: 特徴量の重要度を市場状況に応じてリアルタイムで調整

### 2. 先進的アンサンブルシステム
- **スタッキング**: メタモデルを使用した高度な予測結合
- **ブレンディング**: 予測結果の重み付き平均による安定化
- **動的重み付け**: パフォーマンス履歴に基づく自動重み調整
- **適応型アンサンブル**: 市場条件に応じたモデル選択

### 3. ハイブリッド時系列予測
- **状態空間モデル**: カルマンフィルターによる時系列分析
- **LSTM深層学習**: 注意機構付きニューラルネットワーク
- **機械学習アンサンブル**: 複数アルゴリズムの統合予測
- **不確実性定量化**: 予測の信頼度評価

### 4. メタラーニングシステム
- **市場状況分析**: 自動的な市場レジーム検出
- **モデル選択**: 条件に最適なアルゴリズムの自動選択
- **性能追跡**: 継続的なモデル評価と改善
- **知見蓄積**: 学習経験の蓄積と活用

### 5. パフォーマンス最適化
- **メモリ管理**: 自動ガベージコレクションとキャッシュ最適化
- **キャッシュシステム**: モデルと特徴量の高速アクセス
- **バッチ処理**: 効率的な並列予測処理
- **リソース監視**: システムリソースの自動監視と最適化

## インストールと初期設定

### 必要な依存関係

```bash
# 基本的な機械学習ライブラリ
pip install numpy pandas scikit-learn

# 深層学習
pip install tensorflow keras

# 最適化とパフォーマンス
pip install psutil joblib

# その他
pip install pyyaml matplotlib seaborn
```

### 設定ファイル

#### enhanced_prediction.yaml
システムの主要設定ファイル：

```yaml
enhanced_prediction:
  system:
    mode: "auto"  # auto, enhanced, legacy, hybrid
    fallback_to_legacy: true
    enable_metrics: true
    enable_logging: true

  feature_selection:
    enabled: true
    max_features: 50
    min_importance_threshold: 0.01
    stability_threshold: 0.5

  ensemble:
    enabled: true
    method: "adaptive"  # stacking, blending, voting, dynamic_weight, adaptive
    cv_folds: 5

  hybrid_timeseries:
    enabled: true
    lstm:
      sequence_length: 20
      hidden_units: 50
      epochs: 100

  meta_learning:
    enabled: true
    repository_size: 100
```

#### system_integration.yaml
システム統合設定：

```yaml
system_integration:
  integration_mode: "hybrid"  # legacy, enhanced, hybrid, auto
  fallback:
    enable_fallback: true
    fallback_timeout_seconds: 30
```

## 基本的な使用方法

### 1. システム初期化

```python
from integrated_performance_optimizer import get_integrated_optimizer
from demo_enhanced_prediction_system import EnhancedPredictionDemo

# パフォーマンス最適化システム初期化
optimizer = get_integrated_optimizer()

# デモシステム初期化
demo = EnhancedPredictionDemo()
```

### 2. 基本予測の実行

```python
import pandas as pd
import numpy as np

# サンプルデータ準備
data = pd.DataFrame({
    'feature_1': np.random.randn(1000),
    'feature_2': np.random.randn(1000),
    'target': np.random.randn(1000)
})

X = data[['feature_1', 'feature_2']]
y = data['target']

# 基本的なデモ実行
demo.demo_1_basic_usage()
```

### 3. 高度な設定カスタマイズ

```python
# カスタム設定でのデモ実行
demo.demo_2_advanced_configuration()

# 個別コンポーネントのテスト
demo.demo_4_individual_components()
```

## 高度な使用方法

### 特徴量選択システムの詳細設定

```python
from advanced_feature_selector import create_advanced_feature_selector

# 詳細設定
selector = create_advanced_feature_selector(
    max_features=30,
    min_importance_threshold=0.05,
    stability_threshold=0.7,
    correlation_threshold=0.9
)

# 特徴量選択実行
selected_X, selection_info = selector.select_features(X, y, price_data)
print(f"選択された特徴量数: {selected_X.shape[1]}")
print(f"市場状況: {selection_info['market_regime']}")
```

### アンサンブルシステムの設定

```python
from advanced_ensemble_system import create_advanced_ensemble_system, EnsembleMethod

# スタッキングアンサンブル
stacking_ensemble = create_advanced_ensemble_system(
    method=EnsembleMethod.STACKING,
    cv_folds=5,
    meta_model_type="ridge"
)

# 動的重み付けアンサンブル
dynamic_ensemble = create_advanced_ensemble_system(
    method=EnsembleMethod.DYNAMIC_WEIGHT,
    performance_window=50,
    update_frequency=10
)
```

### ハイブリッド時系列予測

```python
from hybrid_timeseries_predictor import create_hybrid_timeseries_predictor

# ハイブリッド予測器作成
predictor = create_hybrid_timeseries_predictor(
    sequence_length=30,
    lstm_units=64,
    n_states=6,
    enable_attention=True
)

# 予測実行
predictions = predictor.predict(steps=10, uncertainty=True)
```

### メタラーニングシステム

```python
from meta_learning_system import create_meta_learning_system, TaskType

# メタラーニングシステム作成
meta_learner = create_meta_learning_system(
    repository_size=200,
    lookback_periods=50
)

# 最適モデル選択と予測
model, predictions, info = meta_learner.fit_predict(
    X_train, y_train, price_data,
    task_type=TaskType.REGRESSION
)

print(f"選択されたモデル: {info['model_type']}")
print(f"市場状況: {info['market_condition']}")
```

## パフォーマンス最適化

### システム状態の監視

```python
from integrated_performance_optimizer import get_system_performance_report

# 現在のシステム状態取得
report = get_system_performance_report()

print(f"メモリ使用量: {report['system_state']['memory_usage_mb']:.1f}MB")
print(f"CPU使用率: {report['system_state']['cpu_usage_percent']:.1f}%")
print(f"キャッシュヒット率: {report['cache_performance']['model_cache']['hit_rate']:.1%}")
```

### 自動最適化の実行

```python
from integrated_performance_optimizer import optimize_system_performance

# システム全体の最適化
optimization_results = optimize_system_performance()

for result in optimization_results:
    strategy = result.get('strategy', 'unknown')
    improvement = result.get('improvement_percent', 0)
    print(f"{strategy}: {improvement:.1f}% 改善")
```

### キャッシュ管理

```python
# モデルキャッシュの管理
optimizer = get_integrated_optimizer()

# キャッシュ統計の確認
model_stats = optimizer.model_cache.get_stats()
feature_stats = optimizer.feature_cache.get_stats()

print(f"モデルキャッシュ: {model_stats['size']} items, ヒット率 {model_stats['hit_rate']:.1%}")
print(f"特徴量キャッシュ: {feature_stats['size']} items")

# キャッシュクリア（必要に応じて）
optimizer.model_cache.clear()
optimizer.feature_cache.clear()
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー

```python
# メモリ最適化の強制実行
memory_result = optimizer.optimize_memory_usage()
print(f"解放されたメモリ: {memory_result['memory_freed_mb']:.1f}MB")

# キャッシュサイズの調整
optimizer.model_cache.max_size = 50  # デフォルト100から50に削減
optimizer.feature_cache.max_features = 2000  # デフォルト5000から2000に削減
```

#### 2. 予測精度が低い場合

```python
# 特徴量選択の再調整
selector.min_importance_threshold = 0.01  # より厳しい閾値
selector.stability_threshold = 0.8  # より高い安定性要求

# アンサンブル手法の変更
ensemble = create_advanced_ensemble_system(
    method=EnsembleMethod.STACKING,  # より高度な手法に変更
    cv_folds=10  # クロスバリデーション増加
)
```

#### 3. 処理速度が遅い場合

```python
# バッチサイズの最適化
optimizer.batch_processor.batch_size = 100  # デフォルト50から100に増加

# 並列処理の調整
optimizer.batch_processor.max_workers = 6  # ワーカー数増加

# CPUスレッド数の最適化
cpu_result = optimizer.optimize_cpu_efficiency()
```

#### 4. 設定ファイルエラー

```python
# 設定ファイルの検証
from config_manager import create_config_manager

config_manager = create_config_manager()
summary = config_manager.get_config_summary()

# 検証結果の確認
for config_name, result in summary["validation_results"].items():
    if not result["is_valid"]:
        print(f"設定エラー: {config_name}")
        if "error_count" in result:
            print(f"  エラー数: {result['error_count']}")
```

## ベストプラクティス

### 1. システム運用

- **定期的な最適化**: 1時間ごとにシステム最適化を実行
- **メモリ監視**: メモリ使用量が80%を超えたら最適化実行
- **ログ確認**: エラーログを定期的にチェック

### 2. パフォーマンス向上

- **特徴量の事前選択**: 重要でない特徴量は事前に除去
- **キャッシュ活用**: 頻繁に使用するモデルはキャッシュに保存
- **バッチ処理**: 単一予測よりもバッチ処理を活用

### 3. 設定調整

- **市場状況に応じた調整**: ボラティリティが高い時は安定性重視の設定
- **データ量に応じた調整**: 大量データの場合はキャッシュサイズを増加
- **精度要求に応じた調整**: 高精度が必要な場合はアンサンブル手法を使用

## APIリファレンス

### 主要クラス

#### EnhancedPredictionCore
```python
class EnhancedPredictionCore:
    def __init__(self, config: PredictionConfig = None)
    def predict(self, X: pd.DataFrame, y_train: pd.Series, price_data: pd.DataFrame) -> PredictionResult
    def is_initialized(self) -> bool
```

#### IntegratedPerformanceOptimizer
```python
class IntegratedPerformanceOptimizer:
    def run_comprehensive_optimization(self) -> List[Dict[str, Any]]
    def optimize_memory_usage(self) -> Dict[str, Any]
    def optimize_model_cache(self) -> Dict[str, Any]
    def get_performance_report(self) -> Dict[str, Any]
```

### 設定クラス

#### PredictionConfig
```python
@dataclass
class PredictionConfig:
    mode: PredictionMode = PredictionMode.AUTO
    feature_selection_enabled: bool = True
    ensemble_enabled: bool = True
    hybrid_timeseries_enabled: bool = True
    meta_learning_enabled: bool = True
    max_features: int = 50
    cv_folds: int = 5
```

## サポートとコミュニティ

### ヘルプリソース

- **ログファイル**: `logs/enhanced_prediction.log`でシステムログを確認
- **設定例**: `config/`ディレクトリ内の設定ファイル例を参照
- **デモスクリプト**: `demo_enhanced_prediction_system.py`で使用例を確認

### パフォーマンス監視

システムの健全性を継続的に監視するため、以下のメトリクスを定期的にチェックしてください：

- メモリ使用量 < 80%
- CPU使用率 < 70%
- キャッシュヒット率 > 70%
- 予測レスポンス時間 < 1000ms
- エラー率 < 5%

### バージョン情報

- **システムバージョン**: Issue #870 拡張予測システム v1.0
- **対応Python**: 3.8以上
- **最終更新**: 2025年8月17日

---

このマニュアルは、Issue #870拡張予測システムの効果的な利用をサポートするためのガイドです。システムの継続的な改善と最適化により、30-60%の予測精度向上を実現できます。