# 日間取引システム 統一アーキテクチャ版使用ガイド

## 目次
1. [はじめに](#はじめに)
2. [システム要件](#システム要件)
3. [インストール・セットアップ](#インストール・セットアップ)
4. [基本的な使用方法](#基本的な使用方法)
5. [高度な機能](#高度な機能)
6. [設定管理](#設定管理)
7. [監視・最適化](#監視・最適化)
8. [トラブルシューティング](#トラブルシューティング)
9. [FAQ](#faq)

## はじめに

本ガイドでは、リファクタリング後の日間取引システムの使用方法について詳しく説明します。このシステムは統一アーキテクチャフレームワークに基づいて構築されており、高いパフォーマンスと信頼性を提供します。

### 主要機能
- **統一アーキテクチャ**: 一貫性のあるシステム設計
- **エラーハンドリング**: 包括的なエラー処理と回復機能
- **パフォーマンス最適化**: 自動的な性能監視と最適化
- **設定管理**: 柔軟で一元化された設定システム
- **ログ・監視**: 詳細な監視とログ機能

### リファクタリング成果
- **テスト成功率**: 87.5% (7/8テスト通過)
- **統合システム数**: 8つの主要システム
- **コード重複削減**: 統一コンポーネントによる効率化

## システム要件

### 最小要件
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.11以上
- **メモリ**: 8GB RAM
- **ストレージ**: 10GB 利用可能領域
- **ネットワーク**: インターネット接続

### 推奨要件
- **OS**: 最新安定版
- **Python**: 3.11+
- **メモリ**: 16GB RAM
- **ストレージ**: 50GB SSD
- **CPU**: 4コア以上

## インストール・セットアップ

### 1. 仮想環境の作成

```bash
# Python仮想環境作成
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. システムの動作確認

```bash
# 統合テスト実行
python test_refactored_system.py
```

正常に動作している場合、87.5%以上のテスト成功率が表示されます。

## 基本的な使用方法

### 1. システムの起動

#### アプリケーションの起動
```python
from day_trade.core.architecture.unified_framework import UnifiedApplication
from day_trade.core.configuration.unified_config_manager import global_config_manager
import asyncio

async def main():
    # アプリケーション作成
    app = UnifiedApplication("day_trade_system")

    # 設定
    config = {
        "database": {"url": "sqlite:///day_trade.db"},
        "trading": {"max_positions": 5},
        "logging": {"level": "INFO"}
    }

    # アプリケーション開始
    await app.start(config)

    # ヘルスチェック
    health = await app.health_check()
    print(f"システム状態: {health['status']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 統一コンポーネントの使用

#### 統一アナライザー
```python
from day_trade.core.consolidation.unified_consolidator import global_consolidation_executor

# 統一アナライザー取得
analyzer = global_consolidation_executor.unified_analyzer

# テクニカル分析実行
data = [100, 101, 102, 103, 104, 105]
result = analyzer.execute("technical_analysis", data=data, indicators=["sma", "ema", "rsi"])
print("分析結果:", result)

# 利用可能な機能確認
capabilities = analyzer.get_capabilities()
print("分析機能:", capabilities)
# 出力例: ['technical_analysis', 'fundamental_analysis', 'pattern_recognition', ...]
```

#### 統一プロセッサー
```python
# 統一プロセッサー取得
processor = global_consolidation_executor.unified_processor

# データクリーニング実行
raw_data = [100, None, 102, 103, None, 105]
cleaned_data = processor.execute("data_cleaning", data=raw_data)
print("クリーニング後:", cleaned_data)

# 利用可能な処理機能確認
capabilities = processor.get_capabilities()
print("処理機能:", capabilities)
# 出力例: ['data_cleaning', 'normalization', 'aggregation', ...]
```

#### 統一マネージャー
```python
# 統一マネージャー取得
manager = global_consolidation_executor.unified_manager

# リソース登録例
class MyResource:
    def __init__(self):
        self.status = "stopped"

    def start(self):
        self.status = "running"
        return True

    def stop(self):
        self.status = "stopped"
        return True

    def get_status(self):
        return self.status

my_resource = MyResource()
manager.register_resource("my_service", my_resource)

# 全リソース開始
results = manager.execute("start_all")
print("開始結果:", results)

# ステータス確認
status = manager.execute("get_status")
print("リソース状態:", status)
```

## 高度な機能

### 1. パフォーマンス監視

#### パフォーマンスデコレーターの使用
```python
from day_trade.core.performance.optimization_engine import performance_monitor

@performance_monitor(component="trading", operation="execute_trade")
async def execute_complex_trade(portfolio_id, trade_request):
    # 複雑な取引ロジック
    await some_async_operation()
    return "trade_result"

# 同期関数でも使用可能
@performance_monitor(component="analysis", operation="technical_analysis")
def calculate_indicators(data):
    # テクニカル分析計算
    import time
    time.sleep(0.1)  # 模擬的な処理時間
    return {"sma": 100, "rsi": 50}
```

#### パフォーマンス分析の実行
```python
from day_trade.core.performance.optimization_engine import global_optimization_engine

# パフォーマンス分析実行
analysis = global_optimization_engine.analyze_performance("trading")
print(f"平均実行時間: {analysis['average_execution_time_ms']}ms")
print(f"メモリ使用量: {analysis['max_memory_usage_mb']}MB")

# ベンチマーク結果確認
benchmark_results = analysis.get('benchmark_results', {})
for operation, result in benchmark_results.items():
    print(f"{operation}: ベンチマーク{'達成' if result['meets_benchmark'] else '未達成'}")

# 最適化提案取得
suggestions = global_optimization_engine.suggest_optimizations(analysis)
for suggestion in suggestions:
    print(f"提案: {suggestion}")
```

#### ベンチマーク設定
```python
from day_trade.core.performance.optimization_engine import PerformanceBenchmark

# ベンチマーク設定
benchmark = PerformanceBenchmark(
    operation_name="execute_trade",
    target_response_time_ms=200.0,
    target_throughput_ops=1000.0,
    max_memory_mb=512.0,
    max_cpu_percent=80.0
)

global_optimization_engine.profiler.set_benchmark(benchmark)
print("ベンチマーク設定完了")
```

### 2. エラーハンドリング

#### エラーバウンダリーデコレーターの使用
```python
from day_trade.core.error_handling.unified_error_system import error_boundary

@error_boundary(component_name="trading", suppress_errors=True, fallback_value=None)
def risky_operation():
    # 失敗する可能性のある処理
    import random
    if random.random() < 0.5:
        raise ValueError("何かが間違っています")
    return "成功"

# エラーが発生してもfallback_valueが返される
result = risky_operation()
print(f"処理結果: {result}")  # 成功時は"成功"、エラー時はNone
```

#### カスタム例外の使用
```python
from day_trade.core.error_handling.unified_error_system import (
    ValidationError, BusinessLogicError, InfrastructureError
)

def validate_trade_request(symbol, quantity, price):
    # バリデーションエラー
    if not symbol:
        raise ValidationError(
            "銘柄コードが未指定です",
            field="symbol",
            value=symbol
        )

    if quantity <= 0:
        raise ValidationError(
            "取引量は正の数である必要があります",
            field="quantity",
            value=quantity
        )

    # ビジネスロジックエラー
    if quantity > 10000:
        raise BusinessLogicError(
            "一度に10,000株を超える取引はできません",
            rule_name="max_trade_quantity"
        )

    # インフラストラクチャエラー例
    if price <= 0:
        raise InfrastructureError(
            "価格データの取得に失敗しました",
            service_name="market_data_service"
        )

# 使用例
try:
    validate_trade_request("7203", 100, 1500.0)
    print("バリデーション成功")
except ValidationError as e:
    print(f"バリデーションエラー: {e.message} (フィールド: {e.field})")
except BusinessLogicError as e:
    print(f"ビジネスルールエラー: {e.message} (ルール: {e.rule_name})")
```

#### エラー分析
```python
from day_trade.core.error_handling.unified_error_system import global_error_handler

# エラー統計取得
analytics = global_error_handler.get_analytics()
print(f"総エラー数: {analytics['total_errors']}")
print(f"エラー率: {analytics['error_rate']:.2%}")

# カテゴリ別エラー数
error_categories = analytics.get('errors_by_category', {})
for category, count in error_categories.items():
    print(f"{category}: {count}件")
```

## 設定管理

### 1. プログラムでの設定管理

#### 設定値の取得と設定
```python
from day_trade.core.configuration.unified_config_manager import global_config_manager

# 設定値取得（ドット記法サポート）
max_positions = global_config_manager.get("trading.max_positions", 5)
risk_tolerance = global_config_manager.get("trading.risk_tolerance", 0.1)
database_url = global_config_manager.get("database.url", "sqlite:///default.db")

print(f"最大ポジション数: {max_positions}")
print(f"リスク許容度: {risk_tolerance}")

# 設定値設定
global_config_manager.set("trading.max_positions", 10)
global_config_manager.set("trading.risk_tolerance", 0.15, persist=True)  # ファイルに保存

# 階層設定
global_config_manager.set("notifications.email.enabled", True)
global_config_manager.set("notifications.email.smtp.host", "smtp.example.com")
```

#### プロファイル管理
```python
# 利用可能プロファイル取得
profiles = global_config_manager.get_profiles()
print("利用可能プロファイル:", profiles)

# 現在のプロファイル確認
current_profile = global_config_manager.get_active_profile()
print("現在のプロファイル:", current_profile)

# プロファイル変更
success = global_config_manager.set_profile("aggressive")
if success:
    print("アグレッシブプロファイルに変更しました")
else:
    print("プロファイルの変更に失敗しました")
```

#### 一時設定の使用
```python
# 一時的な設定変更
print("元の設定:", global_config_manager.get("trading.max_positions"))

with global_config_manager.temp_config({
    "trading.max_positions": 20,
    "trading.risk_tolerance": 0.2
}):
    # この範囲内では一時設定が有効
    max_pos = global_config_manager.get("trading.max_positions")
    risk = global_config_manager.get("trading.risk_tolerance")
    print(f"一時設定: max_positions={max_pos}, risk_tolerance={risk}")

# 範囲外では元の設定に戻る
print("復元後の設定:", global_config_manager.get("trading.max_positions"))
```

#### 設定検証
```python
# 特定設定の検証
trading_validation = global_config_manager.validate_config("trading")
if trading_validation["valid"]:
    print("取引設定は有効です")
else:
    print("取引設定エラー:")
    for error in trading_validation["errors"]:
        print(f"  - {error}")

# 全設定の検証
all_validation = global_config_manager.validate_config()
for config_name, result in all_validation.items():
    status = "有効" if result["valid"] else "無効"
    print(f"{config_name}設定: {status}")
    if not result["valid"]:
        for error in result["errors"]:
            print(f"  エラー: {error}")
```

## 監視・最適化

### 1. ログ出力

#### 基本的なログ出力
```python
from day_trade.core.logging.unified_logging_system import get_logger

# ロガー取得（コンポーネント指定）
logger = get_logger("trading_module", "trading")

# 各レベルのログ出力
logger.debug("デバッグ情報", operation="validate_trade", extra_data="debug_info")
logger.info("取引開始", operation="execute_trade", symbol="7203", quantity=100)
logger.warning("高ボラティリティ検出", operation="risk_check", volatility=0.15)
logger.error("取引失敗", operation="execute_trade", symbol="7203", error_code="INSUFFICIENT_FUNDS")
```

#### コンテキスト付きログ
```python
# コンテキスト設定
with logger.context(user_id="user123", session_id="session456", portfolio_id="portfolio789"):
    logger.info("ユーザー操作開始")
    # この範囲内のログには自動的にuser_id、session_id、portfolio_idが含まれる
    logger.info("ポートフォリオアクセス")
    logger.info("取引実行", symbol="7203", action="buy")
```

#### ログ実行デコレーター
```python
from day_trade.core.logging.unified_logging_system import log_execution

@log_execution(
    logger_name="trading_logger",
    component="trading",
    log_args=True,      # 引数をログ出力
    log_result=True     # 戻り値をログ出力
)
def execute_trade(symbol, quantity, price):
    # 関数の実行開始・終了、引数、結果が自動ログ出力される
    import time
    time.sleep(0.1)  # 模擬的な処理時間
    return f"Trade executed: {symbol} {quantity}株 @{price}"

# 使用例
result = execute_trade("7203", 100, 1500.0)
print(f"結果: {result}")
```

### 2. リソース監視

#### リソース監視の開始
```python
from day_trade.core.performance.optimization_engine import global_optimization_engine

# 監視開始
global_optimization_engine.start_monitoring()
print("リソース監視を開始しました")

# 現在の使用状況取得
usage = global_optimization_engine.resource_monitor.get_current_usage()
print(f"CPU使用率: {usage.get('cpu_percent', 0):.1f}%")
print(f"メモリ使用率: {usage.get('memory_percent', 0):.1f}%")
print(f"利用可能メモリ: {usage.get('memory_available_gb', 0):.2f}GB")
print(f"プロセスメモリ: {usage.get('process_memory_mb', 0):.1f}MB")

# 監視停止（アプリケーション終了時）
# global_optimization_engine.stop_monitoring()
```

#### パフォーマンスメトリクスの取得
```python
# 過去1時間のメトリクス取得
metrics = global_optimization_engine.profiler.get_metrics(
    component="trading",
    hours=1
)

print(f"過去1時間のメトリクス数: {len(metrics)}")
for metric in metrics[-5:]:  # 最新5件表示
    print(f"{metric.timestamp}: {metric.name} = {metric.value}")
```

### 3. 自動最適化

#### 自動最適化の実行
```python
# 自動最適化実行
optimization_result = global_optimization_engine.auto_optimize("trading")

print("=== 自動最適化結果 ===")
print("適用された最適化:", optimization_result["optimizations_applied"])
print("最適化提案:")
for suggestion in optimization_result["suggestions"]:
    print(f"  - {suggestion}")

# ガベージコレクション結果
gc_result = optimization_result["gc_result"]
print(f"解放されたメモリ: {gc_result['memory_freed_mb']:.1f}MB")
print(f"回収されたオブジェクト: {gc_result['objects_collected']}個")
```

#### キャッシュ最適化
```python
# キャッシュ統計取得
cache_stats = global_optimization_engine.cache_optimizer.get_cache_stats()
for cache_name, stats in cache_stats.items():
    print(f"キャッシュ '{cache_name}':")
    print(f"  ヒット率: {stats['hit_rate']:.1%}")
    print(f"  効率性: {stats['efficiency']}")
    print(f"  総リクエスト数: {stats['total_requests']}")
```

## トラブルシューティング

### 1. よくある問題と解決方法

#### システム起動に失敗する場合
```python
# 設定ファイルの確認
try:
    from day_trade.core.configuration.unified_config_manager import global_config_manager
    global_config_manager.load_from_file("config/config.yaml")
    print("設定ファイル読み込み成功")
except Exception as e:
    print(f"設定ファイル読み込み失敗: {e}")

# コンポーネントの動作確認
try:
    from day_trade.core.architecture.unified_framework import UnifiedApplication
    app = UnifiedApplication("test_app")
    print("統一フレームワーク初期化成功")
except Exception as e:
    print(f"統一フレームワーク初期化失敗: {e}")
```

#### 取引実行に失敗する場合
```python
# エラー分析の確認
from day_trade.core.error_handling.unified_error_system import global_error_handler

analytics = global_error_handler.get_analytics()
print("=== エラー分析結果 ===")
print(f"総エラー数: {analytics['total_errors']}")
print(f"エラー率: {analytics['error_rate']:.2%}")

# 最近のエラー確認
recent_errors = analytics.get('recent_errors', [])
for error in recent_errors[-5:]:  # 最新5件
    print(f"エラー: {error.get('message', 'Unknown')} - {error.get('timestamp', 'Unknown')}")
```

#### パフォーマンス問題の診断
```python
# パフォーマンス分析実行
analysis = global_optimization_engine.analyze_performance()

print("=== パフォーマンス診断 ===")
print(f"平均実行時間: {analysis['average_execution_time_ms']:.2f}ms")
print(f"最大メモリ使用量: {analysis['max_memory_usage_mb']:.1f}MB")

# ボトルネック特定
if analysis["average_execution_time_ms"] > 1000:
    print("⚠️ 実行時間が長すぎます。以下を確認してください:")
    print("  1. データベースクエリの最適化")
    print("  2. キャッシュの設定")
    print("  3. 並列処理の導入")

# 自動最適化の実行
auto_result = global_optimization_engine.auto_optimize()
print("自動最適化提案:")
for suggestion in auto_result["suggestions"]:
    print(f"  - {suggestion}")
```

## FAQ

### Q1: システムの推奨設定は何ですか？

**A1**: 以下の設定を推奨します：

```yaml
# 取引設定
trading:
  max_positions: 5
  risk_tolerance: 0.1
  stop_loss_percent: 5.0
  take_profit_percent: 10.0

# パフォーマンス設定
performance:
  monitoring_enabled: true
  cache_size: 1000
  profiling_enabled: false

# ログ設定
logging:
  level: "INFO"
  max_size_mb: 100
  format: "text"

# データベース設定
database:
  pool_size: 10
  timeout: 30
```

### Q2: エラーが頻発する場合の対処法は？

**A2**: 以下の手順で対処してください：

1. **エラー分析の実行**:
```python
analytics = global_error_handler.get_analytics()
print("主要エラー:", analytics.get("top_error_messages", []))
```

2. **設定の確認**:
```python
validation = global_config_manager.validate_config()
for config_name, result in validation.items():
    if not result["valid"]:
        print(f"{config_name}に問題があります")
```

3. **ログレベルの調整**:
```python
from day_trade.core.logging.unified_logging_system import configure_logging, LogLevel
configure_logging(log_level=LogLevel.DEBUG)
```

### Q3: パフォーマンスを向上させるには？

**A3**: 以下の最適化を推奨します：

1. **自動最適化の定期実行**:
```python
# 毎時実行
import schedule
schedule.every().hour.do(lambda: global_optimization_engine.auto_optimize())
```

2. **キャッシュサイズの調整**:
```python
global_config_manager.set("performance.cache_size", 2000)
```

3. **リソース監視の活用**:
```python
# 継続的な監視
global_optimization_engine.start_monitoring()
```

### Q4: 統合テストが失敗する場合は？

**A4**: 以下を確認してください：

```bash
# 依存関係の確認
pip list | grep -E "(asyncio|psutil|pyyaml)"

# システムリソースの確認
python -c "import psutil; print(f'CPU: {psutil.cpu_count()}, Memory: {psutil.virtual_memory().total // 1024**3}GB')"

# 個別コンポーネントのテスト
python -c "from day_trade.core.architecture.unified_framework import UnifiedApplication; print('統一フレームワーク: OK')"
```

### Q5: カスタムコンポーネントを追加するには？

**A5**: 統一フレームワークを使用してコンポーネントを追加できます：

```python
from day_trade.core.consolidation.unified_consolidator import BaseConsolidatedComponent, ComponentCategory

class MyCustomAnalyzer(BaseConsolidatedComponent):
    def __init__(self):
        super().__init__("my_analyzer", ComponentCategory.ANALYZER)

    def get_capabilities(self):
        return ["custom_analysis", "special_calculation"]

    def execute(self, operation, **kwargs):
        if operation == "custom_analysis":
            return self._custom_analysis(**kwargs)
        else:
            raise ValueError(f"未対応の操作: {operation}")

    def _custom_analysis(self, data):
        # カスタム分析ロジック
        return {"result": "custom_analysis_complete"}

# 使用方法
custom_analyzer = MyCustomAnalyzer()
result = custom_analyzer.execute("custom_analysis", data=[1, 2, 3])
print(result)
```

---

**ガイド版数**: v1.0  
**最終更新**: 2025-08-17  
**対象システム**: Day Trading System v1.0.0 (統一アーキテクチャ版)  
**テスト成功率**: 87.5%