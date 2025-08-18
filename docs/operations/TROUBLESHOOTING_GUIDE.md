# デイトレードAIシステム トラブルシューティングガイド

## 🚨 緊急時クイック対応チェックリスト

### システム停止時の確認項目
- [ ] インターネット接続確認
- [ ] Pythonプロセス動作確認 (`ps aux | grep python`)
- [ ] システムリソース確認 (`top`, `free -h`)
- [ ] ディスク容量確認 (`df -h`)
- [ ] 最新エラーログ確認

### 応急処置コマンド
```bash
# システム状態確認
python production_readiness_test.py

# 緊急停止
pkill -f python

# 緊急再起動
python comprehensive_live_validation.py
```

---

## 🔍 問題診断フローチャート

```
システム問題発生
       ↓
   エラーメッセージあり？
   ↓YES          ↓NO
エラーコード参照  パフォーマンス問題？
                ↓YES        ↓NO
            リソース確認    ネットワーク問題？
                           ↓YES      ↓NO
                        接続確認   ログ詳細調査
```

---

## 🚨 エラーメッセージ別対応ガイド

### データ取得関連エラー

#### `No data found, symbol may be delisted`
**原因**: 銘柄コードが無効または市場休業
```python
# 解決方法
symbols_map = {
    "7203": "トヨタ自動車",
    "8306": "三菱UFJ銀行",
    "4751": "サイバーエージェント"
}

# 正しい銘柄コード確認
print("利用可能銘柄:")
for code, name in symbols_map.items():
    print(f"  {code}: {name}")
```

#### `All data sources failed for [symbol]`
**原因**: 全てのデータソースが失敗
```python
# 段階的解決
# 1. ネットワーク接続確認
import requests
try:
    response = requests.get("https://www.google.com", timeout=5)
    print(f"ネットワーク: OK ({response.status_code})")
except:
    print("ネットワーク: NG - インターネット接続を確認")

# 2. Yahoo Finance API確認
import yfinance as yf
try:
    ticker = yf.Ticker("7203.T")
    data = ticker.history(period="1d")
    print(f"Yahoo Finance: OK ({len(data)}レコード)")
except Exception as e:
    print(f"Yahoo Finance: NG - {e}")

# 3. 代替データ使用
from market_data_stability_system import market_data_stability_system
data = await market_data_stability_system.source_manager.fetch_with_fallback("7203", "5d")
if data is not None:
    print("フォールバック: OK")
else:
    print("フォールバック: NG - 手動データ入力が必要")
```

#### `DataFrame.fillna with 'method' is deprecated`
**原因**: Pandasバージョン互換性問題
```python
# 解決方法: コード修正
# 古い書き方
# df.fillna(method='ffill')

# 新しい書き方
df.ffill().bfill()
```

### 予測システムエラー

#### `予測実行エラー`
**診断手順**:
```python
# 1. データ確認
from real_data_provider_v2 import real_data_provider
data = await real_data_provider.get_stock_data("7203", "5d")
print(f"データ件数: {len(data) if data is not None else 0}")
if data is None or len(data) < 5:
    print("❌ データ不足 - 最低5日分のデータが必要")

# 2. 特徴量生成確認
from optimized_prediction_system import optimized_prediction_system
if data is not None:
    features = optimized_prediction_system.create_optimized_features(data)
    print(f"特徴量数: {len(features.columns)}")
    print(f"サンプル数: {len(features)}")
    if len(features) < 20:
        print("⚠️ サンプル数が少ない - 精度が低い可能性")

# 3. モデル状態確認
model_info = optimized_prediction_system.get_model_info("7203")
print(f"モデル情報: {model_info}")
```

**解決方法**:
```python
# モデル再学習
await optimized_prediction_system.train_optimized_models("7203")

# 予測再実行
prediction = await optimized_prediction_system.predict_with_optimized_models("7203")
print(f"予測結果: {prediction}")
```

#### `最適化特徴量作成完了: 80特徴量, 22サンプル`
**状況**: データは正常、サンプル数が少ない
```python
# より長期間のデータを使用
longer_data = await real_data_provider.get_stock_data("7203", "3mo")  # 3ヶ月分
if longer_data is not None and len(longer_data) > 50:
    print("✅ 十分なデータ取得")
    # 予測実行
    prediction = await optimized_prediction_system.predict_with_optimized_models("7203")
else:
    print("⚠️ 長期データも不足 - 予測精度が低い可能性")
```

### リスク管理エラー

#### `リスク計算エラー`
**診断**:
```python
# 市場データ確認
from production_risk_management_validator import production_risk_validator
risk_metrics = await production_risk_validator.risk_engine.calculate_comprehensive_risk_metrics("7203")

if risk_metrics is None:
    # 詳細診断
    data = await production_risk_validator.risk_engine._get_market_data("7203", "3mo")
    print(f"市場データ: {len(data) if data else 0}レコード")

    if data is not None and len(data) > 0:
        returns = data['Close'].pct_change().dropna()
        print(f"リターン計算: {len(returns)}サンプル")

        if len(returns) > 10:
            print("✅ リスク計算に十分なデータ")
        else:
            print("❌ リスク計算にはデータ不足")
```

#### `リスクレベルが高すぎます`
**対応方法**:
```python
# リスク制限の調整検討
from production_risk_management_validator import production_risk_validator

# 現在の制限確認
print("現在のリスク制限:")
for key, value in production_risk_validator.risk_limits.items():
    print(f"  {key}: {value}")

# 一時的な制限緩和（注意して使用）
# production_risk_validator.risk_limits['max_drawdown'] = 0.15  # 10% → 15%

# ポジションサイズ調整
original_size = 1000000
risk_result = await production_risk_validator.validate_trading_risk("7203", original_size)
if not risk_result['validation_passed']:
    # サイズを50%に削減
    reduced_size = original_size * 0.5
    new_result = await production_risk_validator.validate_trading_risk("7203", reduced_size)
    print(f"ポジションサイズ削減結果: {new_result['overall_assessment']}")
```

### パフォーマンス問題

#### 高CPU使用率 (`CPU使用率: >85%`)
```python
# 1. 現在の状況確認
import psutil
cpu_percent = psutil.cpu_percent(interval=1)
memory_percent = psutil.virtual_memory().percent
print(f"CPU: {cpu_percent}%, メモリ: {memory_percent}%")

# 2. 重いプロセス特定
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
    if proc.info['cpu_percent'] > 5.0:
        print(f"プロセス {proc.info['pid']}: {proc.info['name']} - {proc.info['cpu_percent']}%")

# 3. 自動調整による対応
from realtime_monitoring_auto_tuning import realtime_monitoring_system
health = realtime_monitoring_system._collect_system_health()
tuning_actions = realtime_monitoring_system.auto_tuner.evaluate_and_apply_optimizations(health)

for action in tuning_actions:
    print(f"自動調整: {action.reason} - {action.parameter} {action.old_value} → {action.new_value}")
```

#### 高メモリ使用率 (`メモリ使用率: >90%`)
```python
# 1. メモリクリーンアップ
import gc
collected = gc.collect()
print(f"ガベージコレクション: {collected}オブジェクト回収")

# 2. キャッシュサイズ縮小
from realtime_performance_optimizer import realtime_performance_optimizer
old_size = realtime_performance_optimizer.data_cache.max_size
realtime_performance_optimizer.data_cache.max_size = old_size // 2
print(f"キャッシュサイズ縮小: {old_size} → {old_size // 2}")

# 3. キャッシュクリア
realtime_performance_optimizer.data_cache.cache.clear()
realtime_performance_optimizer.prediction_cache.cache.clear()
print("キャッシュクリア完了")

# 4. メモリ使用量再確認
new_memory = psutil.virtual_memory().percent
print(f"メモリ使用率: {new_memory}%")
```

#### 低キャッシュヒット率 (`キャッシュヒット率: <60%`)
```python
# キャッシュ統計確認
from realtime_performance_optimizer import realtime_performance_optimizer

caches = {
    'data_cache': realtime_performance_optimizer.data_cache,
    'prediction_cache': realtime_performance_optimizer.prediction_cache,
    'analysis_cache': realtime_performance_optimizer.analysis_cache
}

for name, cache in caches.items():
    stats = cache.get_stats()
    print(f"{name}:")
    print(f"  ヒット率: {stats['hit_rate']:.1%}")
    print(f"  サイズ: {stats['size']}")
    print(f"  総リクエスト: {stats['total_requests']}")

# キャッシュサイズ増加による改善
if stats['hit_rate'] < 0.6:
    old_size = cache.max_size
    cache.max_size = int(old_size * 1.5)
    print(f"キャッシュサイズ増加: {old_size} → {cache.max_size}")
```

### データベース関連エラー

#### `table alert_rules has no column named name`
**原因**: データベーススキーマ不一致
```python
# データベーススキーマ確認・修正
import sqlite3

db_path = "alert_data/alerts.db"
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()

    # テーブル構造確認
    cursor.execute("PRAGMA table_info(alert_rules);")
    columns = cursor.fetchall()
    print("現在のカラム:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

    # 不足カラム追加
    try:
        cursor.execute("ALTER TABLE alert_rules ADD COLUMN name TEXT;")
        print("nameカラム追加完了")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("nameカラムは既に存在")
        else:
            print(f"カラム追加エラー: {e}")
```

#### `Object of type datetime is not JSON serializable`
**解決**:
```python
# JSON serialization問題の修正
import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# 使用例
data = {'timestamp': datetime.now(), 'value': 100}
json_string = json.dumps(data, cls=DateTimeEncoder)
print(f"JSON: {json_string}")
```

### アラート関連エラー

#### `no running event loop`
**原因**: 非同期処理の不適切な実行
```python
# 修正方法
import asyncio

# 問題のあるコード
# await some_async_function()  # イベントループ外での実行

# 修正版
def fixed_notification_worker():
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # 非同期関数を同期的に実行
    if loop:
        loop.run_until_complete(async_function())
```

#### アラートが発火しない
```python
# アラート設定確認
from realtime_alert_notification_system import realtime_alert_system

# ルール確認
for rule_id, rule in realtime_alert_system.alert_rules.items():
    print(f"ルール {rule_id}:")
    print(f"  有効: {rule.enabled}")
    print(f"  条件: {rule.condition}")
    print(f"  最終発火: {rule.last_triggered}")
    print(f"  発火回数: {rule.trigger_count}")

# テストデータでアラート発火テスト
realtime_alert_system.update_market_data("TEST", {
    'current_price': 3000,
    'current_change': 0.08,  # 8%変動（5%超過でアラート）
    'volume_ratio': 2.0,
    'signal_strength': 50,
    'risk_score': 30,
    'volatility': 0.2,
    'prediction_confidence': 0.6,
    'signal_consensus': 0.5,
    'error_count': 0
})

await realtime_alert_system.check_alert_conditions("TEST")
active_alerts = realtime_alert_system.get_active_alerts()
print(f"アクティブアラート: {len(active_alerts)}件")
```

---

## 🔧 システム復旧手順

### レベル1: 軽微な問題 (5分以内復旧)

```bash
# 1. システム状態確認
python -c "
import asyncio
from realtime_monitoring_auto_tuning import realtime_monitoring_system
async def quick_check():
    health = realtime_monitoring_system._collect_system_health()
    print(f'CPU: {health.cpu_usage:.1f}% | メモリ: {health.memory_usage:.1f}% | 状態: {health.performance_level.value}')
asyncio.run(quick_check())
"

# 2. 軽微な最適化
python -c "
import gc
from realtime_performance_optimizer import realtime_performance_optimizer
collected = gc.collect()
print(f'GC: {collected}オブジェクト回収')
stats = realtime_performance_optimizer.data_cache.get_stats()
print(f'キャッシュヒット率: {stats[\"hit_rate\"]:.1%}')
"

# 3. 動作確認
python production_readiness_test.py
```

### レベル2: 中程度の問題 (15分以内復旧)

```bash
# 1. コンポーネント個別チェック
echo "=== データプロバイダー確認 ==="
python -c "
import asyncio
from real_data_provider_v2 import real_data_provider
async def test_data():
    data = await real_data_provider.get_stock_data('7203', '5d')
    print(f'データ取得: {\"OK\" if data is not None else \"NG\"} ({len(data) if data else 0}件)')
asyncio.run(test_data())
"

echo "=== 予測システム確認 ==="
python -c "
import asyncio
from optimized_prediction_system import optimized_prediction_system
async def test_prediction():
    try:
        prediction = await optimized_prediction_system.predict_with_optimized_models('7203')
        print(f'予測: OK (信頼度: {prediction.confidence:.2f})')
    except Exception as e:
        print(f'予測: NG ({e})')
asyncio.run(test_prediction())
"

# 2. 問題があるコンポーネントの再初期化
python -c "
# キャッシュクリアと再初期化
from realtime_performance_optimizer import realtime_performance_optimizer
realtime_performance_optimizer.data_cache.cache.clear()
realtime_performance_optimizer.prediction_cache.cache.clear()
print('キャッシュクリア完了')

# ガベージコレクション
import gc
collected = gc.collect()
print(f'メモリクリーンアップ: {collected}オブジェクト回収')
"

# 3. システム全体再検証
python comprehensive_live_validation.py
```

### レベル3: 深刻な問題 (1時間以内復旧)

```bash
# 1. 緊急停止
echo "システム緊急停止中..."
pkill -f python
sleep 2

# 2. データベースバックアップ作成
echo "データベースバックアップ作成中..."
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p backup_$timestamp

# 全データベースファイルをバックアップ
find . -name "*.db" -exec cp {} backup_$timestamp/ \;
echo "バックアップ完了: backup_$timestamp"

# 3. データベース整合性チェック
python -c "
import sqlite3
from pathlib import Path

def check_db_integrity():
    db_files = list(Path('.').glob('**/*.db'))
    corrupted = []

    for db_file in db_files:
        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('PRAGMA integrity_check;')
                result = cursor.fetchone()[0]
                if result != 'ok':
                    corrupted.append(str(db_file))
                    print(f'❌ {db_file}: {result}')
                else:
                    print(f'✅ {db_file}: OK')
        except Exception as e:
            corrupted.append(str(db_file))
            print(f'❌ {db_file}: {e}')

    return corrupted

corrupted_dbs = check_db_integrity()
if corrupted_dbs:
    print(f'破損データベース: {len(corrupted_dbs)}件')
    for db in corrupted_dbs:
        print(f'  - {db}')
else:
    print('全データベースの整合性: OK')
"

# 4. 破損データベースの修復
python -c "
import sqlite3
import os
from pathlib import Path

def repair_databases():
    db_files = list(Path('.').glob('**/*.db'))

    for db_file in db_files:
        try:
            # バックアップ作成
            backup_file = f'{db_file}.repair_backup'
            os.system(f'cp {db_file} {backup_file}')

            # 修復試行
            with sqlite3.connect(db_file) as conn:
                conn.execute('VACUUM;')
                conn.execute('REINDEX;')
                print(f'✅ 修復完了: {db_file}')

        except Exception as e:
            print(f'❌ 修復失敗: {db_file} - {e}')
            # バックアップから復元
            if os.path.exists(backup_file):
                os.system(f'cp {backup_file} {db_file}')

repair_databases()
"

# 5. 段階的システム復旧
echo "段階的システム復旧開始..."

echo "Phase 1: 基本機能確認"
python production_readiness_test.py

echo "Phase 2: 個別コンポーネント確認"
python -c "
import asyncio

async def component_check():
    components = []

    # データプロバイダー
    try:
        from real_data_provider_v2 import real_data_provider
        data = await real_data_provider.get_stock_data('7203', '5d')
        components.append(('データプロバイダー', 'OK' if data else 'NG'))
    except Exception as e:
        components.append(('データプロバイダー', f'NG: {e}'))

    # 予測システム
    try:
        from optimized_prediction_system import optimized_prediction_system
        pred = await optimized_prediction_system.predict_with_optimized_models('7203')
        components.append(('予測システム', 'OK' if pred else 'NG'))
    except Exception as e:
        components.append(('予測システム', f'NG: {e}'))

    # リスク管理
    try:
        from production_risk_management_validator import production_risk_validator
        risk = await production_risk_validator.validate_trading_risk('7203', 1000000)
        components.append(('リスク管理', 'OK' if risk['validation_passed'] else 'WARNING'))
    except Exception as e:
        components.append(('リスク管理', f'NG: {e}'))

    print('=== コンポーネント状況 ===')
    for name, status in components:
        emoji = '✅' if 'OK' in status else '⚠️' if 'WARNING' in status else '❌'
        print(f'{emoji} {name}: {status}')

    return components

asyncio.run(component_check())
"

echo "Phase 3: システム統合確認"
python comprehensive_live_validation.py

echo "復旧完了確認"
python -c "
print('=== システム復旧確認 ===')
import os
print(f'作業ディレクトリ: {os.getcwd()}')
print(f'Pythonバージョン: {os.popen(\"python --version\").read().strip()}')
print('重要ファイル存在確認:')
files = [
    'production_readiness_test.py',
    'comprehensive_live_validation.py',
    'real_data_provider_v2.py',
    'optimized_prediction_system.py'
]
for file in files:
    exists = os.path.exists(file)
    print(f'  {\"✅\" if exists else \"❌\"} {file}')
"
```

---

## 📊 パフォーマンス診断ツール

### システムパフォーマンス測定スクリプト
```python
#!/usr/bin/env python3
"""
システムパフォーマンス診断ツール
"""
import asyncio
import time
import psutil
import numpy as np
from datetime import datetime

async def performance_diagnostic():
    print("=== システムパフォーマンス診断 ===")
    print(f"診断開始時刻: {datetime.now()}")

    # システムリソース
    print(f"\n📊 システムリソース:")
    print(f"  CPU使用率: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"  メモリ使用率: {psutil.virtual_memory().percent:.1f}%")
    print(f"  ディスク使用率: {psutil.disk_usage('/').percent:.1f}%")

    # ネットワーク遅延測定
    print(f"\n🌐 ネットワーク遅延:")
    import subprocess
    try:
        result = subprocess.run(['ping', '-c', '3', 'finance.yahoo.com'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  Yahoo Finance: OK")
        else:
            print("  Yahoo Finance: NG - 接続問題あり")
    except:
        print("  Yahoo Finance: 測定不可")

    # コンポーネント応答時間測定
    print(f"\n⚡ コンポーネント応答時間:")

    # データ取得速度
    start = time.time()
    try:
        from real_data_provider_v2 import real_data_provider
        data = await real_data_provider.get_stock_data("7203", "5d")
        data_time = time.time() - start
        print(f"  データ取得: {data_time:.2f}秒 ({'OK' if data_time < 3 else 'SLOW'})")
    except Exception as e:
        print(f"  データ取得: エラー - {e}")

    # 予測速度
    start = time.time()
    try:
        from optimized_prediction_system import optimized_prediction_system
        prediction = await optimized_prediction_system.predict_with_optimized_models("7203")
        pred_time = time.time() - start
        print(f"  予測処理: {pred_time:.2f}秒 ({'OK' if pred_time < 5 else 'SLOW'})")
    except Exception as e:
        print(f"  予測処理: エラー - {e}")

    # キャッシュ効果測定
    print(f"\n💾 キャッシュ性能:")
    try:
        from realtime_performance_optimizer import realtime_performance_optimizer

        # 1回目（キャッシュミス）
        start = time.time()
        await realtime_performance_optimizer.optimize_data_retrieval("7203")
        first_time = time.time() - start

        # 2回目（キャッシュヒット）
        start = time.time()
        await realtime_performance_optimizer.optimize_data_retrieval("7203")
        second_time = time.time() - start

        if second_time > 0:
            speedup = first_time / second_time
            print(f"  1回目: {first_time:.2f}秒")
            print(f"  2回目: {second_time:.2f}秒")
            print(f"  高速化: {speedup:.1f}x ({'EXCELLENT' if speedup > 5 else 'GOOD' if speedup > 2 else 'POOR'})")

        # キャッシュ統計
        cache_stats = realtime_performance_optimizer.data_cache.get_stats()
        print(f"  ヒット率: {cache_stats['hit_rate']:.1%}")

    except Exception as e:
        print(f"  キャッシュテスト: エラー - {e}")

    print(f"\n診断完了時刻: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(performance_diagnostic())
```

保存して実行:
```bash
python performance_diagnostic.py
```

---

## 🚨 緊急事態対応プレイブック

### Phase 1: 状況確認 (2分以内)
```bash
#!/bin/bash
echo "=== 緊急事態対応開始 ==="
echo "時刻: $(date)"

# 1. システム生存確認
echo "1. システム生存確認"
if pgrep -f python > /dev/null; then
    echo "✅ Pythonプロセス稼働中"
else
    echo "❌ Pythonプロセス停止"
fi

# 2. リソース確認
echo "2. リソース状況"
echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')%"
echo "  メモリ: $(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')%"
echo "  ディスク: $(df -h . | tail -1 | awk '{print $5}')"

# 3. ネットワーク確認
echo "3. ネットワーク確認"
if ping -c 1 google.com > /dev/null 2>&1; then
    echo "✅ インターネット接続: OK"
else
    echo "❌ インターネット接続: NG"
fi
```

### Phase 2: 応急処置 (5分以内)
```python
#!/usr/bin/env python3
"""
緊急応急処置スクリプト
"""
import os
import gc
import psutil
import asyncio
from datetime import datetime

async def emergency_response():
    print("=== 緊急応急処置開始 ===")
    actions = []

    # 1. メモリクリーンアップ
    try:
        before_mem = psutil.virtual_memory().percent
        collected = gc.collect()
        after_mem = psutil.virtual_memory().percent
        actions.append(f"メモリクリーンアップ: {before_mem:.1f}% → {after_mem:.1f}% ({collected}オブジェクト回収)")
    except Exception as e:
        actions.append(f"メモリクリーンアップ失敗: {e}")

    # 2. キャッシュクリア
    try:
        from realtime_performance_optimizer import realtime_performance_optimizer
        old_size = len(realtime_performance_optimizer.data_cache.cache)
        realtime_performance_optimizer.data_cache.cache.clear()
        realtime_performance_optimizer.prediction_cache.cache.clear()
        actions.append(f"キャッシュクリア: {old_size}エントリ削除")
    except Exception as e:
        actions.append(f"キャッシュクリア失敗: {e}")

    # 3. 基本機能確認
    try:
        from real_data_provider_v2 import real_data_provider
        data = await real_data_provider.get_stock_data("7203", "1d")
        if data is not None:
            actions.append(f"データ取得: OK ({len(data)}レコード)")
        else:
            actions.append("データ取得: NG")
    except Exception as e:
        actions.append(f"データ取得テスト失敗: {e}")

    # 4. システム状態記録
    status_file = f"emergency_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(status_file, 'w') as f:
        f.write(f"緊急対応実行時刻: {datetime.now()}\n")
        for action in actions:
            f.write(f"- {action}\n")

    actions.append(f"状態記録: {status_file}")

    # 結果表示
    for action in actions:
        print(f"  {action}")

    print("=== 緊急応急処置完了 ===")

if __name__ == "__main__":
    asyncio.run(emergency_response())
```

### Phase 3: 回復確認 (10分以内)
```bash
# 回復確認スクリプト
echo "=== 回復確認開始 ==="

# 基本機能テスト
echo "1. 基本機能テスト"
python -c "
import asyncio
async def basic_test():
    try:
        from real_data_provider_v2 import real_data_provider
        data = await real_data_provider.get_stock_data('7203', '1d')
        print(f'  データ取得: {\"OK\" if data is not None else \"NG\"}')
    except Exception as e:
        print(f'  データ取得: エラー - {e}')

    try:
        from optimized_prediction_system import optimized_prediction_system
        pred = await optimized_prediction_system.predict_with_optimized_models('7203')
        print(f'  予測: {\"OK\" if pred else \"NG\"}')
    except Exception as e:
        print(f'  予測: エラー - {e}')

asyncio.run(basic_test())
"

# クイック検証
echo "2. クイック検証実行"
python production_readiness_test.py | tail -5

# システム状態確認
echo "3. システム状態"
python -c "
import psutil
print(f'  CPU: {psutil.cpu_percent():.1f}%')
print(f'  メモリ: {psutil.virtual_memory().percent:.1f}%')
print(f'  プロセス数: {len(psutil.pids())}')
"

echo "=== 回復確認完了 ==="
```

---

## 📞 エスカレーション基準

### 自動エスカレーション条件
- システム停止 > 30分
- データ取得失敗率 > 50%
- 予測精度 < 40%
- メモリ使用率 > 95%
- 連続エラー > 100件

### 手動エスカレーション判断基準
- 業務時間中の部分的機能停止
- データ破損の疑い
- セキュリティインシデント
- 復旧手順で解決できない問題

---

**最終更新**: 2025年8月14日  
**バージョン**: 1.0.0  
**緊急連絡**: システム管理者まで