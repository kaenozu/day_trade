# デイトレードAIシステム運用マニュアル

## 📋 目次

1. [システム概要](#システム概要)
2. [セットアップ](#セットアップ)
3. [基本操作](#基本操作)
4. [システム監視](#システム監視)
5. [リスク管理](#リスク管理)
6. [トラブルシューティング](#トラブルシューティング)
7. [メンテナンス](#メンテナンス)
8. [緊急時対応](#緊急時対応)

---

## システム概要

### 🎯 システムの目的
デイトレードAIシステムは、機械学習を活用した株式市場分析・予測システムです。リアルタイムデータ分析、自動予測、リスク管理を統合し、個人投資家の取引支援を行います。

### 🏗️ システム構成

```
デイトレードAIシステム
├── データ取得層 (real_data_provider_v2.py)
├── 予測エンジン (optimized_prediction_system.py)
├── リスク管理 (advanced_risk_management_system.py)
├── パフォーマンス最適化 (realtime_performance_optimizer.py)
├── アラート通知 (realtime_alert_notification_system.py)
├── 監視・調整 (realtime_monitoring_auto_tuning.py)
└── データ安定性 (market_data_stability_system.py)
```

### 🚀 主要機能
- **リアルタイム株価データ取得**: Yahoo Finance API経由
- **AI予測**: Random Forest + XGBoost アンサンブル学習
- **リスク管理**: VaR計算、ドローダウン監視
- **自動調整**: システム性能の自動最適化
- **アラート通知**: 重要な市場変動の即座通知
- **データ品質管理**: 多段階フォールバック機能

---

## セットアップ

### 📦 必要なライブラリ
```bash
pip install pandas numpy yfinance scikit-learn xgboost psutil asyncio
pip install requests aiohttp sqlite3 threading pathlib
```

### 🔧 初期設定

1. **作業ディレクトリ作成**
```bash
mkdir day_trade
cd day_trade
```

2. **データディレクトリ作成**
```bash
mkdir cache_data
mkdir validation_data
mkdir performance_data
mkdir alert_data
mkdir stability_data
mkdir monitoring_data
mkdir risk_validation_data
```

3. **システム起動確認**
```python
python production_readiness_test.py
```

### ⚙️ 設定ファイル調整

#### リスク管理設定
```python
# production_risk_management_validator.py内
risk_limits = {
    'max_position_risk': 0.02,      # 単一ポジション2%
    'max_daily_var': 0.05,          # 日次VaR 5%
    'max_drawdown': 0.10,           # 最大ドローダウン10%
}
```

#### アラート設定
```python
# realtime_alert_notification_system.py内
notification_config = NotificationConfig(
    console_enabled=True,
    email_enabled=False,  # 必要に応じてTrue
    desktop_enabled=True
)
```

---

## 基本操作

### 🚀 システム起動

#### 1. クイック検証実行
```bash
python production_readiness_test.py
```
**期待される結果**: 平均スコア70%以上で「部分的準備完了」以上

#### 2. 個別コンポーネント起動

**データ取得テスト**
```python
from real_data_provider_v2 import real_data_provider
data = await real_data_provider.get_stock_data("7203", "5d")
print(f"取得データ: {len(data)}レコード")
```

**予測システム起動**
```python
from optimized_prediction_system import optimized_prediction_system
prediction = await optimized_prediction_system.predict_with_optimized_models("7203")
print(f"予測: {prediction.prediction}, 信頼度: {prediction.confidence}")
```

**リスク検証**
```python
from production_risk_management_validator import production_risk_validator
result = await production_risk_validator.validate_trading_risk("7203", 1000000)
print(f"リスク判定: {result['overall_assessment']}")
```

#### 3. 統合システム起動
```bash
python comprehensive_live_validation.py
```

### 📊 日常運用フロー

#### 朝の準備 (9:00前)
1. **システムヘルスチェック**
```bash
python production_readiness_test.py
```

2. **監視システム起動**
```python
from realtime_monitoring_auto_tuning import realtime_monitoring_system
realtime_monitoring_system.start_monitoring()
```

3. **アラートシステム起動**
```python
from realtime_alert_notification_system import realtime_alert_system
realtime_alert_system.start_monitoring()
```

#### 取引中 (9:00-15:00)
1. **リアルタイム監視**
   - CPU/メモリ使用率
   - キャッシュヒット率
   - 応答時間
   - エラー率

2. **定期的な予測実行**
```python
# 注目銘柄の予測
symbols = ["7203", "8306", "4751"]
for symbol in symbols:
    prediction = await optimized_prediction_system.predict_with_optimized_models(symbol)
    print(f"{symbol}: {prediction.prediction} (信頼度: {prediction.confidence:.2f})")
```

3. **リスク監視**
```python
# ポートフォリオリスクチェック
positions = {"7203": 1000000, "8306": 500000}
portfolio_result = await production_risk_validator.validate_portfolio_risk(positions)
print(f"ポートフォリオ判定: {portfolio_result['overall_assessment']}")
```

#### 夕方の振り返り (15:30以降)
1. **パフォーマンスレポート確認**
```python
from realtime_performance_optimizer import realtime_performance_optimizer
report = realtime_performance_optimizer.get_performance_report()
print(f"システム状態: {report['performance_status']}")
```

2. **アラート統計確認**
```python
from realtime_alert_notification_system import realtime_alert_system
stats = realtime_alert_system.get_alert_statistics()
print(f"本日のアラート: {stats['today_alerts']}件")
```

---

## システム監視

### 📈 監視ダッシュボード

#### 主要指標
| 指標 | 正常範囲 | 警告レベル | 危険レベル |
|------|----------|------------|------------|
| CPU使用率 | <70% | 70-85% | >85% |
| メモリ使用率 | <80% | 80-90% | >90% |
| キャッシュヒット率 | >80% | 60-80% | <60% |
| 応答時間 | <2秒 | 2-5秒 | >5秒 |
| エラー率 | <1% | 1-5% | >5% |

#### リアルタイム監視コマンド
```bash
# システム全体のヘルスチェック
python -c "
from realtime_monitoring_auto_tuning import realtime_monitoring_system
import asyncio
async def check():
    health = realtime_monitoring_system._collect_system_health()
    print(f'CPU: {health.cpu_usage:.1f}%')
    print(f'メモリ: {health.memory_usage:.1f}%')
    print(f'パフォーマンス: {health.performance_level.value}')
asyncio.run(check())
"
```

### 🔔 アラート設定

#### 重要度別アラート
- **CRITICAL**: システムエラー、極度の高リスク
- **HIGH**: 価格急変、取引機会
- **MEDIUM**: 技術的シグナル、出来高急増
- **LOW**: 一般的な市場情報

#### アラート確認・解決
```python
# アクティブアラート確認
active_alerts = realtime_alert_system.get_active_alerts()
for alert in active_alerts:
    print(f"{alert.symbol}: {alert.title} ({alert.priority.value})")

# アラート確認・解決
realtime_alert_system.acknowledge_alert("alert_id")
realtime_alert_system.resolve_alert("alert_id")
```

### 📊 ログ監視

#### ログファイル場所
- **システムログ**: コンソール出力
- **データベース**: `*.db` ファイル
- **キャッシュ**: `cache_data/` ディレクトリ

#### 重要なログメッセージ
- `INFO: 最適化特徴量作成完了` - 予測準備完了
- `WARNING: All data sources failed` - データ取得失敗
- `ERROR: 予測実行エラー` - 予測システムエラー
- `INFO: アラート発火` - 重要イベント発生

---

## リスク管理

### 🛡️ リスク管理フレームワーク

#### 3段階リスク評価
1. **個別銘柄リスク**: VaR、ボラティリティ、ドローダウン
2. **ポジションリスク**: 投資額、リスク割合
3. **ポートフォリオリスク**: 分散度、相関リスク

#### リスクレベル定義
| レベル | スコア | 判定 | 対応 |
|--------|--------|------|------|
| VERY_LOW | 0-10 | 非常に安全 | ポジション増加可 |
| LOW | 11-25 | 安全 | 通常取引 |
| MODERATE | 26-40 | 中リスク | 慎重監視 |
| HIGH | 41-60 | 高リスク | ポジション縮小 |
| VERY_HIGH | 61-80 | 非常に高リスク | 取引回避推奨 |
| EXTREME | 81-100 | 極度に危険 | 取引禁止 |

### 📋 リスクチェックリスト

#### 取引前チェック
- [ ] システムヘルス確認 (CPU<80%, メモリ<85%)
- [ ] データ品質確認 (取得成功率>90%)
- [ ] 個別銘柄リスク評価実行
- [ ] リスク制限内かチェック
- [ ] アラート状況確認

#### 取引中監視
- [ ] リアルタイムリスクモニタリング
- [ ] VaR制限遵守確認
- [ ] ドローダウン監視
- [ ] システムパフォーマンス監視

#### 取引後検証
- [ ] 実績とリスク予測の比較
- [ ] リスクモデルの精度検証
- [ ] システムログ確認

---

## トラブルシューティング

### 🚨 一般的な問題と解決方法

#### 1. データ取得エラー
**症状**: `All data sources failed for [symbol]`

**原因と解決**:
```python
# 原因1: ネットワーク接続問題
# 解決: インターネット接続確認、プロキシ設定チェック

# 原因2: Yahoo Finance API制限
# 解決: 取得間隔調整
import time
time.sleep(1)  # 1秒待機

# 原因3: 銘柄コード間違い
# 解決: 正しい銘柄コード確認
# 7203: トヨタ自動車, 8306: 三菱UFJ銀行, 4751: サイバーエージェント
```

#### 2. 予測システムエラー
**症状**: `予測実行エラー` または `予測結果が不正`

**解決手順**:
```python
# 1. データ確認
data = await real_data_provider.get_stock_data("7203", "5d")
print(f"データ件数: {len(data)}, カラム: {data.columns.tolist()}")

# 2. 特徴量確認
from optimized_prediction_system import optimized_prediction_system
features = optimized_prediction_system.create_optimized_features(data)
print(f"特徴量数: {len(features.columns)}")

# 3. モデル再学習
await optimized_prediction_system.train_optimized_models("7203")
```

#### 3. メモリ不足
**症状**: `high_memory` アラート、システム応答遅延

**解決手順**:
```python
# 1. ガベージコレクション強制実行
import gc
collected = gc.collect()
print(f"回収されたオブジェクト: {collected}")

# 2. キャッシュサイズ調整
from realtime_performance_optimizer import realtime_performance_optimizer
realtime_performance_optimizer.data_cache.max_size = 500  # サイズ縮小

# 3. 不要なデータ削除
realtime_performance_optimizer.data_cache.cache.clear()
```

#### 4. 高CPU使用率
**症状**: CPU使用率>85%、応答時間悪化

**解決手順**:
```python
# 1. ワーカー数減少
from realtime_monitoring_auto_tuning import realtime_monitoring_system
# 自動調整が作動するのを待つか、手動調整
import multiprocessing
optimal_workers = max(1, multiprocessing.cpu_count() // 2)

# 2. 処理間隔調整
# ポーリング間隔を延長 (60秒 → 120秒)
time.sleep(120)

# 3. 不要な並行処理削減
```

### 🔧 システム復旧手順

#### レベル1: 軽微な問題
1. **システム再起動**
```bash
# 個別コンポーネント再起動
python production_readiness_test.py
```

2. **キャッシュクリア**
```python
from realtime_performance_optimizer import realtime_performance_optimizer
realtime_performance_optimizer.data_cache.cache.clear()
```

#### レベル2: 中程度の問題
1. **データベースリセット**
```python
import os
import sqlite3

# 古いデータベースファイルのバックアップと削除
db_dirs = ["validation_data", "performance_data", "alert_data",
           "stability_data", "monitoring_data", "risk_validation_data"]
for db_dir in db_dirs:
    for file in os.listdir(db_dir):
        if file.endswith('.db'):
            os.rename(f"{db_dir}/{file}", f"{db_dir}/{file}.backup")
```

2. **設定リセット**
```python
# デフォルト設定で再初期化
from realtime_monitoring_auto_tuning import RealtimeMonitoringSystem
new_monitoring = RealtimeMonitoringSystem()
```

#### レベル3: 深刻な問題
1. **完全システムリセット**
```bash
# 全データディレクトリバックアップ
cp -r cache_data cache_data_backup
cp -r validation_data validation_data_backup

# システム再セットアップ
python -c "
import shutil
import os
dirs = ['cache_data', 'validation_data', 'performance_data',
        'alert_data', 'stability_data', 'monitoring_data', 'risk_validation_data']
for d in dirs:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
"
```

### 📞 エラーコード一覧

| コード | 説明 | 重要度 | 対応 |
|--------|------|--------|------|
| DATA_001 | データ取得失敗 | 中 | ネットワーク確認 |
| PRED_001 | 予測モデルエラー | 高 | モデル再学習 |
| RISK_001 | リスク計算エラー | 高 | データ品質確認 |
| PERF_001 | パフォーマンス劣化 | 中 | リソース確認 |
| ALERT_001 | アラート送信失敗 | 低 | 通知設定確認 |
| SYS_001 | システムリソース不足 | 高 | リソース追加 |

---

## メンテナンス

### 🔄 定期メンテナンス

#### 日次 (毎日 16:00)
```bash
# 1. パフォーマンスレポート確認
python -c "
from realtime_performance_optimizer import realtime_performance_optimizer
report = realtime_performance_optimizer.get_performance_report()
print('=== 日次パフォーマンスレポート ===')
print(f'システム状態: {report[\"performance_status\"]}')
print(f'平均CPU: {report[\"system_metrics\"][\"avg_cpu_usage\"]:.1f}%')
print(f'平均メモリ: {report[\"system_metrics\"][\"avg_memory_usage\"]:.1f}%')
"

# 2. アラート統計確認
python -c "
from realtime_alert_notification_system import realtime_alert_system
stats = realtime_alert_system.get_alert_statistics()
print('=== 日次アラート統計 ===')
print(f'本日のアラート: {stats[\"today_alerts\"]}件')
print(f'アクティブアラート: {stats[\"active_alerts\"]}件')
"

# 3. キャッシュ最適化
python -c "
import gc
collected = gc.collect()
print(f'ガベージコレクション: {collected}オブジェクト回収')
"
```

#### 週次 (毎週日曜日)
```bash
# 1. データベース最適化
python -c "
import sqlite3
from pathlib import Path

db_dirs = ['validation_data', 'performance_data', 'alert_data',
           'stability_data', 'monitoring_data', 'risk_validation_data']

for db_dir in db_dirs:
    db_path = Path(db_dir)
    if db_path.exists():
        for db_file in db_path.glob('*.db'):
            with sqlite3.connect(db_file) as conn:
                conn.execute('VACUUM;')
                print(f'最適化完了: {db_file}')
"

# 2. ログファイルローテーション
python -c "
import os
from datetime import datetime

backup_suffix = datetime.now().strftime('%Y%m%d')
print(f'ログバックアップ作成: {backup_suffix}')
"

# 3. システム全体チェック
python comprehensive_live_validation.py
```

#### 月次 (毎月第1日曜日)
```bash
# 1. 予測モデル再学習
python -c "
import asyncio
from optimized_prediction_system import optimized_prediction_system

async def retrain_models():
    symbols = ['7203', '8306', '4751']
    for symbol in symbols:
        print(f'{symbol}のモデル再学習開始...')
        await optimized_prediction_system.train_optimized_models(symbol)
        print(f'{symbol}完了')

asyncio.run(retrain_models())
"

# 2. リスク管理パラメータ見直し
python -c "
from production_risk_management_validator import production_risk_validator
print('=== 現在のリスク制限 ===')
for key, value in production_risk_validator.risk_limits.items():
    print(f'{key}: {value}')
"

# 3. システム性能ベンチマーク
python production_readiness_test.py
```

### 🗂️ データ管理

#### データ保持期間
- **キャッシュデータ**: 24時間
- **パフォーマンスメトリクス**: 30日間
- **アラート履歴**: 90日間
- **リスク評価履歴**: 1年間
- **取引履歴**: 永続保存

#### データクリーンアップスクリプト
```python
# データクリーンアップ実行
import os
import sqlite3
from datetime import datetime, timedelta

def cleanup_old_data():
    cutoff_date = (datetime.now() - timedelta(days=90)).isoformat()

    # アラートデータクリーンアップ
    with sqlite3.connect("alert_data/alerts.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM alert_history WHERE timestamp < ?",
            (cutoff_date,)
        )
        deleted = cursor.rowcount
        print(f"古いアラート削除: {deleted}件")

    # パフォーマンスデータクリーンアップ (30日)
    cutoff_date_30 = (datetime.now() - timedelta(days=30)).isoformat()
    with sqlite3.connect("performance_data/optimization_metrics.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM performance_metrics WHERE timestamp < ?",
            (cutoff_date_30,)
        )
        deleted = cursor.rowcount
        print(f"古いパフォーマンスデータ削除: {deleted}件")

cleanup_old_data()
```

---

## 緊急時対応

### 🚨 緊急事態の分類

#### レベル1: サービス停止
**状況**: システムが応答しない、重要なエラーが連続発生

**対応手順**:
1. **即座停止**
```bash
# 全プロセス停止
pkill -f python
```

2. **問題特定**
```bash
# システムリソース確認
top
df -h
free -h
```

3. **ログ確認**
```bash
# 最近のエラーログ確認
python -c "
import logging
logging.basicConfig(level=logging.ERROR)
# 最新のエラーを確認
"
```

4. **安全な再起動**
```bash
# 段階的再起動
python production_readiness_test.py  # 基本機能確認
python comprehensive_live_validation.py  # 全体確認
```

#### レベル2: データ破損
**状況**: データベースエラー、不正なデータ検出

**対応手順**:
1. **バックアップ確認**
```bash
ls -la *_backup/
ls -la *.db.backup
```

2. **データ復旧**
```python
import shutil
import os

# バックアップからの復旧例
backup_dirs = [
    "cache_data_backup",
    "validation_data_backup",
    "performance_data_backup"
]

for backup_dir in backup_dirs:
    if os.path.exists(backup_dir):
        original_dir = backup_dir.replace("_backup", "")
        shutil.rmtree(original_dir)
        shutil.copytree(backup_dir, original_dir)
        print(f"復旧完了: {original_dir}")
```

3. **整合性チェック**
```python
# データ整合性確認
import sqlite3
from pathlib import Path

def check_db_integrity():
    db_files = list(Path(".").glob("**/*.db"))
    for db_file in db_files:
        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check;")
                result = cursor.fetchone()[0]
                print(f"{db_file}: {result}")
        except Exception as e:
            print(f"{db_file}: エラー - {e}")

check_db_integrity()
```

#### レベル3: セキュリティインシデント
**状況**: 不正アクセス検出、異常なシステム動作

**対応手順**:
1. **即座隔離**
```bash
# ネットワーク接続遮断
sudo iptables -P INPUT DROP
sudo iptables -P OUTPUT DROP
```

2. **証跡保全**
```bash
# ログファイル保護
cp -r validation_data security_incident_$(date +%Y%m%d)
cp -r performance_data security_incident_$(date +%Y%m%d)
```

3. **システム点検**
```bash
# システム状態確認
ps aux | grep python
netstat -tulpn
```

4. **安全な復旧**
```bash
# クリーンインストール後の復旧
# 信頼できるバックアップからのみ復旧
```

### 📞 緊急連絡先

#### システム管理者
- **内部連絡**: システムログを確認し、問題レベルを判定
- **外部連絡**: 重大な問題の場合は開発チームに連絡

#### エスカレーション基準
| レベル | 状況 | 対応時間 | エスカレーション先 |
|--------|------|----------|-------------------|
| P1 | システム完全停止 | 1時間以内 | 開発チーム |
| P2 | 部分的機能停止 | 4時間以内 | 開発チーム |
| P3 | パフォーマンス劣化 | 1日以内 | システム管理者 |
| P4 | 軽微な問題 | 3日以内 | システム管理者 |

---

## 🔧 補足情報

### システム要件
- **OS**: Windows 10/11 または Linux
- **Python**: 3.8以上
- **メモリ**: 4GB以上推奨
- **ディスク**: 10GB以上の空き容量
- **ネットワーク**: インターネット接続必須

### ライセンスと法的注意事項
- このシステムは投資助言を提供するものではありません
- すべての投資決定は自己責任で行ってください
- 市場リスクについて十分理解した上で使用してください

### サポート
- **GitHub**: システムに関する問題報告
- **ドキュメント**: 最新版の使用方法確認
- **コミュニティ**: ユーザー同士の情報交換

---

**最終更新**: 2025年8月14日  
**バージョン**: 1.0.0  
**作成者**: デイトレードAI開発チーム