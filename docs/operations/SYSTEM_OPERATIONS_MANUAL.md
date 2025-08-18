# Day Trading System 統合システム運用マニュアル

**統合データベース管理システム対応版** - 2025年8月版

---

## 📋 概要

このマニュアルでは、Day Trading Systemの日常運用、監視、メンテナンス手順を詳細に説明します。統合データベース管理システムを中心とした包括的な運用ガイドです。

### 🎯 対象読者
- **システム運用者**: 日常の監視・運用作業担当者
- **データベース管理者**: データベース関連の専門作業担当者
- **システム管理者**: インフラ・セキュリティ管理担当者
- **開発者**: システム修正・改良担当者

---

## 🏗️ システム構成

### アーキテクチャ概要

```
Day Trading System 統合アーキテクチャ
├── 統合データベース管理システム
│   ├── PostgreSQL/SQLite 本番データベース
│   ├── 自動バックアップシステム
│   ├── リアルタイム監視システム
│   ├── 復元・ロールバック機能
│   └── ダッシュボード・レポート機能
├── DDD (ドメイン駆動設計) コア
│   ├── ドメインエンティティ
│   ├── 値オブジェクト
│   ├── ドメインサービス
│   └── 集約ルート
├── 統合エラーハンドリングシステム
├── 統合ログシステム
└── パフォーマンス最適化エンジン
```

### 主要コンポーネント

| コンポーネント | 説明 | 配置 |
|---------------|------|------|
| **統合データベースマネージャー** | 本番データベース管理の中心 | `src/day_trade/infrastructure/database/` |
| **バックアップシステム** | 自動・手動バックアップ機能 | `src/day_trade/infrastructure/database/backup_manager.py` |
| **監視システム** | リアルタイム監視・アラート | `src/day_trade/infrastructure/database/monitoring_system.py` |
| **ダッシュボード** | 可視化・レポート機能 | `src/day_trade/infrastructure/database/dashboard.py` |
| **復元システム** | バックアップからの復元機能 | `src/day_trade/infrastructure/database/restore_manager.py` |

---

## 📅 日常運用作業

### 朝の運用確認 (毎日 9:00)

#### 1. システム状態確認

```bash
# システム全体状態確認
cd /opt/daytrading
source venv/bin/activate
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    status = manager.get_system_status()
    print(f'システム状態: {status[\"overall_health\"]}')
    print(f'コンポーネント数: {len(status[\"components\"])}')
    print(f'アクティブアラート: {len(manager.get_active_alerts())}件')
else:
    print('システム初期化エラー')
"
```

#### 2. 夜間バックアップ確認

```bash
# 夜間バックアップの実行状況確認
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    backups = manager.list_backups()
    if backups:
        latest = backups[0]
        print(f'最新バックアップ: {latest[\"filename\"]}')
        print(f'サイズ: {latest[\"size_mb\"]}MB')
        print(f'作成時刻: {latest[\"created_at\"]}')
        print(f'ステータス: {latest[\"status\"]}')
    else:
        print('❌ バックアップが見つかりません')
"
```

#### 3. エラーログ確認

```bash
# 過去24時間のエラーログ確認
tail -n 100 logs/production/errors.jsonl | jq -r '.timestamp + \" [\" + .level + \"] \" + .message'

# 重要なエラーのカウント
grep -c "ERROR\|CRITICAL" logs/production/production.log
```

#### 4. パフォーマンス状況確認

```bash
# 現在のメトリクス確認
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    metrics = manager.get_current_metrics()
    if metrics:
        print(f'CPU使用率: {metrics[\"cpu_usage\"]}%')
        print(f'メモリ使用量: {metrics[\"memory_usage_mb\"]}MB')
        print(f'アクティブ接続数: {metrics[\"active_connections\"]}')
        print(f'接続プール使用率: {metrics[\"connection_pool_usage\"]:.1%}')
    else:
        print('メトリクス取得失敗')
"
```

---

### 昼の運用確認 (毎日 13:00)

#### 1. トレーディング処理状況確認

```bash
# 取引処理状況の確認
python -c "
import sqlite3
from datetime import datetime, timedelta

# データベース接続
conn = sqlite3.connect('data/production/trading.db')
cursor = conn.cursor()

# 今日の取引統計
today = datetime.now().strftime('%Y-%m-%d')
cursor.execute('''
    SELECT COUNT(*) as trade_count,
           AVG(execution_time) as avg_time,
           SUM(profit_loss) as total_pnl
    FROM trades
    WHERE DATE(created_at) = ?
''', (today,))

result = cursor.fetchone()
print(f'本日の取引数: {result[0]}')
print(f'平均実行時間: {result[1]:.2f}秒')
print(f'累計損益: {result[2]:.2f}円')

conn.close()
"
```

#### 2. ML分析パフォーマンス確認

```bash
# ML分析処理時間の確認
grep "ML分析完了" logs/production/application.jsonl | tail -5 | jq -r '.analysis_duration_seconds'
```

---

### 夕方の運用確認 (毎日 17:00)

#### 1. 日次レポート確認

```bash
# 日次レポート生成と確認
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    report = manager.generate_report()
    print(f'レポート生成: {report[\"status\"]}')
    if report[\"status\"] == 'success':
        print(f'ファイル: {report[\"file_path\"]}')
        print(f'データ期間: {report[\"period\"]}')
"
```

#### 2. ディスク容量確認

```bash
# ディスク使用量確認
df -h /opt/daytrading

# データベースファイルサイズ確認
du -sh data/production/
du -sh logs/production/
du -sh backups/
```

#### 3. 接続プール状況確認

```bash
# 接続プール詳細状況
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager and manager.production_db_manager:
    health = manager.production_db_manager.health_check()
    pool_status = health.get('pool_status', {})
    print(f'プールサイズ: {pool_status.get(\"size\", 0)}')
    print(f'チェックアウト中: {pool_status.get(\"checked_out\", 0)}')
    print(f'オーバーフロー: {pool_status.get(\"overflow\", 0)}')
    print(f'使用率: {pool_status.get(\"utilization\", 0):.1%}')
"
```

---

## 📊 監視・アラート管理

### アクティブアラート確認

#### 現在のアラート状況

```bash
# アクティブアラート一覧
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    alerts = manager.get_active_alerts()
    if alerts:
        print(f'アクティブアラート: {len(alerts)}件')
        for alert in alerts:
            print(f'- {alert[\"severity\"]}: {alert[\"message\"]}')
    else:
        print('アクティブアラートなし')
"
```

#### アラート統計情報

```bash
# アラート統計
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager and manager.monitoring_system:
    stats = manager.monitoring_system.get_alert_statistics()
    print(f'総アラート数: {stats[\"total_alerts\"]}')
    print(f'Critical: {stats[\"critical_count\"]}件')
    print(f'Warning: {stats[\"warning_count\"]}件')
    print(f'Info: {stats[\"info_count\"]}件')
"
```

### アラート対応手順

#### Critical アラート対応

1. **システム停止**: 緊急時はシステムを安全に停止
   ```bash
   sudo systemctl stop daytrading
   ```

2. **緊急バックアップ作成**:
   ```bash
   python -c "
   from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
   manager = get_unified_database_manager()
   if manager:
       result = manager.create_backup('emergency')
       print(f'緊急バックアップ: {result[\"status\"]}')
   "
   ```

3. **問題分析**: ログとメトリクスを確認
4. **復旧作業**: 問題に応じた復旧手順実行
5. **システム再起動**: 問題解決後にシステム再開

#### Warning アラート対応

1. **詳細調査**: メトリクス履歴を確認
2. **予防的措置**: 必要に応じてリソース調整
3. **継続監視**: 状況の推移を観察

---

## 🔧 週次メンテナンス作業

### 毎週月曜日 (10:00)

#### 1. バックアップ整合性テスト

```bash
# 過去1週間のバックアップ一覧確認
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
from datetime import datetime, timedelta

manager = get_unified_database_manager()
if manager:
    backups = manager.list_backups()
    week_ago = datetime.now() - timedelta(days=7)

    recent_backups = [
        b for b in backups
        if datetime.fromisoformat(b['created_at']) >= week_ago
    ]

    print(f'過去1週間のバックアップ: {len(recent_backups)}件')
    for backup in recent_backups:
        print(f'- {backup[\"filename\"]}: {backup[\"size_mb\"]}MB ({backup[\"status\"]})')
"

# 最新バックアップの整合性テスト（ドライラン）
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    backups = manager.list_backups()
    if backups:
        latest = backups[0]['filename']
        result = manager.restore_database(latest, dry_run=True)
        print(f'整合性テスト: {result[\"status\"]}')
"
```

#### 2. パフォーマンス分析

```bash
# 週次パフォーマンス統計
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
from datetime import datetime, timedelta

manager = get_unified_database_manager()
if manager and manager.monitoring_system:
    # 過去1週間のメトリクス取得
    metrics_history = manager.monitoring_system.get_metrics_history(hours=168)  # 7日

    if metrics_history:
        cpu_avg = sum(m['cpu_usage'] for m in metrics_history) / len(metrics_history)
        memory_avg = sum(m['memory_usage_mb'] for m in metrics_history) / len(metrics_history)

        print(f'過去1週間の平均:')
        print(f'- CPU使用率: {cpu_avg:.1f}%')
        print(f'- メモリ使用量: {memory_avg:.1f}MB')
        print(f'- データポイント数: {len(metrics_history)}')
"
```

#### 3. ログローテーション確認

```bash
# ログファイルサイズ確認
du -sh logs/production/*.log
du -sh logs/production/*.jsonl

# 古いログファイルの圧縮・削除（30日以上）
find logs/production/ -name "*.log.*" -mtime +30 -exec gzip {} \;
find logs/production/ -name "*.log.*.gz" -mtime +90 -delete
```

---

### 毎週金曜日 (16:00)

#### 1. データベース最適化

```bash
# PostgreSQL統計情報更新・バキューム
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
from sqlalchemy import text

manager = get_unified_database_manager()
if manager and manager.production_db_manager:
    with manager.production_db_manager.get_session() as session:
        # 統計情報更新
        session.execute(text('ANALYZE;'))
        print('統計情報更新完了')

        # バキューム（軽量版）
        session.execute(text('VACUUM (ANALYZE);'))
        print('バキューム完了')
"
```

#### 2. 週次セキュリティチェック

```bash
# ファイル権限確認
find /opt/daytrading -type f -name "*.py" ! -perm 644 -ls
find /opt/daytrading -type f -name "*.yaml" ! -perm 600 -ls

# 設定ファイルの機密情報確認
grep -n "password\|secret\|key" config/production/*.yaml | grep -v "environment_variable"
```

---

## 🔍 月次メンテナンス作業

### 毎月第1営業日

#### 1. 包括的システム分析

```bash
# システム全体の健全性レポート
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
import json

manager = get_unified_database_manager()
if manager:
    # 総合ヘルスチェック
    health = manager.run_health_check()

    # システム状況サマリー
    status = manager.get_system_status()

    # データベース情報
    db_info = manager.production_db_manager.get_database_info()

    print('=== 月次システム分析レポート ===')
    print(f'総合ヘルス: {health[\"overall_status\"]}')
    print(f'システム稼働率: {status[\"overall_health\"]}')
    print(f'データベースサイズ: {db_info.get(\"database_size\", \"不明\")}')
    print(f'アクティブ接続数: {db_info.get(\"active_connections\", 0)}')
"
```

#### 2. バックアップ保持期間管理

```bash
# 古いバックアップの削除（30日以上）
find backups/ -name "*.gz" -mtime +30 -delete
find backups/ -name "*.sql" -mtime +7 -delete  # 未圧縮は7日

# バックアップ容量使用状況
du -sh backups/
df -h /opt/daytrading/backups
```

#### 3. パフォーマンス最適化レビュー

```bash
# 月次パフォーマンス統計
python scripts/monthly_performance_report.py

# スロークエリ分析
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager

manager = get_unified_database_manager()
if manager and manager.monitoring_system:
    # 監視システムからスロークエリ情報取得
    metrics = manager.monitoring_system.get_current_metrics()
    if metrics:
        print(f'スロークエリ数: {metrics.get(\"slow_queries_count\", 0)}')
"
```

---

## 🚨 トラブル対応

### システム起動失敗

#### 症状
```
sudo systemctl start daytrading
# ステータス: failed
```

#### 対応手順
1. **ログ確認**:
   ```bash
   sudo journalctl -u daytrading -n 50
   tail -f logs/production/production.log
   ```

2. **設定確認**:
   ```bash
   # 環境変数確認
   cat .env

   # 設定ファイル構文確認
   python -c "import yaml; yaml.safe_load(open('config/production/database.yaml'))"
   ```

3. **データベース接続確認**:
   ```bash
   sudo systemctl status postgresql

   # 接続テスト
   python -c "
   from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
   try:
       manager = get_unified_database_manager()
       health = manager.production_db_manager.health_check()
       print(f'DB接続: {health[\"status\"]}')
   except Exception as e:
       print(f'DB接続エラー: {e}')
   "
   ```

### メモリ不足エラー

#### 症状
```
OutOfMemoryError: Cannot allocate memory
```

#### 対応手順
1. **メモリ使用量確認**:
   ```bash
   free -h
   ps aux | sort -nrk 4 | head -10
   ```

2. **設定調整**:
   ```yaml
   # config/production/database.yaml
   database:
     pool_size: 10        # 20から削減
     max_overflow: 15     # 30から削減
   ```

3. **システム再起動**:
   ```bash
   sudo systemctl restart daytrading
   ```

### バックアップ失敗

#### 症状
```
Backup creation failed: No space left on device
```

#### 対応手順
1. **ディスク容量確認**:
   ```bash
   df -h /opt/daytrading/backups
   ```

2. **古いバックアップ削除**:
   ```bash
   # 30日以上古いバックアップを削除
   find backups/ -name "*.gz" -mtime +30 -delete
   ```

3. **バックアップ再実行**:
   ```bash
   python -c "
   from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
   manager = get_unified_database_manager()
   result = manager.create_backup('manual_recovery')
   print(f'リカバリバックアップ: {result[\"status\"]}')
   "
   ```

---

## 📈 パフォーマンス最適化

### データベース最適化

#### 定期実行推奨（月1回）

```bash
# PostgreSQL最適化メンテナンス
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
from sqlalchemy import text

manager = get_unified_database_manager()
if manager and manager.production_db_manager:
    with manager.production_db_manager.get_session() as session:
        # インデックス再構築
        session.execute(text('REINDEX DATABASE daytrading_prod;'))
        print('インデックス再構築完了')

        # 完全バキューム（メンテナンス時間中のみ）
        session.execute(text('VACUUM FULL ANALYZE;'))
        print('完全バキューム完了')
"
```

### 接続プール調整

#### 高負荷時の設定

```yaml
# config/production/database.yaml （高負荷対応）
database:
  pool_size: 30
  max_overflow: 50
  pool_timeout: 60
  pool_recycle: 1800
```

#### 安定性重視の設定

```yaml
# config/production/database.yaml （安定性重視）
database:
  pool_size: 15
  max_overflow: 25
  pool_timeout: 30
  pool_recycle: 3600
```

---

## 🔒 セキュリティ運用

### 日次セキュリティチェック

```bash
# 不審なログイン試行確認
grep "authentication failed" logs/production/production.log | tail -10

# ファイル改ざん確認（重要ファイル）
find config/ -type f -mtime -1 -ls
find src/day_trade/infrastructure/ -type f -mtime -1 -ls
```

### 週次セキュリティ更新

```bash
# システムパッケージ更新
sudo apt update && sudo apt list --upgradable

# Python依存関係のセキュリティ確認
pip-audit
```

---

## 📋 運用チェックリスト

### 日次チェックリスト

- [ ] システム状態確認（9:00）
- [ ] 夜間バックアップ確認
- [ ] エラーログ確認
- [ ] パフォーマンス状況確認
- [ ] 昼のトレーディング処理確認（13:00）
- [ ] ML分析パフォーマンス確認
- [ ] 日次レポート確認（17:00）
- [ ] ディスク容量確認
- [ ] 接続プール状況確認

### 週次チェックリスト

- [ ] バックアップ整合性テスト（月曜）
- [ ] パフォーマンス分析
- [ ] ログローテーション確認
- [ ] データベース最適化（金曜）
- [ ] 週次セキュリティチェック

### 月次チェックリスト

- [ ] 包括的システム分析
- [ ] バックアップ保持期間管理
- [ ] パフォーマンス最適化レビュー
- [ ] セキュリティ更新適用
- [ ] 設定ファイル見直し

---

## 📞 緊急時連絡手順

### 緊急事態の定義

1. **システム完全停止**
2. **データ損失の可能性**
3. **セキュリティインシデント**
4. **取引処理異常**

### 緊急時対応手順

1. **即座に実行**:
   ```bash
   # システム停止
   sudo systemctl stop daytrading

   # 緊急バックアップ
   python -c "
   from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
   manager = get_unified_database_manager()
   if manager:
       result = manager.create_backup('emergency_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
       print(f'緊急バックアップ: {result}')
   "
   ```

2. **状況記録**:
   - エラーメッセージのスクリーンショット
   - ログファイルのコピー
   - 発生時刻の正確な記録

3. **復旧作業**:
   - 問題の分析と特定
   - 必要に応じてバックアップからの復元
   - システムの段階的復旧

---

**このマニュアルは統合データベース管理システムの安定運用を確保するための包括的なガイドです。定期的な更新と実際の運用経験に基づく改善を継続してください。**

---

*最終更新: 2025年8月18日*  
*ドキュメントバージョン: 1.0.0 (統合データベース管理システム対応)*