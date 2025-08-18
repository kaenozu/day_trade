# Day Trading System API リファレンス

**統合データベース管理システム対応版** - 2025年8月版

---

## 📋 概要

このAPIリファレンスでは、Day Trading Systemの統合データベース管理システムが提供する全てのAPI、メソッド、クラスの詳細な使用方法を説明します。

### 🎯 対象読者
- **開発者**: システム拡張・カスタマイズ担当者
- **統合担当者**: 外部システム連携担当者
- **運用チーム**: API経由での監視・管理担当者
- **テスター**: システムテスト・検証担当者

---

## 🚀 クイックスタート

### 基本的な使用例

```python
# 統合データベース管理システム初期化
from src.day_trade.infrastructure.database.unified_database_manager import (
    initialize_unified_database_manager
)

# システム初期化
manager = initialize_unified_database_manager(
    config_path="config/production/database.yaml",
    auto_start=True
)

# システム状態確認
status = manager.get_system_status()
print(f"システム状態: {status['overall_health']}")

# バックアップ作成
backup_result = manager.create_backup("manual")
print(f"バックアップ作成: {backup_result['status']}")

# 現在のメトリクス取得
metrics = manager.get_current_metrics()
if metrics:
    print(f"CPU使用率: {metrics['cpu_usage']}%")
```

---

## 📚 Core APIs

### 統合データベース管理システム

#### UnifiedDatabaseManager

**主要エントリーポイント**: 統合データベース管理の中心的なクラス

```python
class UnifiedDatabaseManager:
    """統合データベース管理システムのメインクラス"""
```

##### 初期化

```python
def __init__(self, config_path: Optional[str] = None, auto_start: bool = False):
    """
    統合データベースマネージャー初期化

    Args:
        config_path: 設定ファイルパス (デフォルト: "config/database_production.yaml")
        auto_start: 自動開始フラグ (監視・バックアップの自動開始)

    Raises:
        ApplicationError: 初期化失敗時
    """
```

**使用例**:
```python
# 基本初期化
manager = UnifiedDatabaseManager()

# カスタム設定での初期化
manager = UnifiedDatabaseManager(
    config_path="config/custom_database.yaml",
    auto_start=True
)
```

##### システム管理

###### `get_system_status() -> Dict[str, Any]`

システム全体の状態を取得

**戻り値**:
```python
{
    "overall_health": "healthy",  # healthy | degraded | unhealthy
    "initialized": True,
    "components": {
        "production_db": "healthy",
        "backup_system": "healthy",
        "monitoring_system": "healthy",
        "dashboard": "healthy"
    },
    "uptime_seconds": 3600,
    "last_health_check": "2025-08-18T10:30:00Z"
}
```

**使用例**:
```python
status = manager.get_system_status()
if status["overall_health"] != "healthy":
    print("システムに問題があります")
    for component, state in status["components"].items():
        if state != "healthy":
            print(f"問題のあるコンポーネント: {component}")
```

###### `run_health_check() -> Dict[str, Any]`

包括的ヘルスチェック実行

**戻り値**:
```python
{
    "overall_status": "healthy",  # healthy | degraded | critical
    "timestamp": "2025-08-18T10:30:00Z",
    "components": [
        {
            "name": "database_connection",
            "status": "healthy",
            "response_time_ms": 15.2,
            "details": {...}
        },
        {
            "name": "backup_system",
            "status": "healthy",
            "last_backup": "2025-08-18T02:00:00Z"
        }
    ],
    "issues": [],  # 問題がある場合のリスト
    "recommendations": []  # 推奨事項
}
```

**使用例**:
```python
health = manager.run_health_check()
if health["overall_status"] == "critical":
    print("緊急対応が必要:")
    for issue in health["issues"]:
        print(f"- {issue}")
```

###### `shutdown() -> Dict[str, Any]`

システム安全停止

**戻り値**:
```python
{
    "status": "success",  # success | partial | failed
    "stopped_components": ["monitoring", "backup_scheduler"],
    "duration_seconds": 5.2,
    "timestamp": "2025-08-18T10:30:00Z"
}
```

##### バックアップ管理

###### `create_backup(backup_type: str = "manual") -> Dict[str, Any]`

データベースバックアップ作成

**パラメータ**:
- `backup_type`: バックアップの種類 ("manual", "scheduled", "emergency")

**戻り値**:
```python
{
    "status": "success",  # success | failed
    "backup_path": "/opt/daytrading/backups/backup_20250818_103000.sql.gz",
    "backup_id": "backup_20250818_103000",
    "size_mb": 15.7,
    "duration_seconds": 12.3,
    "timestamp": "2025-08-18T10:30:00Z",
    "verification_status": "verified",
    "metadata": {
        "compression": "gzip",
        "database_size_mb": 45.2,
        "table_count": 12
    }
}
```

**使用例**:
```python
# 手動バックアップ
backup = manager.create_backup("manual")
if backup["status"] == "success":
    print(f"バックアップ完了: {backup['backup_path']}")
    print(f"サイズ: {backup['size_mb']}MB")

# 緊急バックアップ
emergency_backup = manager.create_backup("emergency")
```

###### `list_backups(limit: int = 50) -> List[Dict[str, Any]]`

バックアップ一覧取得

**パラメータ**:
- `limit`: 取得する最大件数

**戻り値**:
```python
[
    {
        "filename": "backup_20250818_103000.sql.gz",
        "backup_id": "backup_20250818_103000",
        "size_mb": 15.7,
        "created_at": "2025-08-18T10:30:00Z",
        "backup_type": "manual",
        "status": "verified",
        "retention_until": "2025-09-17T10:30:00Z"
    },
    # ... 他のバックアップ
]
```

**使用例**:
```python
backups = manager.list_backups(limit=10)
print(f"利用可能バックアップ: {len(backups)}件")

for backup in backups:
    print(f"- {backup['filename']}: {backup['size_mb']}MB ({backup['status']})")
```

###### `restore_database(backup_filename: str, dry_run: bool = False) -> Dict[str, Any]`

データベース復元

**パラメータ**:
- `backup_filename`: 復元するバックアップファイル名
- `dry_run`: ドライラン実行フラグ（実際には復元せず、検証のみ）

**戻り値**:
```python
{
    "status": "success",  # success | failed
    "backup_filename": "backup_20250818_103000.sql.gz",
    "duration_seconds": 25.7,
    "restored_tables": 12,
    "restored_records": 15420,
    "pre_restore_backup": "backup_pre_restore_20250818_104500.sql.gz",
    "verification_passed": True,
    "dry_run": False
}
```

**使用例**:
```python
# ドライラン実行
dry_result = manager.restore_database("backup_20250818_103000.sql.gz", dry_run=True)
if dry_result["status"] == "success":
    print("復元可能なバックアップです")

    # 実際の復元実行
    restore_result = manager.restore_database("backup_20250818_103000.sql.gz")
    if restore_result["status"] == "success":
        print(f"復元完了: {restore_result['restored_records']}レコード")
```

##### 監視・メトリクス

###### `get_current_metrics() -> Optional[Dict[str, Any]]`

現在のシステムメトリクス取得

**戻り値**:
```python
{
    "timestamp": "2025-08-18T10:30:00Z",
    "active_connections": 5,
    "max_connections": 20,
    "connection_pool_usage": 0.25,
    "queries_per_second": 12.5,
    "average_query_time": 0.015,
    "slow_queries_count": 0,
    "cpu_usage": 25.3,
    "memory_usage_mb": 512.7,
    "disk_usage_percent": 45.2,
    "disk_io_read_mb": 15.2,
    "disk_io_write_mb": 8.7,
    "connection_errors": 0,
    "query_errors": 0,
    "deadlocks": 0,
    "database_size_mb": 45.2,
    "table_count": 12,
    "index_count": 18
}
```

**使用例**:
```python
metrics = manager.get_current_metrics()
if metrics:
    # CPU使用率監視
    if metrics["cpu_usage"] > 80:
        print(f"高CPU使用率: {metrics['cpu_usage']}%")

    # 接続プール監視
    if metrics["connection_pool_usage"] > 0.8:
        print("接続プール使用率が高い")

    # スロークエリ監視
    if metrics["slow_queries_count"] > 0:
        print(f"スロークエリ検出: {metrics['slow_queries_count']}件")
```

###### `get_active_alerts() -> List[Dict[str, Any]]`

アクティブアラート取得

**戻り値**:
```python
[
    {
        "id": "high_cpu_usage_cpu_usage",
        "rule_name": "high_cpu_usage",
        "metric_name": "cpu_usage",
        "current_value": 85.2,
        "threshold": 80.0,
        "severity": "warning",
        "message": "CPU使用率が高い: 現在値=85.20, 閾値=80.0",
        "timestamp": "2025-08-18T10:25:00Z",
        "resolved": False
    }
]
```

**使用例**:
```python
alerts = manager.get_active_alerts()
if alerts:
    print(f"アクティブアラート: {len(alerts)}件")

    for alert in alerts:
        print(f"[{alert['severity']}] {alert['message']}")

        # Critical アラートの場合は緊急対応
        if alert["severity"] == "critical":
            print("緊急対応が必要です")
```

##### ダッシュボード・レポート

###### `get_dashboard_data() -> Dict[str, Any]`

ダッシュボードデータ取得

**戻り値**:
```python
{
    "system_overview": {
        "status": "healthy",
        "uptime_hours": 24.5,
        "total_transactions": 1542,
        "success_rate": 99.2
    },
    "performance_metrics": {
        "avg_response_time_ms": 15.2,
        "peak_cpu_usage": 65.3,
        "peak_memory_usage_mb": 892.1
    },
    "database_stats": {
        "size_mb": 45.2,
        "backup_count": 12,
        "last_backup": "2025-08-18T02:00:00Z"
    },
    "alerts_summary": {
        "active_count": 0,
        "resolved_today": 2,
        "critical_count": 0
    },
    "generated_at": "2025-08-18T10:30:00Z"
}
```

**使用例**:
```python
dashboard = manager.get_dashboard_data()
print(f"システム状態: {dashboard['system_overview']['status']}")
print(f"稼働時間: {dashboard['system_overview']['uptime_hours']}時間")
print(f"成功率: {dashboard['system_overview']['success_rate']}%")
```

###### `generate_report(report_type: str = "daily") -> Dict[str, Any]`

システムレポート生成

**パラメータ**:
- `report_type`: レポートタイプ ("daily", "weekly", "monthly", "custom")

**戻り値**:
```python
{
    "status": "success",
    "report_type": "daily",
    "file_path": "/opt/daytrading/reports/daily_report_20250818.pdf",
    "period": {
        "start": "2025-08-17T00:00:00Z",
        "end": "2025-08-18T00:00:00Z"
    },
    "summary": {
        "total_transactions": 1542,
        "average_response_time": 15.2,
        "system_availability": 99.8,
        "backup_count": 1,
        "alert_count": 3
    },
    "generated_at": "2025-08-18T10:30:00Z"
}
```

**使用例**:
```python
# 日次レポート生成
daily_report = manager.generate_report("daily")
if daily_report["status"] == "success":
    print(f"レポート生成完了: {daily_report['file_path']}")
    print(f"取引数: {daily_report['summary']['total_transactions']}")

# 週次レポート生成
weekly_report = manager.generate_report("weekly")
```

---

## 🗄️ Database APIs

### ProductionDatabaseManager

**データベース固有操作**: PostgreSQL/SQLite本番環境管理

```python
class ProductionDatabaseManager:
    """本番環境データベース管理クラス"""
```

##### 初期化・接続

###### `__init__(config_path: Optional[str] = None)`

```python
def __init__(self, config_path: Optional[str] = None):
    """
    本番データベースマネージャー初期化

    Args:
        config_path: データベース設定ファイルパス
    """
```

###### `initialize() -> None`

データベースマネージャー初期化実行

```python
def initialize() -> None:
    """
    データベース接続プール・マイグレーション管理初期化

    Raises:
        ProductionDatabaseError: 初期化失敗時
    """
```

###### `get_session()`

データベースセッション取得（コンテキストマネージャー）

```python
@contextmanager
def get_session():
    """
    SQLAlchemyセッション取得

    Yields:
        Session: データベースセッション

    Usage:
        with manager.get_session() as session:
            result = session.execute(text("SELECT 1"))
    """
```

**使用例**:
```python
db_manager = ProductionDatabaseManager()
db_manager.initialize()

# セッション使用例
with db_manager.get_session() as session:
    # クエリ実行
    result = session.execute(text("SELECT COUNT(*) FROM trades"))
    count = result.scalar()
    print(f"取引数: {count}")

    # データ挿入
    session.execute(text("""
        INSERT INTO trades (symbol, quantity, price, timestamp)
        VALUES (:symbol, :quantity, :price, :timestamp)
    """), {
        "symbol": "AAPL",
        "quantity": 100,
        "price": 150.25,
        "timestamp": datetime.now()
    })
    # セッション終了時に自動コミット
```

##### データベース情報

###### `get_database_info() -> Dict[str, Any]`

データベース詳細情報取得

**戻り値**:
```python
{
    "database_type": "PostgreSQL",
    "version": "PostgreSQL 13.7 on x86_64-pc-linux-gnu",
    "database_size": "15 MB",
    "active_connections": 5,
    "environment": "production",
    "pool_status": {
        "size": 20,
        "checked_out": 3,
        "overflow": 0,
        "utilization": 0.15
    }
}
```

###### `health_check() -> Dict[str, Any]`

データベースヘルスチェック

**戻り値**:
```python
{
    "status": "healthy",  # healthy | unhealthy
    "response_time_ms": 15.2,
    "pool_status": {
        "size": 20,
        "checked_out": 3,
        "overflow": 0,
        "utilization": 0.15
    },
    "slow_queries_count": 0,
    "last_check": "2025-08-18T10:30:00Z"
}
```

##### マイグレーション

###### `run_migrations() -> Dict[str, Any]`

データベースマイグレーション実行

**戻り値**:
```python
{
    "success": True,
    "duration_seconds": 5.2,
    "from_revision": "abc123",
    "to_revision": "def456",
    "applied_revisions": ["def456"]
}
```

**使用例**:
```python
# マイグレーション実行
migration_result = db_manager.run_migrations()
if migration_result["success"]:
    print(f"マイグレーション完了: {migration_result['duration_seconds']}秒")
    print(f"適用リビジョン: {migration_result['applied_revisions']}")
```

---

## 📊 Monitoring APIs

### DatabaseMonitoringSystem

**リアルタイム監視**: メトリクス収集・アラート管理

```python
class DatabaseMonitoringSystem:
    """データベース監視システム"""
```

##### 初期化・制御

###### `__init__(engine: Engine, config: Dict[str, Any])`

```python
def __init__(self, engine: Engine, config: Dict[str, Any]):
    """
    監視システム初期化

    Args:
        engine: SQLAlchemyエンジン
        config: 監視設定
    """
```

###### `start_monitoring() -> None`

監視開始

```python
def start_monitoring() -> None:
    """
    バックグラウンド監視スレッド開始
    設定された間隔でメトリクス収集・アラートチェック実行
    """
```

###### `stop_monitoring() -> None`

監視停止

```python
def stop_monitoring() -> None:
    """
    監視スレッド安全停止
    """
```

**使用例**:
```python
from src.day_trade.infrastructure.database.monitoring_system import (
    initialize_monitoring_system
)

# 監視システム初期化
monitoring = initialize_monitoring_system(engine, config)

# 監視開始
monitoring.start_monitoring()
print("監視システム開始")

# しばらく稼働...
time.sleep(60)

# 監視停止
monitoring.stop_monitoring()
print("監視システム停止")
```

##### メトリクス収集

###### `collect_metrics() -> Optional[DatabaseMetrics]`

現在のメトリクス収集

**戻り値**:
```python
DatabaseMetrics(
    timestamp=datetime(2025, 8, 18, 10, 30, 0),
    active_connections=5,
    max_connections=20,
    connection_pool_usage=0.25,
    queries_per_second=12.5,
    average_query_time=0.015,
    slow_queries_count=0,
    cpu_usage=25.3,
    memory_usage_mb=512.7,
    disk_usage_percent=45.2,
    disk_io_read_mb=15.2,
    disk_io_write_mb=8.7,
    connection_errors=0,
    query_errors=0,
    deadlocks=0,
    database_size_mb=45.2,
    table_count=12,
    index_count=18
)
```

###### `get_metrics_history(hours: int = 1) -> List[Dict[str, Any]]`

メトリクス履歴取得

**パラメータ**:
- `hours`: 取得する過去時間数

**戻り値**: メトリクス辞書のリスト

**使用例**:
```python
# 過去1時間のメトリクス取得
recent_metrics = monitoring.get_metrics_history(hours=1)
print(f"過去1時間のデータポイント: {len(recent_metrics)}件")

# CPU使用率の推移分析
cpu_usage_values = [m["cpu_usage"] for m in recent_metrics]
avg_cpu = sum(cpu_usage_values) / len(cpu_usage_values)
max_cpu = max(cpu_usage_values)
print(f"平均CPU使用率: {avg_cpu:.1f}%")
print(f"最大CPU使用率: {max_cpu:.1f}%")
```

##### アラート管理

###### `check_alerts(metrics: DatabaseMetrics) -> List[Alert]`

アラートチェック実行

**パラメータ**:
- `metrics`: チェック対象のメトリクス

**戻り値**: 新規発生アラートのリスト

###### `get_active_alerts() -> List[Dict[str, Any]]`

アクティブアラート取得

**戻り値**:
```python
[
    {
        "id": "high_cpu_usage_cpu_usage",
        "rule_name": "high_cpu_usage",
        "metric_name": "cpu_usage",
        "current_value": 85.2,
        "threshold": 80.0,
        "severity": "warning",
        "message": "CPU使用率が高い: 現在値=85.20, 閾値=80.0",
        "timestamp": "2025-08-18T10:25:00Z",
        "resolved": False
    }
]
```

###### `get_alert_statistics() -> Dict[str, Any]`

アラート統計情報取得

**戻り値**:
```python
{
    "total_alerts": 15,
    "active_alerts": 2,
    "resolved_alerts": 13,
    "critical_count": 1,
    "warning_count": 12,
    "info_count": 2
}
```

###### `add_alert_callback(callback: Callable[[Alert], None]) -> None`

アラート通知コールバック追加

**パラメータ**:
- `callback`: アラート発生時に呼び出される関数

**使用例**:
```python
def email_alert_handler(alert):
    """メール通知ハンドラー"""
    if alert.severity in ["critical", "high"]:
        send_email(
            to="admin@company.com",
            subject=f"[ALERT] {alert.rule_name}",
            body=alert.message
        )

def slack_alert_handler(alert):
    """Slack通知ハンドラー"""
    slack_client.send_message(
        channel="#alerts",
        text=f"🚨 {alert.message}"
    )

# コールバック追加
monitoring.add_alert_callback(email_alert_handler)
monitoring.add_alert_callback(slack_alert_handler)
```

##### 監視状態

###### `get_monitoring_status() -> Dict[str, Any]`

監視システム状態取得

**戻り値**:
```python
{
    "enabled": True,
    "running": True,
    "interval_seconds": 30,
    "metrics_count": 120,
    "active_alerts_count": 2,
    "alert_history_count": 15,
    "alert_rules_count": 8,
    "last_collection": "2025-08-18T10:30:00Z"
}
```

---

## 💾 Backup APIs

### BackupManager

**バックアップ管理**: 自動・手動バックアップ機能

```python
class BackupManager:
    """バックアップ管理システム"""
```

##### 初期化・設定

###### `__init__(engine: Engine, config: Dict[str, Any])`

```python
def __init__(self, engine: Engine, config: Dict[str, Any]):
    """
    バックアップマネージャー初期化

    Args:
        engine: データベースエンジン
        config: バックアップ設定
    """
```

###### `start_scheduler() -> None`

自動バックアップスケジューラー開始

```python
def start_scheduler() -> None:
    """
    設定された間隔で自動バックアップ実行
    """
```

###### `stop_scheduler() -> None`

自動バックアップスケジューラー停止

**使用例**:
```python
from src.day_trade.infrastructure.database.backup_manager import (
    initialize_backup_manager
)

# バックアップマネージャー初期化
backup_manager = initialize_backup_manager(engine, config)

# 自動バックアップ開始
backup_manager.start_scheduler()
print("自動バックアップ開始")

# 必要に応じて停止
backup_manager.stop_scheduler()
```

##### バックアップ実行

###### `create_backup(backup_type: str = "manual") -> Dict[str, Any]`

バックアップ作成

**パラメータ**:
- `backup_type`: バックアップタイプ ("manual", "scheduled", "emergency")

**戻り値**:
```python
{
    "status": "success",
    "backup_id": "backup_20250818_103000",
    "backup_path": "/opt/daytrading/backups/backup_20250818_103000.sql.gz",
    "size_mb": 15.7,
    "duration_seconds": 12.3,
    "compression_ratio": 0.35,
    "verification_status": "verified",
    "metadata": {
        "database_type": "postgresql",
        "database_size_mb": 45.2,
        "table_count": 12,
        "record_count": 15420
    }
}
```

###### `verify_backup(backup_filename: str) -> Dict[str, Any]`

バックアップ整合性確認

**パラメータ**:
- `backup_filename`: 確認するバックアップファイル名

**戻り値**:
```python
{
    "status": "verified",  # verified | corrupted | not_found
    "backup_filename": "backup_20250818_103000.sql.gz",
    "file_exists": True,
    "file_size_mb": 15.7,
    "checksum_valid": True,
    "compression_valid": True,
    "readable": True,
    "verification_time": "2025-08-18T10:30:00Z"
}
```

**使用例**:
```python
# バックアップ作成
backup_result = backup_manager.create_backup("manual")
if backup_result["status"] == "success":
    backup_file = backup_result["backup_id"] + ".sql.gz"

    # 整合性確認
    verification = backup_manager.verify_backup(backup_file)
    if verification["status"] == "verified":
        print("バックアップ整合性OK")
    else:
        print("バックアップに問題があります")
```

##### バックアップ管理

###### `list_backups(limit: int = 50) -> List[Dict[str, Any]]`

バックアップ一覧取得

###### `delete_backup(backup_filename: str) -> Dict[str, Any]`

バックアップ削除

**パラメータ**:
- `backup_filename`: 削除するバックアップファイル名

**戻り値**:
```python
{
    "status": "success",  # success | failed | not_found
    "backup_filename": "backup_20250818_103000.sql.gz",
    "deleted_size_mb": 15.7,
    "timestamp": "2025-08-18T10:30:00Z"
}
```

###### `cleanup_old_backups(retention_days: int = 30) -> Dict[str, Any]`

古いバックアップ削除

**パラメータ**:
- `retention_days`: 保持日数

**戻り値**:
```python
{
    "status": "success",
    "deleted_count": 5,
    "freed_space_mb": 78.5,
    "retention_days": 30,
    "remaining_backups": 12
}
```

**使用例**:
```python
# 古いバックアップクリーンアップ（30日以上）
cleanup_result = backup_manager.cleanup_old_backups(retention_days=30)
print(f"削除したバックアップ: {cleanup_result['deleted_count']}件")
print(f"解放した容量: {cleanup_result['freed_space_mb']}MB")
```

---

## 🔧 Error Handling APIs

### 統合エラーハンドリングシステム

#### エラークラス階層

```python
# ベースエラークラス
class ApplicationError(Exception):
    """アプリケーション固有エラーの基底クラス"""

class DataAccessError(ApplicationError):
    """データアクセス関連エラー"""

class SystemError(ApplicationError):
    """システム関連エラー"""

class SecurityError(ApplicationError):
    """セキュリティ関連エラー"""

class ValidationError(ApplicationError):
    """バリデーション関連エラー"""
```

#### エラーバウンダリデコレーター

###### `@error_boundary(component_name: str, operation_name: str, suppress_errors: bool = False)`

関数・メソッドのエラーハンドリング

**パラメータ**:
- `component_name`: コンポーネント名
- `operation_name`: 操作名
- `suppress_errors`: エラー抑制フラグ

**使用例**:
```python
from src.day_trade.core.error_handling.unified_error_system import (
    error_boundary, DataAccessError
)

@error_boundary(
    component_name="trading_service",
    operation_name="execute_trade",
    suppress_errors=False
)
def execute_trade(symbol: str, quantity: int, price: float) -> Dict[str, Any]:
    """取引実行"""
    try:
        # 取引ロジック
        result = trading_api.execute(symbol, quantity, price)
        return {"status": "success", "trade_id": result.id}
    except Exception as e:
        raise DataAccessError(f"取引実行失敗: {e}") from e

# 使用
try:
    trade_result = execute_trade("AAPL", 100, 150.25)
    print(f"取引完了: {trade_result['trade_id']}")
except DataAccessError as e:
    print(f"取引エラー: {e}")
```

#### グローバルエラーハンドラー

###### `global_error_handler.handle_error(error: Exception, context: Dict[str, Any])`

グローバルエラー処理

**使用例**:
```python
from src.day_trade.core.error_handling.unified_error_system import (
    global_error_handler
)

try:
    # 危険な操作
    risky_operation()
except Exception as e:
    # グローバルハンドラーで処理
    global_error_handler.handle_error(e, {
        "operation": "risky_operation",
        "user_id": "user123",
        "timestamp": datetime.now().isoformat()
    })
```

---

## 🔒 Security APIs

### セキュリティ監視システム

#### SecurityMonitor

```python
class SecurityMonitor:
    """セキュリティ監視システム"""
```

###### `record_failed_login(ip_address: str, user_id: Optional[str] = None)`

ログイン失敗記録

###### `is_ip_blocked(ip_address: str) -> bool`

IPブロック状態確認

###### `get_security_summary() -> Dict[str, Any]`

セキュリティサマリー取得

**戻り値**:
```python
{
    "blocked_ips_count": 3,
    "recent_events_count": 15,
    "event_breakdown": {
        "failed_login": 8,
        "suspicious_request": 4,
        "ip_blocked": 3
    },
    "failed_attempts_count": 12,
    "last_update": "2025-08-18T10:30:00Z"
}
```

**使用例**:
```python
from src.day_trade.core.security.security_monitor import get_security_monitor

security = get_security_monitor()

# ログイン失敗記録
security.record_failed_login("192.168.1.100", "user123")

# IPブロック確認
if security.is_ip_blocked("192.168.1.100"):
    print("このIPはブロックされています")

# セキュリティサマリー
summary = security.get_security_summary()
print(f"ブロック済みIP: {summary['blocked_ips_count']}件")
```

#### DataEncryption

```python
class DataEncryption:
    """データ暗号化クラス"""
```

###### `encrypt(data: Union[str, bytes]) -> str`

データ暗号化

###### `decrypt(encrypted_data: str) -> str`

データ復号化

###### `encrypt_file(file_path: str, output_path: str = None) -> str`

ファイル暗号化

###### `decrypt_file(encrypted_file_path: str, output_path: str = None) -> str`

ファイル復号化

**使用例**:
```python
from src.day_trade.core.security.data_encryption import get_data_encryption

encryption = get_data_encryption()

# データ暗号化
sensitive_data = "機密情報"
encrypted = encryption.encrypt(sensitive_data)
print(f"暗号化データ: {encrypted}")

# データ復号化
decrypted = encryption.decrypt(encrypted)
print(f"復号化データ: {decrypted}")

# ファイル暗号化
encrypted_file = encryption.encrypt_file("sensitive_file.txt")
print(f"暗号化ファイル: {encrypted_file}")
```

---

## 🧪 Testing APIs

### テストユーティリティ

#### DatabaseTestUtils

```python
class DatabaseTestUtils:
    """データベーステスト用ユーティリティ"""
```

###### `create_test_database() -> str`

テスト用データベース作成

###### `cleanup_test_database(db_name: str)`

テスト用データベース削除

###### `load_test_data(db_session: Session, data_file: str)`

テストデータ投入

**使用例**:
```python
from src.day_trade.testing.database_test_utils import DatabaseTestUtils

def test_trading_operations():
    test_utils = DatabaseTestUtils()

    # テストDB作成
    test_db = test_utils.create_test_database()

    try:
        # テストデータ投入
        with get_test_session(test_db) as session:
            test_utils.load_test_data(session, "test_trades.json")

            # テスト実行
            result = execute_test_trade(session, "AAPL", 100, 150.25)
            assert result["status"] == "success"

    finally:
        # クリーンアップ
        test_utils.cleanup_test_database(test_db)
```

---

## 📋 Configuration APIs

### 設定管理

#### EnvironmentConfig

```python
class EnvironmentConfig:
    """環境設定管理"""
```

###### `get_database_config() -> Dict[str, Any]`

データベース設定取得

###### `get_monitoring_config() -> Dict[str, Any]`

監視設定取得

###### `get_security_config() -> Dict[str, Any]`

セキュリティ設定取得

**使用例**:
```python
from src.day_trade.config.environment_config import get_environment_config

config = get_environment_config()

# データベース設定
db_config = config.get_database_config()
print(f"データベースURL: {db_config['url']}")

# 監視設定
monitoring_config = config.get_monitoring_config()
print(f"監視間隔: {monitoring_config['interval_seconds']}秒")
```

---

## 🔍 Utility Functions

### よく使用されるユーティリティ関数

#### 初期化関数

```python
# 統合システム初期化
def initialize_unified_database_manager(
    config_path: str = None,
    auto_start: bool = False
) -> UnifiedDatabaseManager:
    """統合データベース管理システム初期化"""

# 各コンポーネント初期化
def initialize_production_database(config_path: str = None) -> ProductionDatabaseManager:
def initialize_monitoring_system(engine: Engine, config: Dict) -> DatabaseMonitoringSystem:
def initialize_backup_manager(engine: Engine, config: Dict) -> BackupManager:
```

#### 設定取得関数

```python
def get_unified_database_manager() -> Optional[UnifiedDatabaseManager]:
    """統合データベース管理システム取得（シングルトン）"""

def get_production_database_manager() -> ProductionDatabaseManager:
    """本番データベースマネージャー取得"""

def get_monitoring_system() -> Optional[DatabaseMonitoringSystem]:
    """監視システム取得"""

def get_security_monitor() -> SecurityMonitor:
    """セキュリティモニター取得"""
```

---

## 📊 Response Formats

### 標準レスポンス形式

#### 成功レスポンス

```python
{
    "status": "success",
    "data": { /* 結果データ */ },
    "timestamp": "2025-08-18T10:30:00Z",
    "duration_seconds": 0.123
}
```

#### エラーレスポンス

```python
{
    "status": "error",
    "error": {
        "type": "DataAccessError",
        "message": "データベース接続エラー",
        "code": "DB_CONNECTION_FAILED",
        "details": { /* エラー詳細 */ }
    },
    "timestamp": "2025-08-18T10:30:00Z"
}
```

#### 部分成功レスポンス

```python
{
    "status": "partial",
    "data": { /* 成功した部分のデータ */ },
    "warnings": [
        "一部の操作が失敗しました",
        "バックアップ作成に時間がかかりました"
    ],
    "timestamp": "2025-08-18T10:30:00Z"
}
```

---

## 🚀 Advanced Usage Examples

### 完全な統合例

```python
#!/usr/bin/env python3
"""
Day Trading System 統合使用例
"""

import time
from datetime import datetime
from src.day_trade.infrastructure.database.unified_database_manager import (
    initialize_unified_database_manager
)

def main():
    # 1. システム初期化
    print("統合データベース管理システム初期化...")
    manager = initialize_unified_database_manager(
        config_path="config/production/database.yaml",
        auto_start=True
    )

    # 2. システム状態確認
    status = manager.get_system_status()
    print(f"システム状態: {status['overall_health']}")

    if status["overall_health"] != "healthy":
        print("システムに問題があります。終了します。")
        return

    # 3. ヘルスチェック実行
    health = manager.run_health_check()
    print(f"ヘルスチェック: {health['overall_status']}")

    # 4. 現在のメトリクス確認
    metrics = manager.get_current_metrics()
    if metrics:
        print(f"CPU使用率: {metrics['cpu_usage']}%")
        print(f"メモリ使用量: {metrics['memory_usage_mb']}MB")
        print(f"アクティブ接続: {metrics['active_connections']}")

    # 5. アクティブアラート確認
    alerts = manager.get_active_alerts()
    if alerts:
        print(f"アクティブアラート: {len(alerts)}件")
        for alert in alerts:
            print(f"- [{alert['severity']}] {alert['message']}")
    else:
        print("アクティブアラートなし")

    # 6. バックアップ作成
    print("バックアップ作成中...")
    backup_result = manager.create_backup("api_example")
    if backup_result["status"] == "success":
        print(f"バックアップ完了: {backup_result['backup_path']}")
        print(f"サイズ: {backup_result['size_mb']}MB")

    # 7. バックアップ一覧確認
    backups = manager.list_backups(limit=5)
    print(f"最新バックアップ {len(backups)}件:")
    for backup in backups:
        print(f"- {backup['filename']}: {backup['size_mb']}MB")

    # 8. ダッシュボードデータ取得
    dashboard = manager.get_dashboard_data()
    if dashboard:
        print("システム概要:")
        overview = dashboard["system_overview"]
        print(f"- 状態: {overview['status']}")
        print(f"- 稼働時間: {overview['uptime_hours']}時間")
        print(f"- 成功率: {overview['success_rate']}%")

    # 9. レポート生成
    print("日次レポート生成中...")
    report = manager.generate_report("daily")
    if report["status"] == "success":
        print(f"レポート生成完了: {report['file_path']}")

    # 10. データベース直接操作例
    if manager.production_db_manager:
        with manager.production_db_manager.get_session() as session:
            from sqlalchemy import text

            # サンプルクエリ実行
            result = session.execute(text("SELECT 'API Test Success' as message"))
            message = result.scalar()
            print(f"データベース接続テスト: {message}")

    print("統合システム使用例完了")

if __name__ == "__main__":
    main()
```

### 監視・アラート設定例

```python
"""
カスタム監視・アラート設定例
"""

from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager

def setup_custom_monitoring():
    manager = get_unified_database_manager()
    if not manager or not manager.monitoring_system:
        print("監視システムが利用できません")
        return

    monitoring = manager.monitoring_system

    # カスタムアラート通知設定
    def custom_alert_handler(alert):
        """カスタムアラートハンドラー"""
        print(f"🚨 カスタムアラート: {alert.message}")

        # Critical の場合は緊急対応
        if alert.severity == "critical":
            print("緊急対応が必要です！")
            # 緊急バックアップ作成
            emergency_backup = manager.create_backup("emergency_alert")
            print(f"緊急バックアップ作成: {emergency_backup['status']}")

        # Slack通知（実装例）
        # send_slack_notification(alert)

        # メール通知（実装例）
        # send_email_notification(alert)

    # アラートコールバック追加
    monitoring.add_alert_callback(custom_alert_handler)

    # 監視開始
    monitoring.start_monitoring()
    print("カスタム監視システム開始")

if __name__ == "__main__":
    setup_custom_monitoring()
```

---

## 📚 SDK Integration

### Python SDK 統合例

```python
"""
Day Trading System Python SDK
"""

class DayTradingSDK:
    """Day Trading System SDK"""

    def __init__(self, config_path: str = None):
        self.manager = initialize_unified_database_manager(
            config_path=config_path,
            auto_start=True
        )

    def health_check(self) -> bool:
        """システムヘルスチェック"""
        health = self.manager.run_health_check()
        return health["overall_status"] == "healthy"

    def backup(self, backup_type: str = "manual") -> str:
        """バックアップ作成"""
        result = self.manager.create_backup(backup_type)
        if result["status"] == "success":
            return result["backup_path"]
        else:
            raise Exception(f"バックアップ失敗: {result.get('error')}")

    def get_metrics(self) -> dict:
        """現在のメトリクス取得"""
        return self.manager.get_current_metrics()

    def get_alerts(self) -> list:
        """アクティブアラート取得"""
        return self.manager.get_active_alerts()

# SDK使用例
sdk = DayTradingSDK("config/production/database.yaml")

# ヘルスチェック
if sdk.health_check():
    print("システム正常")

    # バックアップ作成
    backup_path = sdk.backup("sdk_test")
    print(f"バックアップ作成: {backup_path}")

    # メトリクス取得
    metrics = sdk.get_metrics()
    print(f"CPU: {metrics['cpu_usage']}%")
```

---

**このAPIリファレンスを参考に、Day Trading Systemの統合データベース管理システムを効果的に活用してください。各APIの詳細な使用方法と実用的な例を通じて、システムの全機能を最大限に活用できます。**

---

*最終更新: 2025年8月18日*  
*ドキュメントバージョン: 1.0.0 (統合データベース管理システム対応)*