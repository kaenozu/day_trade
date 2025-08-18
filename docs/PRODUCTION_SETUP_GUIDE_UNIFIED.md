# Day Trading System 統合本番環境セットアップガイド

**統合データベース管理システム対応版** - 2025年8月版

---

## 📋 概要

このガイドでは、新しく実装された統合データベース管理システムを含む、Day Trading Systemの本番環境セットアップ手順を説明します。

### 🎯 対象システム
- **統合データベース管理システム**: PostgreSQL/SQLite対応、自動バックアップ、リアルタイム監視
- **DDD アーキテクチャ**: ドメイン駆動設計による高品質なコード構造
- **統合エラーハンドリング**: 本番環境に適した包括的エラー管理
- **パフォーマンス最適化**: 高速データ処理と効率的なリソース使用

---

## 🖥️ システム要件

### ハードウェア要件

| 項目 | 最小要件 | 推奨要件 | 備考 |
|------|----------|----------|------|
| CPU | 4コア 2.0GHz | 8コア 3.0GHz+ | トレーディング処理用 |
| メモリ | 8GB | 16GB+ | データ分析・ML処理用 |
| ストレージ | 100GB SSD | 500GB+ NVMe SSD | 高速I/O必要 |
| ネットワーク | 100Mbps | 1Gbps+ | リアルタイム取引用 |

### ソフトウェア要件

```
OS: Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+
Python: 3.9以上
PostgreSQL: 13以上（推奨）または SQLite 3.35+
Redis: 6.0+（キャッシュ用）
Nginx: 1.18+（リバースプロキシ用）
```

---

## 🔧 ステップ1: 基本環境準備

### 1.1 システムユーザー作成

```bash
# Linux環境
sudo useradd -m -s /bin/bash daytrading
sudo usermod -aG sudo daytrading
sudo mkdir -p /opt/daytrading
sudo chown -R daytrading:daytrading /opt/daytrading
sudo su - daytrading
```

### 1.2 必要パッケージインストール

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3 python3-pip python3-venv \
    postgresql postgresql-contrib \
    redis-server nginx \
    git curl wget \
    htop iotop \
    certbot python3-certbot-nginx

# Python仮想環境
cd /opt/daytrading
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 1.3 プロジェクトデプロイ

```bash
# ソースコード取得
git clone https://github.com/your-org/day_trade.git .
pip install -r requirements.txt

# ディレクトリ構造作成
mkdir -p {data,logs,backups,reports,config/production,ssl}
chmod 755 data logs backups reports
chmod 700 config/production ssl
```

---

## 🗄️ ステップ2: データベース設定

### 2.1 PostgreSQL初期設定

```bash
# PostgreSQL初期化
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql

# データベース・ユーザー作成
sudo -u postgres psql << EOF
CREATE USER daytrading_prod WITH PASSWORD 'your_secure_password_here';
CREATE DATABASE daytrading_prod OWNER daytrading_prod;
GRANT ALL PRIVILEGES ON DATABASE daytrading_prod TO daytrading_prod;

-- パフォーマンス用設定
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
\q
EOF
```

### 2.2 本番用データベース設定

```yaml
# config/production/database.yaml
database:
  url: "postgresql://daytrading_prod:${PROD_DB_PASSWORD}@${PROD_DB_HOST:localhost}:5432/daytrading_prod"
  
  # 接続プール設定（本番最適化）
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  pool_pre_ping: true
  
  # SSL設定
  ssl_mode: "require"
  ssl_cert: "${SSL_CERT_PATH}"
  ssl_key: "${SSL_KEY_PATH}"
  ssl_ca: "${SSL_CA_PATH}"
  
  # パフォーマンス設定
  echo: false
  echo_pool: false
  isolation_level: "READ_COMMITTED"
  
  # 接続引数
  connect_args:
    sslmode: "require"
    connect_timeout: 10
    application_name: "DayTradingSystem_Production"

# 自動バックアップ設定
backup:
  enabled: true
  interval_hours: 6
  retention_days: 30
  backup_path: "/opt/daytrading/backups"
  compression: true
  auto_start_scheduler: true
  
  # バックアップ前確認
  backup_before_migration: true
  verification_enabled: true

# リアルタイム監視設定
monitoring:
  enabled: true
  interval_seconds: 30
  metrics_retention_hours: 48
  slow_query_threshold_ms: 1000
  connection_pool_warning_threshold: 0.8
  deadlock_detection: true
  auto_start: true
  
  # アラートルール
  alert_rules:
    - name: "critical_memory_usage"
      metric_name: "memory_usage_mb"
      operator: ">="
      threshold: 12000
      duration_seconds: 300
      severity: "critical"
      enabled: true

# ダッシュボード設定
dashboard:
  enabled: true
  refresh_interval_seconds: 10
  report_path: "/opt/daytrading/reports"
  auto_generate_reports: true
  report_interval_hours: 24

# 復元設定
restore:
  verification_enabled: true
  auto_backup_before_restore: true
  rollback_enabled: true

# 環境別設定
environments:
  production:
    database:
      pool_size: 50
      max_overflow: 100
      url: "postgresql://daytrading_prod:${PROD_DB_PASSWORD}@${PROD_DB_HOST}:5432/daytrading_prod"
      
  staging:
    database:
      pool_size: 10
      max_overflow: 20
      url: "postgresql://daytrading_stage:${STAGE_DB_PASSWORD}@${STAGE_DB_HOST}:5432/daytrading_stage"
```

### 2.3 マイグレーション実行

```bash
# 環境変数設定
export ENVIRONMENT=production
export PROD_DB_PASSWORD="your_secure_password_here"
export PROD_DB_HOST="localhost"

# マイグレーション実行
source venv/bin/activate
alembic upgrade head

# 初期データ投入（必要に応じて）
python scripts/initial_data_setup.py
```

---

## ⚙️ ステップ3: 統合システム初期化

### 3.1 統合データベース管理システム初期化

```python
# scripts/init_production_system.py
#!/usr/bin/env python3
"""
本番環境統合システム初期化スクリプト
"""

import os
import sys
from pathlib import Path

# プロジェクトパス追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from day_trade.infrastructure.database.unified_database_manager import (
    initialize_unified_database_manager
)
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)

def main():
    """本番システム初期化"""
    try:
        # 設定ファイルパス
        config_path = "config/production/database.yaml"
        
        # 統合システム初期化
        logger.info("統合データベース管理システム初期化開始")
        unified_manager = initialize_unified_database_manager(
            config_path=config_path,
            auto_start=True  # 監視・バックアップ自動開始
        )
        
        # システム状態確認
        status = unified_manager.get_system_status()
        logger.info(f"初期化完了 - ヘルス状態: {status['overall_health']}")
        
        # 初期ヘルスチェック
        health_check = unified_manager.run_health_check()
        logger.info(f"ヘルスチェック結果: {health_check['overall_status']}")
        
        # 初回バックアップ作成
        backup_result = unified_manager.create_backup("initial_production")
        logger.info(f"初回バックアップ作成: {backup_result['status']}")
        
        logger.info("本番環境初期化が正常に完了しました")
        return True
        
    except Exception as e:
        logger.error(f"初期化失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### 3.2 systemdサービス設定

```bash
# /etc/systemd/system/daytrading.service
sudo tee /etc/systemd/system/daytrading.service << 'EOF'
[Unit]
Description=Day Trading System - Unified Database Management
After=network.target postgresql.service redis.service
Requires=postgresql.service
Wants=redis.service

[Service]
Type=forking
User=daytrading
Group=daytrading
WorkingDirectory=/opt/daytrading
Environment=PATH=/opt/daytrading/venv/bin
EnvironmentFile=/opt/daytrading/.env

# 初期化スクリプト実行
ExecStartPre=/opt/daytrading/venv/bin/python scripts/init_production_system.py

# メインアプリケーション起動
ExecStart=/opt/daytrading/venv/bin/python -m src.day_trade.main

# 優雅な停止
ExecStop=/bin/kill -TERM $MAINPID
ExecStopPost=/opt/daytrading/venv/bin/python scripts/graceful_shutdown.py

# 自動再起動設定
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30

# セキュリティ設定
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/opt/daytrading/data /opt/daytrading/logs /opt/daytrading/backups /opt/daytrading/reports

# リソース制限
LimitNOFILE=65536
LimitNPROC=32768

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable daytrading
```

---

## 🔒 ステップ4: セキュリティ設定

### 4.1 環境変数設定

```bash
# /opt/daytrading/.env
cat > .env << 'EOF'
# 本番環境設定
ENVIRONMENT=production

# データベース設定
PROD_DB_PASSWORD=your_very_secure_password_here
PROD_DB_HOST=localhost
PROD_DB_PORT=5432
PROD_DB_NAME=daytrading_prod

# セキュリティキー（32文字以上の文字列を使用）
SECRET_KEY=your_very_long_and_random_secret_key_here_must_be_32chars_or_more
JWT_SECRET_KEY=another_very_long_and_random_jwt_secret_key_here

# API設定
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=4
API_MAX_REQUESTS=1000
API_MAX_REQUESTS_JITTER=50

# SSL/TLS設定
SSL_CERT_PATH=/opt/daytrading/ssl/cert.pem
SSL_KEY_PATH=/opt/daytrading/ssl/key.pem
SSL_CA_PATH=/opt/daytrading/ssl/ca.pem

# ログ設定
LOG_LEVEL=INFO
LOG_PATH=/opt/daytrading/logs
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10

# 監視設定
MONITORING_ENABLED=true
DASHBOARD_ENABLED=true
BACKUP_ENABLED=true

# パフォーマンス設定
MAX_POOL_SIZE=50
CACHE_TTL=300
QUERY_TIMEOUT=30

# 取引設定
MAX_DAILY_TRADES=1000
RISK_THRESHOLD=0.02
MAX_POSITION_SIZE=100000
EOF

chmod 600 .env
chown daytrading:daytrading .env
```

### 4.2 SSL証明書設定

```bash
# Let's Encrypt証明書取得
sudo certbot certonly --nginx -d your-trading-domain.com

# 証明書をアプリケーション用にコピー
sudo cp /etc/letsencrypt/live/your-trading-domain.com/fullchain.pem /opt/daytrading/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-trading-domain.com/privkey.pem /opt/daytrading/ssl/key.pem
sudo chown daytrading:daytrading /opt/daytrading/ssl/*.pem
sudo chmod 600 /opt/daytrading/ssl/*.pem

# 自動更新設定
sudo crontab -e
# 毎月1日に証明書更新
0 3 1 * * certbot renew --quiet && systemctl reload daytrading
```

### 4.3 ファイアウォール設定

```bash
# UFW設定
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 必要なポートのみ開放
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 127.0.0.1 to any port 8000  # 内部API

# 特定IPからの管理アクセス（管理者IPに変更）
sudo ufw allow from YOUR_ADMIN_IP to any port 22

sudo ufw --force enable
```

---

## 📊 ステップ5: 監視・運用設定

### 5.1 ログ監視設定

```bash
# /etc/logrotate.d/daytrading
sudo tee /etc/logrotate.d/daytrading << 'EOF'
/opt/daytrading/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 daytrading daytrading
    postrotate
        systemctl reload daytrading > /dev/null 2>&1 || true
    endscript
}
EOF
```

### 5.2 自動監視スクリプト

```bash
# scripts/health_monitor.py
#!/usr/bin/env python3
"""システムヘルス監視スクリプト"""

import sys
import json
import smtplib
from email.mime.text import MimeText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager

def send_alert(subject, message):
    """アラート送信"""
    try:
        msg = MimeText(message)
        msg['Subject'] = f"[Day Trading Alert] {subject}"
        msg['From'] = "system@your-trading-domain.com"
        msg['To'] = "admin@your-trading-domain.com"
        
        server = smtplib.SMTP('localhost')
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"アラート送信失敗: {e}")

def main():
    """ヘルスチェック実行"""
    try:
        manager = get_unified_database_manager()
        if not manager:
            send_alert("システムエラー", "統合データベース管理システムにアクセスできません")
            return False
        
        # ヘルスチェック実行
        health = manager.run_health_check()
        
        # 結果をJSONで出力（ログ用）
        print(json.dumps(health, indent=2))
        
        # 問題がある場合はアラート送信
        if health['overall_status'] != 'healthy':
            issues = '\n'.join(health.get('issues', ['不明なエラー']))
            send_alert(f"システム異常検知", f"ヘルス状態: {health['overall_status']}\n\n問題:\n{issues}")
            return False
        
        return True
        
    except Exception as e:
        send_alert("監視スクリプトエラー", f"ヘルスチェック実行失敗: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### 5.3 cron設定

```bash
# daytrading ユーザーのcrontab
crontab -e
```

```cron
# ヘルスチェック（5分毎）
*/5 * * * * cd /opt/daytrading && /opt/daytrading/venv/bin/python scripts/health_monitor.py >> /opt/daytrading/logs/health_monitor.log 2>&1

# 日次レポート生成（毎日午前2時）
0 2 * * * cd /opt/daytrading && /opt/daytrading/venv/bin/python -c "from src.day_trade.infrastructure.database.dashboard import get_dashboard; dashboard = get_dashboard(); dashboard and dashboard.generate_daily_report()" >> /opt/daytrading/logs/daily_report.log 2>&1

# 週次バックアップ確認（毎週日曜午前3時）
0 3 * * 0 cd /opt/daytrading && /opt/daytrading/venv/bin/python scripts/backup_verification.py >> /opt/daytrading/logs/backup_verification.log 2>&1

# 月次システム最適化（毎月1日午前4時）
0 4 1 * * cd /opt/daytrading && /opt/daytrading/venv/bin/python scripts/monthly_optimization.py >> /opt/daytrading/logs/optimization.log 2>&1
```

---

## 🚀 ステップ6: サービス起動・確認

### 6.1 サービス起動

```bash
# データベース起動確認
sudo systemctl start postgresql
sudo systemctl status postgresql

# Day Trading System起動
sudo systemctl start daytrading
sudo systemctl status daytrading

# ログ確認
sudo journalctl -u daytrading -f
```

### 6.2 動作確認

```bash
# システム状態確認
cd /opt/daytrading
source venv/bin/activate

python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    status = manager.get_system_status()
    print(f'システム状態: {status[\"overall_health\"]}')
    print(f'コンポーネント: {len(status[\"components\"])}個')
else:
    print('システム初期化エラー')
"

# バックアップ機能確認
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    backups = manager.list_backups()
    print(f'バックアップ数: {len(backups)}')
    if backups:
        latest = backups[0]
        print(f'最新バックアップ: {latest[\"filename\"]} ({latest[\"size_mb\"]}MB)')
"

# API動作確認（設定に応じて）
curl -k https://localhost:8000/health || curl http://localhost:8000/health
```

---

## 🔍 トラブルシューティング

### よくある問題と解決方法

#### 1. データベース接続エラー

```bash
# 症状: "password authentication failed"
# 解決: パスワード再設定
sudo -u postgres psql -c "ALTER USER daytrading_prod PASSWORD 'new_secure_password';"

# 環境変数更新
echo "PROD_DB_PASSWORD=new_secure_password" >> .env
sudo systemctl restart daytrading
```

#### 2. メモリ不足

```bash
# 症状: OutOfMemory エラー
# 解決: プール設定調整
# config/production/database.yaml で pool_size を削減
# pool_size: 10 (デフォルト20から)
# max_overflow: 20 (デフォルト30から)

sudo systemctl restart daytrading
```

#### 3. バックアップ失敗

```bash
# 症状: バックアップ作成エラー
# 解決: ディスク容量確認
df -h /opt/daytrading/backups

# 古いバックアップ削除
find /opt/daytrading/backups -name "*.gz" -mtime +30 -delete

# 手動バックアップテスト
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
result = manager.create_backup('manual_test')
print(f'バックアップ結果: {result}')
"
```

#### 4. 監視システム異常

```bash
# 症状: メトリクス収集停止
# 解決: 監視システム再起動
python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager.monitoring_system:
    manager.monitoring_system.stop_monitoring()
    manager.monitoring_system.start_monitoring()
    print('監視システム再起動完了')
"
```

### パフォーマンス最適化

```bash
# PostgreSQL設定最適化
sudo -u postgres psql daytrading_prod << 'EOF'
-- インデックス最適化
REINDEX DATABASE daytrading_prod;

-- 統計情報更新
ANALYZE;

-- 不要データクリーンアップ
VACUUM ANALYZE;
EOF

# システムリソース確認
htop
iotop -a
```

---

## 📞 運用・サポート

### 緊急時対応

1. **システム停止**: `sudo systemctl stop daytrading`
2. **緊急バックアップ**: `/opt/daytrading/scripts/emergency_backup.sh`
3. **ログ確認**: `tail -f /opt/daytrading/logs/*.log`
4. **ヘルスチェック**: `/opt/daytrading/scripts/health_monitor.py`

### 定期メンテナンス

- **日次**: ログ確認、バックアップ状態確認
- **週次**: パフォーマンス分析、バックアップテスト
- **月次**: セキュリティ更新、システム最適化
- **四半期**: 包括的システム監査

### 監視項目

- システムリソース（CPU、メモリ、ディスク）
- データベースパフォーマンス
- バックアップ成功率
- アクティブアラート数
- API応答時間

---

## 📝 最終チェックリスト

- [ ] PostgreSQLデータベース設定完了
- [ ] 統合データベース管理システム初期化完了
- [ ] SSL証明書設定完了
- [ ] ファイアウォール設定完了
- [ ] 自動バックアップ動作確認完了
- [ ] 監視システム動作確認完了
- [ ] ログローテーション設定完了
- [ ] systemdサービス起動確認完了
- [ ] ヘルスチェック正常完了
- [ ] 緊急時対応手順確認完了

---

**この統合セットアップガイドに従って、堅牢で高性能な本番環境Day Trading Systemを構築してください。**

---

*最終更新: 2025年8月18日*  
*ドキュメントバージョン: 2.0.0 (統合データベース管理システム対応)*