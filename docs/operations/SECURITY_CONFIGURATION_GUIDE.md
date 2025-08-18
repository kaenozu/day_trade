# Day Trading System セキュリティ設定ガイド

**統合データベース管理システム対応版** - 2025年8月版

---

## 📋 概要

このガイドでは、Day Trading Systemの統合データベース管理システムにおけるセキュリティ設定、脅威対策、監査機能の実装手順を詳細に説明します。

### 🎯 セキュリティ目標

| 目標 | 要件 | 実装状況 |
|------|------|----------|
| **機密性** | データ暗号化、アクセス制御 | ✅ 実装済み |
| **完全性** | データ整合性、改ざん防止 | ✅ 実装済み |
| **可用性** | 冗長化、自動復旧 | ✅ 実装済み |
| **監査性** | ログ記録、証跡管理 | ✅ 実装済み |
| **認証** | 多要素認証、強力な認証 | 🔄 設定要 |
| **認可** | 最小権限原則、役割ベース | ✅ 実装済み |

---

## 🔒 基本セキュリティ設定

### システムレベルセキュリティ

#### 1. ユーザー・権限管理

```bash
# 専用システムユーザー作成
sudo useradd -m -s /bin/bash daytrading
sudo usermod -aG sudo daytrading

# セキュアなホームディレクトリ権限
sudo chmod 750 /home/daytrading
sudo chown daytrading:daytrading /home/daytrading

# アプリケーションディレクトリセキュリティ
sudo mkdir -p /opt/daytrading
sudo chown -R daytrading:daytrading /opt/daytrading
sudo chmod 755 /opt/daytrading
sudo chmod 700 /opt/daytrading/config
sudo chmod 700 /opt/daytrading/ssl
sudo chmod 750 /opt/daytrading/data
sudo chmod 750 /opt/daytrading/logs
sudo chmod 755 /opt/daytrading/backups
```

#### 2. ファイアウォール設定

```bash
# UFW基本設定
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 必要ポートのみ開放
sudo ufw allow ssh
sudo ufw allow 80/tcp   # HTTP (リバースプロキシ)
sudo ufw allow 443/tcp  # HTTPS

# ローカルサービス用ポート（内部アクセスのみ）
sudo ufw allow from 127.0.0.1 to any port 8000  # API
sudo ufw allow from 127.0.0.1 to any port 5432  # PostgreSQL

# 管理者IPからのSSHアクセス（IPアドレスを適切に設定）
sudo ufw allow from YOUR_ADMIN_IP to any port 22

# ファイアウォール有効化
sudo ufw --force enable

# ルール確認
sudo ufw status verbose
```

#### 3. SSL/TLS証明書設定

```bash
# Let's Encrypt証明書取得
sudo certbot certonly --nginx -d your-trading-domain.com

# 証明書を統合システム用にコピー
sudo mkdir -p /opt/daytrading/ssl
sudo cp /etc/letsencrypt/live/your-trading-domain.com/fullchain.pem \
    /opt/daytrading/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-trading-domain.com/privkey.pem \
    /opt/daytrading/ssl/key.pem

# 証明書権限設定
sudo chown daytrading:daytrading /opt/daytrading/ssl/*.pem
sudo chmod 600 /opt/daytrading/ssl/*.pem

# 自動更新設定
sudo crontab -e
# 毎月1日午前3時に証明書更新
0 3 1 * * certbot renew --quiet && systemctl reload daytrading nginx
```

---

## 🛡️ データベースセキュリティ

### PostgreSQL セキュリティ設定

#### 1. 認証・アクセス制御

```bash
# PostgreSQL設定ファイル編集
sudo vim /etc/postgresql/13/main/postgresql.conf

# 以下の設定を適用:
listen_addresses = 'localhost'          # ローカルのみアクセス
port = 5432
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
wal_buffers = 16MB
log_statement = 'mod'                   # 変更SQLをログ記録
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
logging_collector = on
log_directory = '/var/log/postgresql'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
```

```bash
# pg_hba.conf設定（認証方法）
sudo vim /etc/postgresql/13/main/pg_hba.conf

# セキュアな認証設定:
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# ローカル接続（Unix socket）
local   all             postgres                                peer
local   daytrading_prod daytrading_prod                         md5

# IPv4ローカル接続
host    daytrading_prod daytrading_prod 127.0.0.1/32           md5
host    daytrading_prod daytrading_prod ::1/128                md5

# SSL必須接続（必要に応じて）
hostssl daytrading_prod daytrading_prod 0.0.0.0/0             md5
```

#### 2. データベース暗号化

```yaml
# config/production/database.yaml
database:
  url: "postgresql://daytrading_prod:${PROD_DB_PASSWORD}@${PROD_DB_HOST}:5432/daytrading_prod"
  
  # SSL設定
  ssl_mode: "require"
  ssl_cert: "${SSL_CERT_PATH}"
  ssl_key: "${SSL_KEY_PATH}"
  ssl_ca: "${SSL_CA_PATH}"
  
  # 接続引数
  connect_args:
    sslmode: "require"
    sslcert: "/opt/daytrading/ssl/client-cert.pem"
    sslkey: "/opt/daytrading/ssl/client-key.pem"
    sslrootcert: "/opt/daytrading/ssl/ca-cert.pem"
    connect_timeout: 10
    application_name: "DayTradingSystem_Production"
    
  # セキュリティ設定
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  pool_pre_ping: true
  echo: false  # 本番環境ではSQLログを無効
```

#### 3. データベースユーザー権限管理

```sql
-- PostgreSQL権限設定
-- daytrading_prod用データベース作成
CREATE DATABASE daytrading_prod 
    WITH OWNER daytrading_prod 
    ENCODING 'UTF8' 
    LC_COLLATE='ja_JP.UTF-8' 
    LC_CTYPE='ja_JP.UTF-8';

-- 本番ユーザー権限設定
GRANT CONNECT ON DATABASE daytrading_prod TO daytrading_prod;
GRANT USAGE ON SCHEMA public TO daytrading_prod;
GRANT CREATE ON SCHEMA public TO daytrading_prod;

-- テーブル権限（必要最小限）
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO daytrading_prod;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO daytrading_prod;

-- 将来作成されるオブジェクトへの権限
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO daytrading_prod;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT USAGE, SELECT ON SEQUENCES TO daytrading_prod;

-- 読み取り専用ユーザー（監査・分析用）
CREATE USER daytrading_readonly WITH PASSWORD 'readonly_password_here';
GRANT CONNECT ON DATABASE daytrading_prod TO daytrading_readonly;
GRANT USAGE ON SCHEMA public TO daytrading_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO daytrading_readonly;
```

---

## 🔐 アプリケーションセキュリティ

### 環境変数・設定管理

#### 1. セキュアな環境変数設定

```bash
# /opt/daytrading/.env
# ファイル作成
sudo -u daytrading touch /opt/daytrading/.env
sudo chmod 600 /opt/daytrading/.env
sudo chown daytrading:daytrading /opt/daytrading/.env

# 環境変数設定（例）
cat > /opt/daytrading/.env << 'EOF'
# 環境設定
ENVIRONMENT=production

# データベース認証情報
PROD_DB_PASSWORD=$(openssl rand -base64 32)
PROD_DB_HOST=localhost
PROD_DB_PORT=5432
PROD_DB_NAME=daytrading_prod

# セキュリティキー（256ビット以上）
SECRET_KEY=$(openssl rand -base64 64)
JWT_SECRET_KEY=$(openssl rand -base64 64)

# 暗号化キー
ENCRYPTION_KEY=$(openssl rand -base64 32)

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

# 監視・セキュリティ設定
MONITORING_ENABLED=true
SECURITY_AUDIT_ENABLED=true
FAILED_LOGIN_THRESHOLD=5
SESSION_TIMEOUT=3600

# バックアップ暗号化
BACKUP_ENCRYPTION_ENABLED=true
BACKUP_ENCRYPTION_KEY=$(openssl rand -base64 32)

# 取引設定
MAX_DAILY_TRADES=1000
RISK_THRESHOLD=0.02
MAX_POSITION_SIZE=100000
EOF
```

#### 2. 設定ファイル暗号化

```bash
# 機密設定ファイル暗号化スクリプト
cat > /opt/daytrading/scripts/encrypt_config.sh << 'EOF'
#!/bin/bash
set -euo pipefail

CONFIG_DIR="/opt/daytrading/config/production"
ENCRYPTED_DIR="/opt/daytrading/config/encrypted"

# 暗号化ディレクトリ作成
mkdir -p "$ENCRYPTED_DIR"
chmod 700 "$ENCRYPTED_DIR"

# GPG暗号化（パスフレーズはHSM等で管理）
for config_file in "$CONFIG_DIR"/*.yaml; do
    if [[ -f "$config_file" ]]; then
        filename=$(basename "$config_file")
        gpg --symmetric --cipher-algo AES256 --output "$ENCRYPTED_DIR/${filename}.gpg" "$config_file"
        echo "暗号化完了: $filename"
    fi
done

echo "設定ファイル暗号化完了"
EOF

chmod +x /opt/daytrading/scripts/encrypt_config.sh
```

### 統合セキュリティ機能

#### 1. 統合エラーハンドリングのセキュリティ強化

```python
# src/day_trade/core/error_handling/security_error_handler.py
"""
セキュリティ関連エラーハンドリング
"""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque

from day_trade.core.error_handling.unified_error_system import (
    ApplicationError, SecurityError, error_boundary
)
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityEvent:
    """セキュリティイベント"""
    event_type: str
    source_ip: str
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, any]
    severity: str


class SecurityMonitor:
    """セキュリティ監視システム"""
    
    def __init__(self):
        self.failed_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.blocked_ips: Set[str] = set()
        self.security_events: deque = deque(maxlen=1000)
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        
        # 設定
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.rate_limit_threshold = 100  # requests per minute
    
    @error_boundary(
        component_name="security_monitor",
        operation_name="record_failed_login",
        suppress_errors=True
    )
    def record_failed_login(self, ip_address: str, user_id: Optional[str] = None):
        """ログイン失敗記録"""
        now = datetime.now()
        
        # 失敗試行記録
        self.failed_attempts[ip_address].append(now)
        
        # 閾値チェック
        recent_attempts = [
            attempt for attempt in self.failed_attempts[ip_address]
            if now - attempt < self.lockout_duration
        ]
        
        if len(recent_attempts) >= self.max_failed_attempts:
            self.block_ip(ip_address, "Too many failed login attempts")
            
        # セキュリティイベント記録
        self.record_security_event(
            event_type="failed_login",
            source_ip=ip_address,
            user_id=user_id,
            details={"attempt_count": len(recent_attempts)},
            severity="warning" if len(recent_attempts) < self.max_failed_attempts else "critical"
        )
    
    def block_ip(self, ip_address: str, reason: str):
        """IP アドレスブロック"""
        self.blocked_ips.add(ip_address)
        
        logger.warning(
            "IPアドレスブロック",
            ip_address=ip_address,
            reason=reason,
            blocked_count=len(self.blocked_ips)
        )
        
        # セキュリティイベント記録
        self.record_security_event(
            event_type="ip_blocked",
            source_ip=ip_address,
            user_id=None,
            details={"reason": reason},
            severity="critical"
        )
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """IP ブロック状態確認"""
        return ip_address in self.blocked_ips
    
    def record_security_event(self, event_type: str, source_ip: str, 
                            user_id: Optional[str], details: Dict[str, any], 
                            severity: str):
        """セキュリティイベント記録"""
        event = SecurityEvent(
            event_type=event_type,
            source_ip=source_ip,
            user_id=user_id,
            timestamp=datetime.now(),
            details=details,
            severity=severity
        )
        
        self.security_events.append(event)
        
        # 重要イベントはすぐにログ出力
        if severity in ["critical", "high"]:
            logger.warning(
                f"セキュリティイベント: {event_type}",
                source_ip=source_ip,
                user_id=user_id,
                severity=severity,
                details=details
            )
    
    def get_security_summary(self) -> Dict[str, any]:
        """セキュリティサマリー取得"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp >= last_hour
        ]
        
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type] += 1
        
        return {
            "blocked_ips_count": len(self.blocked_ips),
            "recent_events_count": len(recent_events),
            "event_breakdown": dict(event_counts),
            "failed_attempts_count": sum(len(attempts) for attempts in self.failed_attempts.values()),
            "last_update": now.isoformat()
        }


# グローバルセキュリティモニター
_security_monitor: Optional[SecurityMonitor] = None


def get_security_monitor() -> SecurityMonitor:
    """セキュリティモニター取得"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor
```

#### 2. データ暗号化機能

```python
# src/day_trade/core/security/data_encryption.py
"""
データ暗号化・復号化機能
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, bytes, str

from day_trade.core.error_handling.unified_error_system import SecurityError, error_boundary
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)


class DataEncryption:
    """データ暗号化クラス"""
    
    def __init__(self, password: Union[str, bytes] = None):
        """
        暗号化インスタンス初期化
        
        Args:
            password: 暗号化パスワード（環境変数から取得される場合はNone）
        """
        if password is None:
            password = os.getenv('ENCRYPTION_KEY')
            if not password:
                raise SecurityError("暗号化キーが設定されていません")
        
        if isinstance(password, str):
            password = password.encode()
            
        # キー導出
        salt = b'day_trading_salt_2025'  # 本番環境では動的に生成・保存
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher = Fernet(key)
    
    @error_boundary(
        component_name="data_encryption",
        operation_name="encrypt",
        suppress_errors=False
    )
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        データ暗号化
        
        Args:
            data: 暗号化するデータ
            
        Returns:
            暗号化されたデータ（Base64エンコード）
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.cipher.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    @error_boundary(
        component_name="data_encryption",
        operation_name="decrypt",
        suppress_errors=False
    )
    def decrypt(self, encrypted_data: str) -> str:
        """
        データ復号化
        
        Args:
            encrypted_data: 暗号化されたデータ（Base64エンコード）
            
        Returns:
            復号化されたデータ
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            raise SecurityError(f"データ復号化失敗: {e}") from e
    
    @error_boundary(
        component_name="data_encryption",
        operation_name="encrypt_file",
        suppress_errors=False
    )
    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """
        ファイル暗号化
        
        Args:
            file_path: 暗号化するファイルパス
            output_path: 出力ファイルパス（指定しない場合は元ファイル + .encrypted）
            
        Returns:
            暗号化されたファイルパス
        """
        if output_path is None:
            output_path = file_path + '.encrypted'
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self.cipher.encrypt(file_data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            # 権限設定
            os.chmod(output_path, 0o600)
            
            logger.info(f"ファイル暗号化完了: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            raise SecurityError(f"ファイル暗号化失敗: {e}") from e
    
    @error_boundary(
        component_name="data_encryption",
        operation_name="decrypt_file",
        suppress_errors=False
    )
    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """
        ファイル復号化
        
        Args:
            encrypted_file_path: 暗号化されたファイルパス
            output_path: 出力ファイルパス
            
        Returns:
            復号化されたファイルパス
        """
        if output_path is None:
            if encrypted_file_path.endswith('.encrypted'):
                output_path = encrypted_file_path[:-10]  # .encrypted を削除
            else:
                output_path = encrypted_file_path + '.decrypted'
        
        try:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            # 権限設定
            os.chmod(output_path, 0o600)
            
            logger.info(f"ファイル復号化完了: {encrypted_file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            raise SecurityError(f"ファイル復号化失敗: {e}") from e


# グローバル暗号化インスタンス
_data_encryption: Optional[DataEncryption] = None


def get_data_encryption() -> DataEncryption:
    """データ暗号化インスタンス取得"""
    global _data_encryption
    if _data_encryption is None:
        _data_encryption = DataEncryption()
    return _data_encryption
```

---

## 🔍 セキュリティ監査・ログ管理

### 統合ログセキュリティ

#### 1. セキュリティログ設定

```yaml
# config/production/security_logging.yaml
security_logging:
  enabled: true
  log_level: "INFO"
  
  # セキュリティイベントログ
  security_events:
    enabled: true
    file_path: "/opt/daytrading/logs/security/security_events.jsonl"
    rotation_size: "50MB"
    rotation_count: 20
    
  # 認証ログ
  authentication:
    enabled: true
    file_path: "/opt/daytrading/logs/security/auth.jsonl"
    log_successful_logins: true
    log_failed_attempts: true
    
  # データアクセスログ
  data_access:
    enabled: true
    file_path: "/opt/daytrading/logs/security/data_access.jsonl"
    log_queries: true
    log_sensitive_operations: true
    
  # システム変更ログ
  system_changes:
    enabled: true
    file_path: "/opt/daytrading/logs/security/system_changes.jsonl"
    log_config_changes: true
    log_user_changes: true
    
  # セキュリティアラート
  alerts:
    enabled: true
    immediate_alert_threshold: "critical"
    email_notifications: true
    syslog_integration: true
```

#### 2. ログ暗号化・署名

```bash
# ログ暗号化スクリプト
cat > /opt/daytrading/scripts/secure_logs.sh << 'EOF'
#!/bin/bash
set -euo pipefail

LOG_DIR="/opt/daytrading/logs"
SECURE_LOG_DIR="/opt/daytrading/logs/secure"
GPG_KEY_ID="daytrading-logs@local"

# セキュアログディレクトリ作成
mkdir -p "$SECURE_LOG_DIR"
chmod 700 "$SECURE_LOG_DIR"

# 日次ログローテーション・暗号化
find "$LOG_DIR" -name "*.log" -mtime +1 | while read -r log_file; do
    if [[ ! "$log_file" =~ secure/ ]]; then
        # ハッシュ値計算（改ざん検知用）
        sha256sum "$log_file" > "${log_file}.sha256"
        
        # GPG署名・暗号化
        gpg --local-user "$GPG_KEY_ID" --sign --armor "$log_file"
        gpg --encrypt --armor -r "$GPG_KEY_ID" "${log_file}.asc" 
        
        # セキュアディレクトリに移動
        mv "${log_file}.asc.asc" "$SECURE_LOG_DIR/"
        mv "${log_file}.sha256" "$SECURE_LOG_DIR/"
        
        # 元ファイル削除
        rm "$log_file" "${log_file}.asc"
        
        echo "ログセキュア化完了: $(basename "$log_file")"
    fi
done

echo "ログセキュア化処理完了"
EOF

chmod +x /opt/daytrading/scripts/secure_logs.sh

# cron設定（毎日午前2時実行）
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/daytrading/scripts/secure_logs.sh") | crontab -
```

### セキュリティ監査機能

#### 1. 自動セキュリティ監査

```python
# src/day_trade/core/security/security_auditor.py
"""
セキュリティ監査機能
"""

import os
import json
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from day_trade.core.error_handling.unified_error_system import SecurityError, error_boundary
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)


class SecurityAuditor:
    """セキュリティ監査システム"""
    
    def __init__(self, config_path: str = "config/production/security_audit.yaml"):
        self.config_path = config_path
        self.audit_results: List[Dict[str, Any]] = []
        self.last_audit: Optional[datetime] = None
    
    @error_boundary(
        component_name="security_auditor",
        operation_name="run_full_audit",
        suppress_errors=False
    )
    def run_full_audit(self) -> Dict[str, Any]:
        """完全セキュリティ監査実行"""
        audit_start = datetime.now()
        audit_results = {
            "audit_id": f"audit_{audit_start.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": audit_start.isoformat(),
            "results": {}
        }
        
        try:
            # 1. ファイル権限監査
            audit_results["results"]["file_permissions"] = self._audit_file_permissions()
            
            # 2. 設定ファイル監査
            audit_results["results"]["configuration"] = self._audit_configuration()
            
            # 3. ネットワークセキュリティ監査
            audit_results["results"]["network_security"] = self._audit_network_security()
            
            # 4. データベースセキュリティ監査
            audit_results["results"]["database_security"] = self._audit_database_security()
            
            # 5. ログセキュリティ監査
            audit_results["results"]["log_security"] = self._audit_log_security()
            
            # 6. 脆弱性スキャン
            audit_results["results"]["vulnerability_scan"] = self._vulnerability_scan()
            
            # 監査完了
            audit_end = datetime.now()
            audit_results["duration_seconds"] = (audit_end - audit_start).total_seconds()
            audit_results["status"] = "completed"
            
            # 結果保存
            self._save_audit_results(audit_results)
            self.last_audit = audit_start
            
            logger.info(
                "セキュリティ監査完了",
                audit_id=audit_results["audit_id"],
                duration=audit_results["duration_seconds"]
            )
            
            return audit_results
            
        except Exception as e:
            audit_results["status"] = "failed"
            audit_results["error"] = str(e)
            logger.error(f"セキュリティ監査失敗: {e}")
            raise SecurityError(f"セキュリティ監査失敗: {e}") from e
    
    def _audit_file_permissions(self) -> Dict[str, Any]:
        """ファイル権限監査"""
        results = {
            "status": "passed",
            "issues": [],
            "checked_paths": []
        }
        
        # 重要ディレクトリの権限チェック
        critical_paths = [
            ("/opt/daytrading/config", 0o700),
            ("/opt/daytrading/ssl", 0o700),
            ("/opt/daytrading/.env", 0o600),
            ("/opt/daytrading/logs", 0o750),
            ("/opt/daytrading/backups", 0o755)
        ]
        
        for path, expected_perm in critical_paths:
            if os.path.exists(path):
                current_perm = oct(os.stat(path).st_mode)[-3:]
                expected_perm_str = oct(expected_perm)[-3:]
                
                results["checked_paths"].append({
                    "path": path,
                    "current_permission": current_perm,
                    "expected_permission": expected_perm_str,
                    "status": "ok" if current_perm == expected_perm_str else "issue"
                })
                
                if current_perm != expected_perm_str:
                    results["issues"].append(f"権限不正: {path} (現在: {current_perm}, 期待: {expected_perm_str})")
                    results["status"] = "failed"
        
        return results
    
    def _audit_configuration(self) -> Dict[str, Any]:
        """設定ファイル監査"""
        results = {
            "status": "passed",
            "issues": [],
            "checked_configs": []
        }
        
        # 設定ファイルのセキュリティチェック
        config_files = [
            "/opt/daytrading/config/production/database.yaml",
            "/opt/daytrading/.env"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                # 機密情報の平文確認
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # 危険なパターンチェック
                dangerous_patterns = [
                    "password=plaintext",
                    "secret=123456",
                    "debug=true",
                    "ssl_verify=false"
                ]
                
                config_issues = []
                for pattern in dangerous_patterns:
                    if pattern.lower() in content.lower():
                        config_issues.append(f"危険な設定: {pattern}")
                
                results["checked_configs"].append({
                    "file": config_file,
                    "issues": config_issues,
                    "status": "ok" if not config_issues else "issue"
                })
                
                if config_issues:
                    results["issues"].extend(config_issues)
                    results["status"] = "failed"
        
        return results
    
    def _audit_network_security(self) -> Dict[str, Any]:
        """ネットワークセキュリティ監査"""
        results = {
            "status": "passed",
            "issues": [],
            "open_ports": [],
            "firewall_status": "unknown"
        }
        
        try:
            # UFWステータス確認
            ufw_result = subprocess.run(
                ["sudo", "ufw", "status"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if "Status: active" in ufw_result.stdout:
                results["firewall_status"] = "active"
            else:
                results["firewall_status"] = "inactive"
                results["issues"].append("ファイアウォールが無効")
                results["status"] = "failed"
            
            # 開放ポート確認
            netstat_result = subprocess.run(
                ["netstat", "-tuln"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            # 危険なポート開放チェック
            dangerous_ports = ["22", "3389", "5432", "3306"]  # SSH, RDP, PostgreSQL, MySQL
            for line in netstat_result.stdout.split('\n'):
                if "LISTEN" in line:
                    for port in dangerous_ports:
                        if f":{port}" in line and "127.0.0.1" not in line:
                            results["issues"].append(f"危険なポート開放: {port}")
                            results["status"] = "failed"
                        
                    results["open_ports"].append(line.strip())
        
        except subprocess.TimeoutExpired:
            results["issues"].append("ネットワーク監査タイムアウト")
            results["status"] = "failed"
        except Exception as e:
            results["issues"].append(f"ネットワーク監査エラー: {e}")
            results["status"] = "failed"
        
        return results
    
    def _audit_database_security(self) -> Dict[str, Any]:
        """データベースセキュリティ監査"""
        results = {
            "status": "passed",
            "issues": [],
            "connection_security": "unknown",
            "user_permissions": []
        }
        
        try:
            from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
            from sqlalchemy import text
            
            manager = get_unified_database_manager()
            if manager and manager.production_db_manager:
                with manager.production_db_manager.get_session() as session:
                    # SSL接続確認
                    ssl_result = session.execute(text("SHOW ssl;"))
                    ssl_status = ssl_result.scalar()
                    
                    if ssl_status == 'on':
                        results["connection_security"] = "ssl_enabled"
                    else:
                        results["connection_security"] = "ssl_disabled"
                        results["issues"].append("データベースSSL無効")
                        results["status"] = "failed"
                    
                    # ユーザー権限確認
                    users_result = session.execute(text("""
                        SELECT usename, usesuper, usecreatedb, usecreaterole
                        FROM pg_user
                        WHERE usename LIKE '%daytrading%';
                    """))
                    
                    for user in users_result.fetchall():
                        user_info = {
                            "username": user[0],
                            "superuser": user[1],
                            "createdb": user[2],
                            "createrole": user[3]
                        }
                        results["user_permissions"].append(user_info)
                        
                        # 過度な権限チェック
                        if user[1]:  # superuser
                            results["issues"].append(f"過度な権限: {user[0]} はスーパーユーザー")
                            results["status"] = "failed"
        
        except Exception as e:
            results["issues"].append(f"データベース監査エラー: {e}")
            results["status"] = "failed"
        
        return results
    
    def _audit_log_security(self) -> Dict[str, Any]:
        """ログセキュリティ監査"""
        results = {
            "status": "passed",
            "issues": [],
            "log_files": [],
            "log_permissions": []
        }
        
        log_directory = Path("/opt/daytrading/logs")
        if log_directory.exists():
            for log_file in log_directory.rglob("*.log"):
                file_stat = log_file.stat()
                file_perm = oct(file_stat.st_mode)[-3:]
                
                log_info = {
                    "file": str(log_file),
                    "permission": file_perm,
                    "size_mb": round(file_stat.st_size / 1024 / 1024, 2)
                }
                results["log_files"].append(log_info)
                
                # ログファイル権限チェック（640 or 644が適切）
                if file_perm not in ["640", "644"]:
                    results["issues"].append(f"ログファイル権限不正: {log_file} ({file_perm})")
                    results["status"] = "failed"
        
        return results
    
    def _vulnerability_scan(self) -> Dict[str, Any]:
        """脆弱性スキャン"""
        results = {
            "status": "passed",
            "issues": [],
            "scanned_components": []
        }
        
        try:
            # Pythonパッケージ脆弱性チェック（safety）
            safety_result = subprocess.run(
                ["safety", "check"], 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if safety_result.returncode != 0:
                vulnerabilities = safety_result.stdout.split('\n')
                for vuln in vulnerabilities:
                    if vuln.strip():
                        results["issues"].append(f"脆弱性: {vuln.strip()}")
                        results["status"] = "failed"
            
            results["scanned_components"].append("python_packages")
        
        except subprocess.TimeoutExpired:
            results["issues"].append("脆弱性スキャンタイムアウト")
            results["status"] = "failed"
        except FileNotFoundError:
            results["issues"].append("脆弱性スキャンツール未インストール (safety)")
        except Exception as e:
            results["issues"].append(f"脆弱性スキャンエラー: {e}")
            results["status"] = "failed"
        
        return results
    
    def _save_audit_results(self, results: Dict[str, Any]):
        """監査結果保存"""
        audit_dir = Path("/opt/daytrading/logs/security/audits")
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        audit_file = audit_dir / f"{results['audit_id']}.json"
        with open(audit_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 権限設定
        audit_file.chmod(0o600)
        
        logger.info(f"監査結果保存: {audit_file}")


# グローバル監査インスタンス
_security_auditor: Optional[SecurityAuditor] = None


def get_security_auditor() -> SecurityAuditor:
    """セキュリティ監査インスタンス取得"""
    global _security_auditor
    if _security_auditor is None:
        _security_auditor = SecurityAuditor()
    return _security_auditor
```

---

## 🛡️ インシデント対応

### セキュリティインシデント対応手順

#### 1. インシデント検知・初動対応

```bash
# 緊急時セキュリティ対応スクリプト
cat > /opt/daytrading/scripts/emergency_security_response.sh << 'EOF'
#!/bin/bash
set -euo pipefail

INCIDENT_TYPE=${1:-"unknown"}
INCIDENT_ID="incident_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/opt/daytrading/logs/security/incidents"

echo "=== セキュリティインシデント対応開始 ==="
echo "インシデントタイプ: $INCIDENT_TYPE"
echo "インシデントID: $INCIDENT_ID"

# ログディレクトリ作成
mkdir -p "$LOG_DIR"

# 1. システム状態記録
echo "システム状態記録中..."
{
    echo "=== システム状態 $(date) ==="
    systemctl status daytrading
    ps aux | grep daytrading
    netstat -tuln
    who
    last -n 20
} > "$LOG_DIR/${INCIDENT_ID}_system_state.log"

# 2. 緊急バックアップ作成
echo "緊急バックアップ作成中..."
sudo -u daytrading python -c "
from src.day_trade.infrastructure.database.unified_database_manager import get_unified_database_manager
manager = get_unified_database_manager()
if manager:
    result = manager.create_backup('emergency_security_${INCIDENT_ID}')
    print(f'緊急バックアップ: {result[\"status\"]}')
"

# 3. ネットワーク接続記録
echo "ネットワーク接続記録中..."
{
    echo "=== アクティブ接続 $(date) ==="
    ss -tuln
    ss -tp
    iptables -L -n
} > "$LOG_DIR/${INCIDENT_ID}_network_state.log"

# 4. セキュリティ関連ログ収集
echo "セキュリティログ収集中..."
{
    echo "=== 認証ログ ==="
    tail -100 /var/log/auth.log
    echo "=== アプリケーションログ ==="
    tail -100 /opt/daytrading/logs/production/production.log
    echo "=== データベースログ ==="
    tail -100 /var/log/postgresql/postgresql-*.log
} > "$LOG_DIR/${INCIDENT_ID}_security_logs.log"

# 5. インシデント種別別対応
case "$INCIDENT_TYPE" in
    "unauthorized_access")
        echo "不正アクセス対応実行中..."
        # 疑わしい接続の切断
        # ファイアウォールルール追加
        ;;
    "data_breach")
        echo "データ侵害対応実行中..."
        # システム即座停止
        systemctl stop daytrading
        ;;
    "malware")
        echo "マルウェア対応実行中..."
        # ネットワーク分離
        # スキャン実行
        ;;
    *)
        echo "汎用セキュリティ対応実行中..."
        ;;
esac

echo "=== インシデント対応完了 ==="
echo "インシデントID: $INCIDENT_ID"
echo "ログディレクトリ: $LOG_DIR"
EOF

chmod +x /opt/daytrading/scripts/emergency_security_response.sh
```

#### 2. 侵入検知・対応

```python
# src/day_trade/core/security/intrusion_detection.py
"""
侵入検知システム
"""

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Pattern, Set
from dataclasses import dataclass
from collections import defaultdict, deque

from day_trade.core.error_handling.unified_error_system import SecurityError, error_boundary
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)


@dataclass
class IntrusionAlert:
    """侵入検知アラート"""
    alert_id: str
    alert_type: str
    source_ip: str
    timestamp: datetime
    severity: str
    description: str
    evidence: Dict[str, any]


class IntrusionDetectionSystem:
    """侵入検知システム"""
    
    def __init__(self):
        self.suspicious_patterns: List[Pattern] = []
        self.blocked_ips: Set[str] = set()
        self.intrusion_alerts: deque = deque(maxlen=1000)
        self.activity_timeline: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 初期化
        self._load_detection_patterns()
        
    def _load_detection_patterns(self):
        """検知パターン読み込み"""
        # SQL injection patterns
        sql_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+.*",
            r"(?i)(\-\-|\#|\/\*|\*\/)",
            r"(?i)(\'|\").*(\1)",
            r"(?i)(or|and)\s+\d+\s*=\s*\d+"
        ]
        
        # XSS patterns
        xss_patterns = [
            r"(?i)<script[^>]*>.*</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=",
            r"(?i)<.*?javascript:.*?>"
        ]
        
        # Command injection patterns
        cmd_patterns = [
            r"(?i)(;|\||\&)\s*(cat|ls|ps|id|whoami|pwd)",
            r"(?i)\$\(.*\)",
            r"(?i)`.*`",
            r"(?i)(rm|mv|cp|chmod|chown)\s+"
        ]
        
        # Directory traversal patterns
        path_patterns = [
            r"\.\.\/",
            r"\.\.\\",
            r"\/etc\/passwd",
            r"\/etc\/shadow"
        ]
        
        all_patterns = sql_patterns + xss_patterns + cmd_patterns + path_patterns
        self.suspicious_patterns = [re.compile(pattern) for pattern in all_patterns]
        
        logger.info(f"侵入検知パターン読み込み完了: {len(self.suspicious_patterns)}件")
    
    @error_boundary(
        component_name="intrusion_detection",
        operation_name="analyze_request",
        suppress_errors=True
    )
    def analyze_request(self, source_ip: str, request_data: str, 
                       request_type: str = "unknown") -> bool:
        """
        リクエスト分析
        
        Args:
            source_ip: 送信元IPアドレス
            request_data: リクエストデータ
            request_type: リクエスト種別
            
        Returns:
            True if suspicious, False otherwise
        """
        is_suspicious = False
        matched_patterns = []
        
        # パターンマッチング
        for pattern in self.suspicious_patterns:
            if pattern.search(request_data):
                matched_patterns.append(pattern.pattern)
                is_suspicious = True
        
        # 活動記録
        self.activity_timeline[source_ip].append({
            "timestamp": datetime.now(),
            "request_type": request_type,
            "suspicious": is_suspicious,
            "patterns": matched_patterns
        })
        
        # 疑わしい活動の場合、アラート生成
        if is_suspicious:
            self._generate_intrusion_alert(
                alert_type="suspicious_request",
                source_ip=source_ip,
                severity="medium",
                description=f"疑わしいリクエスト検知: {request_type}",
                evidence={
                    "request_data": request_data[:500],  # 最初の500文字のみ
                    "matched_patterns": matched_patterns,
                    "request_type": request_type
                }
            )
        
        return is_suspicious
    
    @error_boundary(
        component_name="intrusion_detection",
        operation_name="analyze_activity_pattern",
        suppress_errors=True
    )
    def analyze_activity_pattern(self, source_ip: str) -> Dict[str, any]:
        """
        活動パターン分析
        
        Args:
            source_ip: 分析対象IPアドレス
            
        Returns:
            分析結果
        """
        activities = self.activity_timeline.get(source_ip, deque())
        if not activities:
            return {"status": "no_activity"}
        
        now = datetime.now()
        recent_activities = [
            activity for activity in activities
            if now - activity["timestamp"] <= timedelta(hours=1)
        ]
        
        analysis = {
            "total_requests": len(activities),
            "recent_requests": len(recent_activities),
            "suspicious_requests": sum(1 for a in recent_activities if a["suspicious"]),
            "request_frequency": len(recent_activities) / 60,  # per minute
            "risk_level": "low"
        }
        
        # リスクレベル判定
        if analysis["suspicious_requests"] > 5:
            analysis["risk_level"] = "high"
        elif analysis["request_frequency"] > 10:  # 10 requests/minute
            analysis["risk_level"] = "high"
        elif analysis["suspicious_requests"] > 0:
            analysis["risk_level"] = "medium"
        
        # 高リスクの場合、アラート生成
        if analysis["risk_level"] == "high":
            self._generate_intrusion_alert(
                alert_type="high_risk_activity",
                source_ip=source_ip,
                severity="high",
                description=f"高リスク活動パターン検知",
                evidence=analysis
            )
            
            # 自動ブロック
            self.block_ip(source_ip, "High risk activity pattern")
        
        return analysis
    
    def block_ip(self, ip_address: str, reason: str):
        """IPアドレスブロック"""
        self.blocked_ips.add(ip_address)
        
        # システムレベルでのブロック（iptables）
        try:
            import subprocess
            subprocess.run([
                "sudo", "iptables", "-A", "INPUT", 
                "-s", ip_address, "-j", "DROP"
            ], timeout=10)
            
            logger.warning(f"IPブロック実行: {ip_address} - {reason}")
        except Exception as e:
            logger.error(f"IPブロック失敗: {e}")
        
        # アラート生成
        self._generate_intrusion_alert(
            alert_type="ip_blocked",
            source_ip=ip_address,
            severity="critical",
            description=f"IPアドレスブロック実行",
            evidence={"reason": reason}
        )
    
    def _generate_intrusion_alert(self, alert_type: str, source_ip: str, 
                                severity: str, description: str, 
                                evidence: Dict[str, any]):
        """侵入アラート生成"""
        alert = IntrusionAlert(
            alert_id=f"intrusion_{int(time.time())}_{source_ip.replace('.', '_')}",
            alert_type=alert_type,
            source_ip=source_ip,
            timestamp=datetime.now(),
            severity=severity,
            description=description,
            evidence=evidence
        )
        
        self.intrusion_alerts.append(alert)
        
        logger.warning(
            f"侵入検知アラート: {description}",
            alert_id=alert.alert_id,
            source_ip=source_ip,
            severity=severity,
            alert_type=alert_type
        )
    
    def get_active_threats(self) -> List[Dict[str, any]]:
        """アクティブ脅威取得"""
        now = datetime.now()
        recent_threshold = now - timedelta(hours=24)
        
        recent_alerts = [
            alert for alert in self.intrusion_alerts
            if alert.timestamp >= recent_threshold
        ]
        
        # 脅威レベル別分類
        threats_by_severity = defaultdict(list)
        for alert in recent_alerts:
            threats_by_severity[alert.severity].append({
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "source_ip": alert.source_ip,
                "timestamp": alert.timestamp.isoformat(),
                "description": alert.description
            })
        
        return dict(threats_by_severity)
    
    def get_security_status(self) -> Dict[str, any]:
        """セキュリティステータス取得"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        
        recent_alerts = [
            alert for alert in self.intrusion_alerts
            if alert.timestamp >= last_hour
        ]
        
        return {
            "blocked_ips_count": len(self.blocked_ips),
            "recent_alerts_count": len(recent_alerts),
            "high_risk_sources": len([
                ip for ip, activities in self.activity_timeline.items()
                if self.analyze_activity_pattern(ip).get("risk_level") == "high"
            ]),
            "total_monitored_ips": len(self.activity_timeline),
            "last_update": now.isoformat()
        }


# グローバル侵入検知システム
_intrusion_detection: Optional[IntrusionDetectionSystem] = None


def get_intrusion_detection_system() -> IntrusionDetectionSystem:
    """侵入検知システム取得"""
    global _intrusion_detection
    if _intrusion_detection is None:
        _intrusion_detection = IntrusionDetectionSystem()
    return _intrusion_detection
```

---

## 📋 セキュリティチェックリスト

### 日次セキュリティチェック

- [ ] システムログ確認
- [ ] 認証失敗ログ確認
- [ ] アクティブアラート確認
- [ ] ブロックIP確認
- [ ] バックアップ整合性確認
- [ ] 証明書有効期限確認

### 週次セキュリティチェック

- [ ] 脆弱性スキャン実行
- [ ] ファイル権限監査
- [ ] ネットワークセキュリティ確認
- [ ] セキュリティパッチ適用
- [ ] ログローテーション確認
- [ ] セキュリティポリシー見直し

### 月次セキュリティチェック

- [ ] 包括的セキュリティ監査
- [ ] ペネトレーションテスト
- [ ] インシデント対応訓練
- [ ] セキュリティ設定見直し
- [ ] アクセス権限監査
- [ ] バックアップ復元テスト

---

## 🔧 セキュリティツール・コマンド

### 日常使用コマンド

```bash
# セキュリティ状態確認
python -c "
from src.day_trade.core.security.security_auditor import get_security_auditor
auditor = get_security_auditor()
print('セキュリティ監査実行中...')
results = auditor.run_full_audit()
print(f'監査結果: {results[\"status\"]}')
if results['status'] == 'failed':
    print('問題発見:')
    for category, result in results['results'].items():
        if result.get('issues'):
            print(f'  {category}: {len(result[\"issues\"])}件')
"

# 侵入検知状況確認
python -c "
from src.day_trade.core.security.intrusion_detection import get_intrusion_detection_system
ids = get_intrusion_detection_system()
status = ids.get_security_status()
print(f'ブロック済みIP: {status[\"blocked_ips_count\"]}')
print(f'最近のアラート: {status[\"recent_alerts_count\"]}')
print(f'高リスクソース: {status[\"high_risk_sources\"]}')
"

# セキュリティログ確認
tail -f /opt/daytrading/logs/security/security_events.jsonl | jq .

# ファイアウォール状態確認
sudo ufw status verbose

# 証明書有効期限確認
openssl x509 -in /opt/daytrading/ssl/cert.pem -text -noout | grep "Not After"
```

---

**このセキュリティ設定ガイドに従って、Day Trading Systemの包括的なセキュリティ対策を実装してください。定期的な見直しと最新の脅威情報に基づく更新を継続することが重要です。**

---

*最終更新: 2025年8月18日*  
*ドキュメントバージョン: 1.0.0 (統合データベース管理システム対応)*