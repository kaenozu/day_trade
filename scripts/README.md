# Day Trade System スクリプト集

**Issue #320対応**: 本番稼働準備・運用スクリプト完全版

---

## 📋 スクリプト一覧

| スクリプト | 用途 | 環境 | 実行方法 |
|-----------|------|------|----------|
| `start_production.py` | 本番システム起動 | Production | `python scripts/start_production.py` |
| `start_development.py` | 開発システム起動 | Development | `python scripts/start_development.py` |
| `stop_system.py` | システム停止 | All | `python scripts/stop_system.py [force/clean]` |
| `system_status.py` | 状態確認 | All | `python scripts/system_status.py [json]` |

---

## 🚀 使用方法

### 本番環境起動
```bash
# 環境変数設定
export DAYTRADE_ENV=production

# システム起動
python scripts/start_production.py
```

**機能**:
- 本番環境設定の自動読み込み
- セキュリティチェック実行
- パフォーマンス監視開始
- 自動復旧システム起動
- 構造化ログ記録

### 開発環境起動
```bash
# 環境変数設定
export DAYTRADE_ENV=development

# システム起動
python scripts/start_development.py
```

**機能**:
- 開発環境設定の自動読み込み
- 詳細デバッグログ出力
- 開発用診断機能
- 短い監視間隔設定

### システム停止
```bash
# 正常停止
python scripts/stop_system.py

# 強制停止
python scripts/stop_system.py force

# クリーンアップのみ
python scripts/stop_system.py clean

# ヘルプ表示
python scripts/stop_system.py help
```

**停止プロセス**:
1. 正常停止シグナル送信（SIGTERM）
2. 30秒間の正常停止待機
3. タイムアウト時の強制停止（SIGKILL）
4. PIDファイル・一時ファイルクリーンアップ

### システム状態確認
```bash
# 人間が読みやすい形式
python scripts/system_status.py

# JSON形式（API連携用）
python scripts/system_status.py json
```

**確認項目**:
- プロセス稼働状況
- メモリ・CPU使用率
- パフォーマンス指標
- ヘルス状態
- ログ統計

---

## 🔧 環境変数

### 必須環境変数
```bash
DAYTRADE_ENV          # 環境設定 (production/development)
```

### オプション環境変数
```bash
DAYTRADE_LOG_LEVEL    # ログレベル (DEBUG/INFO/WARNING/ERROR)
DAYTRADE_MAX_SYMBOLS  # 最大銘柄数
DAYTRADE_MONITORING   # 監視機能有効/無効 (true/false)
DAYTRADE_DB_URL       # データベースURL
DAYTRADE_PAPER_TRADING # ペーパートレード有効/無効 (true/false)
```

---

## 📊 出力例

### システム状態確認出力例
```
============================================================
SYSTEM STATUS REPORT
============================================================
Timestamp: 2025-01-08T12:00:00
System Running: YES

=== PROCESS INFORMATION ===
  Status: RUNNING
  PID: 12345
  Uptime: 2:30:45
  Memory Usage: 1024.5 MB
  CPU Usage: 15.2%

=== ENVIRONMENT INFORMATION ===
  Current Environment: production
  Config Loaded: True
  Config Directory: /path/to/config
  Available Configs: production, development
  Validation Status: valid

=== PERFORMANCE INFORMATION ===
  Monitoring Available: True
  ML Analysis Target: 3.6s
  Memory Limit: 2048 MB
  CPU Limit: 80%

=== HEALTH INFORMATION ===
  Recovery System: True
  Monitoring Active: True
  Degradation Level: 0
  Recent Recovery Actions: 0
  Uptime Estimate: 99.5%

=== RESOURCE INFORMATION ===
  Disk Usage: 45.2% (50.5 GB free)
  Memory Usage: 62.8% (2.8 GB available)
  CPU Usage: 25.4% (4 cores)
  Internet Connection: True

=== LOG INFORMATION ===
  Recent Errors (24h): 0
  Structured Logging: True
  /logs/production: 5 files
```

---

## ⚡ パフォーマンス指標

### 本番環境目標値
- **ML分析速度**: 3.6秒/85銘柄
- **メモリ使用量**: 2048MB以下
- **CPU使用率**: 80%以下
- **可用性**: 99.9%以上
- **復旧時間**: 5分以内

### 開発環境目標値
- **ML分析速度**: 5.0秒/85銘柄
- **メモリ使用量**: 1024MB以下
- **CPU使用率**: 70%以下

---

## 🛠️ トラブルシューティング

### 起動エラー対処法

#### 設定ファイルエラー
```
ConfigurationError: 設定ファイルが見つかりません
```
**対処**: `config/production.json`または`config/development.json`の存在確認

#### モジュールインポートエラー
```
ModuleNotFoundError: No module named 'day_trade'
```
**対処**: プロジェクトルートから実行、PYTHONPATHの設定確認

#### 権限エラー
```
PermissionError: [Errno 13] Permission denied
```
**対処**: ファイル・ディレクトリの権限確認、管理者権限での実行

### 停止エラー対処法

#### PIDファイル見つからない
```
PIDファイルが見つかりません
```
**対処**: システムが既に停止済み、または手動でプロセス終了済み

#### プロセス停止タイムアウト
```
正常停止タイムアウト - 強制終了を実行
```
**対処**: システムが応答しない場合の自動対処、ログで原因確認

---

## 🔒 セキュリティ考慮事項

### PIDファイル管理
- `daytrade.pid`: 実行中プロセスのPID記録
- システム停止時に自動削除
- プロセス存在確認による安全性担保

### ログセキュリティ
- 本番環境では機密情報のマスク化
- アクセスログの記録
- ログファイルの権限制限

### プロセス管理
- 正常停止シグナル優先
- タイムアウト後の強制停止
- 孤児プロセスの自動検出・停止

---

## 📅 定期メンテナンス

### 日次作業
```bash
# システム状態確認
python scripts/system_status.py

# ログエラーチェック
python scripts/system_status.py json | grep -i error
```

### 週次作業
```bash
# システム再起動
python scripts/stop_system.py
python scripts/start_production.py

# ログクリーンアップ
python scripts/stop_system.py clean
```

---

## 🎯 使用例・ユースケース

### 1. 定期的な健全性チェック
```bash
#!/bin/bash
# health_check.sh

status=$(python scripts/system_status.py json | jq -r '.system_running')
if [ "$status" != "true" ]; then
    echo "System not running - attempting restart"
    python scripts/start_production.py
fi
```

### 2. パフォーマンス監視
```bash
#!/bin/bash
# performance_check.sh

memory_usage=$(python scripts/system_status.py json | jq -r '.resources.memory.usage_percent')
if (( $(echo "$memory_usage > 90" | bc -l) )); then
    echo "High memory usage detected: ${memory_usage}%"
    # アラート通知処理
fi
```

### 3. 自動再起動スクリプト
```bash
#!/bin/bash
# auto_restart.sh

echo "Performing scheduled restart"
python scripts/stop_system.py
sleep 10
python scripts/start_production.py
python scripts/system_status.py
```

---

## 📞 サポート・問い合わせ

### ログファイル確認箇所
- **起動ログ**: `logs/production/production.log`
- **エラーログ**: `logs/production/errors.jsonl`
- **パフォーマンスログ**: `logs/production/performance.jsonl`

### デバッグ情報収集
```bash
# 完全な診断情報収集
python scripts/system_status.py json > system_diagnosis.json

# エラー専用ログ抽出
grep -i "error\|exception\|failed" logs/production/production.log
```

---

**🚀 本番稼働準備完了！**

これらのスクリプトにより、Day Trade Systemの安全で確実な運用が可能になります。
- **自動化された起動・停止プロセス**
- **包括的なシステム監視**
- **堅牢なエラーハンドリング**
- **詳細な運用ドキュメント**
