# Day Trade System 本番環境セットアップガイド

**Issue #320対応**: システム本番稼働準備・設定最適化完了

---

## 📋 概要

このガイドは、Day Trade Systemを本番環境で安全かつ確実に稼働させるための手順を説明します。

### 🎯 対象システム
- **Phase 1-4統合システム**: ML分析、ポートフォリオ最適化、トレーディングシミュレーション
- **Issue #311**: パフォーマンス監視システム（3.6秒/85銘柄）
- **Issue #312**: フォールトトレランス・自動復旧システム（99.9%可用性目標）
- **Issue #313**: 総合統合テストシステム（100%テスト成功実績）

---

## 🚀 本番環境要件

### システム要件
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.8以上
- **メモリ**: 4GB以上推奨
- **ディスク**: 10GB以上の空き容量
- **ネットワーク**: インターネット接続必須（APIアクセス用）

### 推奨ハードウェア
- **CPU**: 4コア以上
- **メモリ**: 8GB以上
- **ディスク**: SSD推奨
- **ネットワーク**: 安定したブロードバンド接続

---

## 📁 ディレクトリ構成

```
day_trade/
├── config/
│   ├── production.json     # 本番環境設定
│   ├── development.json    # 開発環境設定
│   └── settings.json       # デフォルト設定
├── scripts/
│   ├── start_production.py # 本番環境起動
│   ├── start_development.py # 開発環境起動
│   ├── stop_system.py      # システム停止
│   └── system_status.py    # 状態確認
├── src/day_trade/
│   └── config/
│       └── environment_config.py # 環境設定管理
├── logs/
│   ├── production/         # 本番ログ
│   └── development/        # 開発ログ
└── data/
    ├── production/         # 本番データ
    └── development/        # 開発データ
```

---

## 🔧 セットアップ手順

### 1. 環境変数設定

本番環境用の環境変数を設定します：

```bash
# Windows (PowerShell)
$env:DAYTRADE_ENV = "production"
$env:DAYTRADE_LOG_LEVEL = "INFO"
$env:DAYTRADE_MONITORING = "true"
$env:DAYTRADE_FAULT_TOLERANCE = "true"

# Linux/macOS
export DAYTRADE_ENV=production
export DAYTRADE_LOG_LEVEL=INFO
export DAYTRADE_MONITORING=true
export DAYTRADE_FAULT_TOLERANCE=true
```

### 2. 設定ファイル確認

`config/production.json`の設定を環境に合わせて調整：

```json
{
  "environment": "production",
  "system": {
    "performance": {
      "ml_analysis_target_seconds": 3.6,
      "memory_limit_mb": 2048,
      "cpu_limit_percent": 80
    },
    "logging": {
      "level": "INFO",
      "log_to_file": true,
      "log_to_console": false
    },
    "monitoring": {
      "enabled": true,
      "interval_seconds": 30
    }
  },
  "security": {
    "data_protection": {
      "encryption_at_rest": true
    }
  }
}
```

### 3. データディレクトリ準備

```bash
mkdir -p data/production
mkdir -p logs/production
mkdir -p reports/production
```

### 4. 依存関係インストール

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ システム起動

### 本番環境起動

```bash
# 本番環境起動
python scripts/start_production.py
```

起動プロセス：
1. 本番環境初期化
2. 設定ファイル検証
3. セキュリティチェック
4. システムコンポーネント初期化
5. パフォーマンス監視開始
6. 自動復旧システム開始
7. メインアプリケーション起動

### 開発環境起動

```bash
# 開発環境起動
python scripts/start_development.py
```

---

## 🛑 システム停止

### 正常停止

```bash
# 正常停止
python scripts/stop_system.py
```

### 強制停止

```bash
# 強制停止
python scripts/stop_system.py force
```

### クリーンアップのみ

```bash
# リソースクリーンアップ
python scripts/stop_system.py clean
```

---

## 📊 システム監視

### 状態確認

```bash
# システム状態確認
python scripts/system_status.py

# JSON形式で出力
python scripts/system_status.py json
```

### 監視項目

1. **プロセス状態**: PID、稼働時間、メモリ・CPU使用率
2. **パフォーマンス**: ML分析速度、処理スループット
3. **ヘルス状態**: 自動復旧システム、デグラデーションレベル
4. **リソース**: ディスク容量、メモリ使用率、ネットワーク接続
5. **ログ**: エラー数、ログファイルサイズ

---

## ⚠️ トラブルシューティング

### よくある問題と解決方法

#### 1. システムが起動しない

**症状**: `start_production.py`実行時にエラー
```
本番環境初期化エラー: ...
```

**対処法**:
1. 環境変数が正しく設定されているか確認
2. 設定ファイル`config/production.json`の構文確認
3. 必要なディレクトリが存在するか確認
4. 依存関係が正しくインストールされているか確認

#### 2. パフォーマンス監視が動作しない

**症状**: 監視機能が利用できない
```
パフォーマンス監視システム初期化スキップ: ...
```

**対処法**:
1. `src/day_trade/utils/performance_monitor.py`が存在するか確認
2. モジュールのインポートパスが正しいか確認
3. 設定で監視が有効になっているか確認

#### 3. 自動復旧システムエラー

**症状**: 自動復旧システムが開始されない
```
自動復旧システム初期化スキップ: ...
```

**対処法**:
1. `src/day_trade/utils/advanced_fault_tolerance.py`が存在するか確認
2. 設定で自動復旧が有効になっているか確認
3. ログファイルでエラー詳細を確認

#### 4. メモリ不足エラー

**症状**: システムがメモリ不足で停止
```
Memory usage exceeded limit
```

**対処法**:
1. `config/production.json`の`memory_limit_mb`を調整
2. システムの物理メモリを確認
3. 他のプロセスのメモリ使用量を確認

#### 5. ネットワークエラー

**症状**: API接続エラー
```
Network connectivity test failed
```

**対処法**:
1. インターネット接続を確認
2. ファイアウォール設定を確認
3. APIエンドポイントの可用性を確認

---

## 🔒 セキュリティ設定

### 本番環境セキュリティチェックリスト

- [ ] データ暗号化有効化
- [ ] アクセス制御設定
- [ ] ログファイル権限設定
- [ ] APIキー環境変数管理
- [ ] 不要ポートの無効化
- [ ] ファイアウォール設定

### APIキー管理

```bash
# 環境変数でAPIキー設定（推奨）
export ALPHA_VANTAGE_API_KEY="your_api_key"
export YAHOO_FINANCE_API_KEY="your_api_key"
```

---

## 📈 パフォーマンス最適化

### 推奨設定

#### 高パフォーマンス設定
```json
{
  "system": {
    "performance": {
      "ml_analysis_target_seconds": 3.6,
      "batch_size": 25,
      "max_concurrent_requests": 3,
      "memory_limit_mb": 4096
    }
  }
}
```

#### 安定性重視設定
```json
{
  "system": {
    "performance": {
      "ml_analysis_target_seconds": 5.0,
      "batch_size": 10,
      "max_concurrent_requests": 2,
      "memory_limit_mb": 2048
    }
  }
}
```

---

## 💾 バックアップとメンテナンス

### 定期バックアップ

1. **設定ファイル**: `config/`ディレクトリ全体
2. **データベース**: `data/production/trading.db`
3. **ログファイル**: `logs/production/`（重要ログのみ）
4. **レポート**: `reports/production/`

### メンテナンス作業

#### 日次作業
- [ ] システム状態確認
- [ ] エラーログチェック
- [ ] パフォーマンス指標確認

#### 週次作業
- [ ] ログファイル整理
- [ ] バックアップ実行
- [ ] システムリソース使用量確認

#### 月次作業
- [ ] 設定ファイル見直し
- [ ] セキュリティアップデート
- [ ] パフォーマンス分析レポート作成

---

## 📞 サポート情報

### ログファイル場所
- **本番ログ**: `logs/production/production.log`
- **構造化ログ**: `logs/production/application.jsonl`
- **エラーログ**: `logs/production/errors.jsonl`
- **パフォーマンスログ**: `logs/production/performance.jsonl`

### 重要な設定パラメータ
- **ML分析目標時間**: 3.6秒（85銘柄）
- **メモリ制限**: 2048MB（本番環境）
- **監視間隔**: 30秒
- **自動復旧タイムアウト**: 5分
- **ログローテーション**: 50MB

---

## 🎉 本番稼働開始

すべてのセットアップが完了したら：

1. **最終確認**
   ```bash
   python scripts/system_status.py
   ```

2. **本番システム起動**
   ```bash
   python scripts/start_production.py
   ```

3. **初期動作確認**
   - システム状態が「RUNNING」
   - パフォーマンス監視が動作中
   - 自動復旧システムが監視中
   - エラーログが空

4. **継続監視開始**
   - 定期的な状態確認
   - パフォーマンス指標監視
   - アラート通知設定

---

**🚀 本番環境の準備が完了しました！**

システムは以下の機能で安定稼働します：
- **高速ML分析**: 85銘柄を3.6秒以内で処理
- **自動復旧機能**: 99.9%可用性を目指す堅牢なシステム
- **リアルタイム監視**: パフォーマンス・ヘルス状態の継続監視
- **構造化ログ**: 問題発生時の迅速な原因特定

本番稼働開始後も、定期的なメンテナンスと監視により、システムの安定性と性能を維持してください。
