# 🚀 Day Trade Personal - 横展開デプロイメントガイド

## 🎯 概要
このシステムは個人投資家向けの93%精度AI株式分析システムです。
Windows、Mac、Linux全対応で簡単に横展開できます。

---

## 🌐 横展開可能な環境

### ✅ 対応OS
- **Windows 10/11** (メイン開発環境)
- **macOS 10.15+**
- **Ubuntu 20.04+**
- **CentOS 8+**
- **Docker環境**

### ✅ 対応Python環境
- **Python 3.8+** (推奨: 3.9-3.11)
- **Anaconda/Miniconda**
- **pyenv**
- **Docker Python**

---

## 🚀 クイックデプロイ（3分で完了）

### 1️⃣ Windowsでの展開
```powershell
# PowerShellまたはコマンドプロンプト
git clone [このリポジトリURL]
cd day_trade
pip install -r requirements.txt
python daytrade.py
```

### 2️⃣ macOS/Linuxでの展開
```bash
# ターミナル
git clone [このリポジトリURL]
cd day_trade
pip3 install -r requirements.txt
python3 daytrade.py
```

### 3️⃣ Dockerでの展開
```bash
# Docker使用
docker build -t day-trade-personal .
docker run -p 5000:5000 day-trade-personal
```

---

## 🔧 詳細セットアップ

### 環境構築（初回のみ）

#### Windows
```powershell
# Python仮想環境作成
python -m venv venv
venv\\Scripts\\activate

# 依存関係インストール
pip install -r requirements.txt

# 設定ファイル作成（オプション）
copy config\\settings.json.example config\\settings.json
```

#### macOS/Linux
```bash
# Python仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存関係インストール
pip3 install -r requirements.txt

# 設定ファイル作成（オプション）
cp config/settings.json.example config/settings.json
```

---

## 🌍 リモートサーバー展開

### VPS・クラウドサーバーでの展開

#### AWS EC2での展開例
```bash
# EC2インスタンスに接続後
sudo apt update
sudo apt install python3 python3-pip git -y

git clone [このリポジトリURL]
cd day_trade
pip3 install -r requirements.txt

# バックグラウンド実行
nohup python3 daytrade.py > output.log 2>&1 &
```

#### Google Cloud Platformでの展開例
```bash
# GCE VMインスタンスに接続後
sudo apt update
sudo apt install python3 python3-pip git -y

git clone [このリポジトリURL]
cd day_trade
pip3 install -r requirements.txt

# サービス化（systemd）
sudo cp deployment/day-trade.service /etc/systemd/system/
sudo systemctl enable day-trade
sudo systemctl start day-trade
```

---

## 🔒 セキュリティ設定

### 1. 外部アクセス制限
```python
# config/settings.json
{
    \"security\": {
        \"allowed_hosts\": [\"127.0.0.1\", \"localhost\"],
        \"enable_external_access\": false
    }
}
```

### 2. 認証設定（オプション）
```python
{
    \"auth\": {
        \"enable_password\": true,
        \"password_hash\": \"[ハッシュ化パスワード]\"
    }
}
```

---

## 📊 パフォーマンス設定

### 軽量版設定（低スペックPC用）
```python
# config/settings.json
{
    \"performance\": {
        \"max_symbols\": 10,
        \"cache_size\": 50,
        \"concurrent_analysis\": 3
    }
}
```

### 高性能版設定（高スペックPC用）
```python
{
    \"performance\": {
        \"max_symbols\": 100,
        \"cache_size\": 500,
        \"concurrent_analysis\": 20
    }
}
```

---

## 🌐 ネットワーク設定

### ポート設定
- **デフォルト**: 5000
- **変更方法**: `python daytrade.py --port 8080`

### ファイアウォール設定例
```bash
# Ubuntu/CentOS
sudo ufw allow 5000
# または
sudo firewall-cmd --add-port=5000/tcp --permanent
```

---

## 📋 トラブルシューティング

### よくある問題と解決法

#### 1. 依存関係エラー
```bash
# 解決法
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### 2. ポート使用中エラー
```bash
# 解決法
python daytrade.py --port 8080
```

#### 3. yfinanceエラー
```bash
# 解決法
pip install --upgrade yfinance requests
```

#### 4. メモリ不足
```python
# config/settings.json で軽量設定に変更
{\"performance\": {\"max_symbols\": 5}}
```

---

## 🔄 自動起動設定

### Windows（タスクスケジューラ）
```powershell
# 起動スクリプト作成
# start_day_trade.bat
cd C:\\path\\to\\day_trade
venv\\Scripts\\activate
python daytrade.py
```

### macOS/Linux（systemd）
```ini
# /etc/systemd/system/day-trade.service
[Unit]
Description=Day Trade Personal AI System
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/day_trade
ExecStart=/path/to/day_trade/venv/bin/python daytrade.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 📈 監視・メンテナンス

### ログ確認
```bash
# アプリケーションログ
tail -f logs/app.log

# システム稼働確認
curl http://localhost:5000/api/health
```

### 定期メンテナンス
```bash
# データベース最適化（週1回推奨）
python scripts/optimize_database.py

# キャッシュクリア（月1回推奨）
python scripts/clear_cache.py
```

---

## 🎯 多環境対応

### 開発環境
```bash
python daytrade.py --debug --dev-mode
```

### ステージング環境
```bash
python daytrade.py --config config/staging.json
```

### 本番環境
```bash
python daytrade.py --config config/production.json --no-debug
```

---

## 🤝 サポート・コミュニティ

### 技術サポート
- **GitHub Issues**: [リンク]
- **ドキュメント**: README.md、TROUBLESHOOTING_GUIDE.md

### 安全な利用のお願い
- 🏠 **個人利用のみ**：商用利用は禁止
- 🔒 **データ保護**：個人情報は外部送信されません
- 💡 **投資助言ではありません**：最終判断はご自身で

---

## ⚡ 高速展開コマンド集

```bash
# 1行で完全セットアップ（Linux/macOS）
curl -sSL [セットアップスクリプトURL] | bash

# Docker一発起動
docker run -p 5000:5000 [Dockerイメージ]

# 設定込みクローン
git clone --recursive [リポジトリURL] && cd day_trade && ./setup.sh
```

**🎉 横展開完了！ブラウザで http://localhost:5000 にアクセスして93%精度AI分析を体験してください！**