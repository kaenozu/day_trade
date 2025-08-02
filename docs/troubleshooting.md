# Day Trade トラブルシューティングガイド

## 目次

1. [一般的な問題と解決方法](#一般的な問題と解決方法)
2. [インストール・環境問題](#インストール・環境問題)
3. [データ取得問題](#データ取得問題)
4. [データベース問題](#データベース問題)
5. [パフォーマンス問題](#パフォーマンス問題)
6. [アラート・通知問題](#アラート・通知問題)
7. [診断ツール](#診断ツール)
8. [ログ分析](#ログ分析)
9. [よくある質問（FAQ）](#よくある質問faq)

## 一般的な問題と解決方法

### コマンドが見つからない

#### 症状
```bash
$ daytrade
bash: daytrade: command not found
```

#### 原因と解決方法

**1. 仮想環境が有効化されていない**
```bash
# 解決方法: 仮想環境を有効化
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 確認
which python
which daytrade
```

**2. パッケージがインストールされていない**
```bash
# 解決方法: パッケージをインストール
pip install -e .[dev]

# または
pip install -e .
```

**3. パスが通っていない**
```bash
# 代替方法: モジュールとして実行
python -m day_trade.cli.main
```

### モジュールが見つからない

#### 症状
```python
ModuleNotFoundError: No module named 'day_trade'
```

#### 解決方法
```bash
# 1. Pythonパスの確認
python -c "import sys; print('\n'.join(sys.path))"

# 2. パッケージのインストール確認
pip list | grep day-trade

# 3. 開発モードでインストール
pip install -e .

# 4. 環境変数の設定（一時的）
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## インストール・環境問題

### Python バージョン問題

#### 症状
```bash
ERROR: Python 3.8 or higher is required
```

#### 解決方法
```bash
# 1. Pythonバージョン確認
python --version
python3 --version

# 2. 適切なバージョンのインストール
# Windows: Python.orgからダウンロード
# Linux: パッケージマネージャーを使用
sudo apt update && sudo apt install python3.9
# macOS: Homebrewを使用
brew install python@3.9

# 3. 仮想環境の再作成
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
# または
venv\Scripts\activate     # Windows
```

### 依存関係の競合

#### 症状
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

#### 解決方法
```bash
# 1. 仮想環境の完全な再作成
rm -rf venv  # Linux/macOS
# または
rmdir /s venv  # Windows

python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install --upgrade pip

# 2. 依存関係の段階的インストール
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. 競合解決ツールの使用
pip install pip-tools
pip-compile requirements.in
```

### 権限問題

#### 症状（Windows）
```bash
PermissionError: [WinError 5] Access is denied
```

#### 解決方法
```bash
# 1. 管理者権限でコマンドプロンプト実行
# 2. ユーザーディレクトリにインストール
pip install --user -e .

# 3. 仮想環境の使用（推奨）
python -m venv venv
venv\Scripts\activate
```

#### 症状（Linux/macOS）
```bash
Permission denied: '/usr/local/lib/python3.x/site-packages'
```

#### 解決方法
```bash
# 1. 仮想環境の使用（推奨）
python3 -m venv venv
source venv/bin/activate

# 2. ユーザーインストール
pip install --user -e .

# 3. sudoの使用（非推奨）
sudo pip install -e .
```

## データ取得問題

### ネットワーク接続エラー

#### 症状
```bash
NetworkError: ネットワーク接続エラー: HTTPSConnectionPool
```

#### 診断手順
```bash
# 1. 基本的なネットワーク接続確認
ping 8.8.8.8
ping finance.yahoo.com

# 2. DNS解決確認
nslookup finance.yahoo.com

# 3. HTTPSアクセス確認
curl -I https://finance.yahoo.com

# 4. プロキシ設定確認
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

#### 解決方法
```bash
# 1. プロキシ設定（企業環境）
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=https://proxy.company.com:8080

# 2. pip設定でプロキシ対応
pip install --proxy http://proxy.company.com:8080 -e .

# 3. 証明書問題の回避（一時的）
pip install --trusted-host pypi.org --trusted-host pypi.python.org -e .
```

### API制限・レート制限

#### 症状
```bash
RateLimitError: レート制限エラー: 429 Too Many Requests
```

#### 解決方法
```python
# 1. リクエスト間隔の調整
from src.day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher

fetcher = EnhancedStockFetcher(
    retry_count=5,
    retry_delay=2.0  # 遅延を増加
)

# 2. 設定ファイルでの調整
# config/settings.json
{
    "api": {
        "request_delay": 2.0,
        "max_requests_per_minute": 30
    }
}
```

### データ品質問題

#### 症状
```bash
DataError: 無効なAPIレスポンス: データが見つかりません
```

#### 診断・解決方法
```python
# 1. 手動でのデータ確認
from src.day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher

fetcher = EnhancedStockFetcher()

# ヘルスチェック実行
health = fetcher.health_check()
print(f"ステータス: {health['status']}")

# システム状態確認
status = fetcher.get_system_status()
for endpoint in status['resilience_status']['endpoints']:
    print(f"{endpoint['name']}: {endpoint['circuit_state']}")

# 2. 劣化モードでの継続
# 自動的にフォールバック機能が働く
```

## データベース問題

### データベース接続エラー

#### 症状
```bash
DatabaseConnectionError: データベース接続エラー
```

#### 解決方法
```bash
# 1. データベースファイルの確認
ls -la day_trade.db

# 2. データベースの初期化
python -m day_trade.models.database --init

# 3. 権限確認・修正（Linux/macOS）
chmod 644 day_trade.db

# 4. データベースURLの確認
echo $DATABASE_URL
```

### データベース破損

#### 症状
```bash
DatabaseIntegrityError: データベース整合性エラー
```

#### 解決方法
```bash
# 1. データベースの整合性チェック
sqlite3 day_trade.db "PRAGMA integrity_check;"

# 2. バックアップからの復元
cp day_trade.db.backup day_trade.db

# 3. データベースの再作成
python -m day_trade.models.database --reset

# 4. 手動修復（慎重に）
sqlite3 day_trade.db "REINDEX;"
```

### マイグレーション問題

#### 症状
```bash
DatabaseOperationalError: table already exists
```

#### 解決方法
```python
# 1. マイグレーション状態の確認
from src.day_trade.models.database import DatabaseManager

db_manager = DatabaseManager()
db_manager.check_schema_version()

# 2. 強制的なスキーマ更新
db_manager.update_schema(force=True)

# 3. 手動でのテーブル削除・再作成
python -m day_trade.models.database --drop-tables
python -m day_trade.models.database --init
```

## パフォーマンス問題

### 処理速度が遅い

#### 症状
- 分析処理に時間がかかりすぎる
- バックテストが完了しない

#### 診断方法
```python
# 1. パフォーマンス分析の実行
from src.day_trade.utils.performance_analyzer import PerformanceProfiler

profiler = PerformanceProfiler()

@profiler.profile_function
def slow_function():
    # 遅い処理
    pass

# 結果確認
stats = profiler.get_stats()
print(stats)

# 2. メモリ使用量確認
import psutil
import os

process = psutil.Process(os.getpid())
print(f"メモリ使用量: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

#### 解決方法
```python
# 1. キャッシュ設定の最適化
from src.day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher

fetcher = EnhancedStockFetcher(
    price_cache_ttl=300,      # 5分間キャッシュ
    historical_cache_ttl=1800 # 30分間キャッシュ
)

# 2. 並列処理の有効化
from concurrent.futures import ThreadPoolExecutor

def analyze_multiple_stocks(symbols):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(analyze_single_stock, symbols))
    return results

# 3. データ処理の最適化
import pandas as pd

# 悪い例
data = []
for row in large_dataset:
    processed_row = process_row(row)
    data.append(processed_row)

# 良い例
df = pd.DataFrame(large_dataset)
processed_df = df.apply(process_row_vectorized, axis=1)
```

### メモリ不足

#### 症状
```bash
MemoryError: Unable to allocate array
```

#### 解決方法
```python
# 1. チャンクサイズの調整
def process_large_dataset(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield process_chunk(chunk)

# 2. 明示的なメモリ解放
import gc

def memory_intensive_function():
    large_data = load_large_data()
    result = process_data(large_data)

    del large_data
    gc.collect()

    return result

# 3. メモリ効率的なデータ型使用
df = pd.read_csv('data.csv')
df = df.astype({
    'int_column': 'int32',      # int64 → int32
    'float_column': 'float32'   # float64 → float32
})
```

## アラート・通知問題

### アラートが発火しない

#### 症状
- 設定したアラートが通知されない
- アラート履歴に記録されない

#### 診断方法
```bash
# 1. アラート設定の確認
daytrade alert list --verbose

# 2. アラート履歴の確認
daytrade alert history --last 24h

# 3. テストアラートの送信
daytrade alert test --type price --symbol 7203
```

#### 解決方法
```python
# 1. アラート設定の修正
from src.day_trade.core.alerts import AlertManager

alert_manager = AlertManager()

# 閾値の調整
alert_manager.set_price_alert(
    symbol="7203",
    above_price=2800,  # 現在価格より高い値に設定
    below_price=2400   # 現在価格より低い値に設定
)

# 2. ログレベルの調整
import logging
logging.getLogger('day_trade.alerts').setLevel(logging.DEBUG)

# 3. アラート機能の有効化確認
# config/settings.json
{
    "alerts": {
        "enabled": true,
        "sound_enabled": true,
        "email_enabled": false
    }
}
```

## 診断ツール

### システム診断スクリプト

```python
# scripts/diagnose.py
#!/usr/bin/env python3
"""システム診断スクリプト"""

import sys
import os
import platform
import subprocess
from pathlib import Path

def check_python_version():
    """Pythonバージョンチェック"""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8以上が必要です")
        return False
    else:
        print("✅ Pythonバージョン OK")
        return True

def check_dependencies():
    """依存関係チェック"""
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'sqlalchemy',
        'rich', 'click', 'structlog'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} インストール済み")
        except ImportError:
            print(f"❌ {package} 未インストール")
            missing.append(package)

    return len(missing) == 0

def check_database():
    """データベースチェック"""
    db_path = Path("day_trade.db")

    if db_path.exists():
        print(f"✅ データベースファイル存在: {db_path.stat().st_size} bytes")

        # 簡単な接続テスト
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✅ テーブル数: {len(tables)}")
            conn.close()
            return True
        except Exception as e:
            print(f"❌ データベース接続エラー: {e}")
            return False
    else:
        print("❌ データベースファイルが見つかりません")
        return False

def check_network():
    """ネットワーク接続チェック"""
    try:
        import requests
        response = requests.get("https://finance.yahoo.com", timeout=10)
        if response.status_code == 200:
            print("✅ Yahoo Finance アクセス OK")
            return True
        else:
            print(f"❌ Yahoo Finance アクセスエラー: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ネットワークエラー: {e}")
        return False

def check_permissions():
    """権限チェック"""
    test_file = Path("test_write.tmp")

    try:
        test_file.write_text("test")
        test_file.unlink()
        print("✅ 書き込み権限 OK")
        return True
    except Exception as e:
        print(f"❌ 書き込み権限エラー: {e}")
        return False

def main():
    """メイン診断関数"""
    print("Day Trade システム診断")
    print("=" * 50)

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"作業ディレクトリ: {os.getcwd()}")
    print()

    checks = [
        ("Pythonバージョン", check_python_version),
        ("依存関係", check_dependencies),
        ("データベース", check_database),
        ("ネットワーク", check_network),
        ("権限", check_permissions),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}チェック:")
        result = check_func()
        results.append((name, result))

    print("\n" + "=" * 50)
    print("診断結果:")

    all_ok = True
    for name, result in results:
        status = "✅ OK" if result else "❌ NG"
        print(f"{name}: {status}")
        if not result:
            all_ok = False

    if all_ok:
        print("\n🎉 全ての診断項目がOKです！")
    else:
        print("\n⚠️  問題が検出されました。上記のエラーを確認してください。")

if __name__ == "__main__":
    main()
```

### 実行方法
```bash
# 診断スクリプトの実行
python scripts/diagnose.py

# 詳細な環境情報収集
python -c "
import sys
import platform
import pkg_resources

print('Python:', sys.version)
print('Platform:', platform.platform())
print('Architecture:', platform.architecture())
print()
print('Installed packages:')
for pkg in sorted(pkg_resources.working_set, key=lambda x: x.project_name):
    print(f'{pkg.project_name}=={pkg.version}')
"
```

## ログ分析

### ログファイルの場所

```bash
# 日次ログファイル
daytrade_$(date +%Y%m%d).log

# 詳細ログディレクトリ
logs/
├── application.log
├── error.log
├── performance.log
└── security.log
```

### ログ分析コマンド

```bash
# 1. エラーログの抽出
grep "ERROR" daytrade_$(date +%Y%m%d).log

# 2. 特定の機能に関するログ
grep "stock_fetcher" daytrade_$(date +%Y%m%d).log

# 3. パフォーマンス情報
grep "Performance metric" daytrade_$(date +%Y%m%d).log

# 4. アラート関連
grep "alert" daytrade_$(date +%Y%m%d).log | grep -i "error\|warning"

# 5. 最新のエラー（最後の10件）
grep "ERROR" daytrade_$(date +%Y%m%d).log | tail -10

# 6. JSON形式ログの分析（jqコマンド使用）
cat logs/application.log | jq 'select(.level == "error")'
cat logs/application.log | jq 'select(.module == "stock_fetcher") | .message'
```

### ログレベルの調整

```bash
# デバッグレベルでの実行
export LOG_LEVEL=DEBUG
daytrade analyze 7203

# 特定モジュールのログレベル調整
export LOG_LEVEL_STOCK_FETCHER=DEBUG
export LOG_LEVEL_PORTFOLIO=INFO
```

## よくある質問（FAQ）

### Q1: インストール後にコマンドが見つからない

**A**: 仮想環境の有効化を確認してください。
```bash
# 仮想環境の有効化
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 代替方法
python -m day_trade.cli.main
```

### Q2: データが取得できない

**A**: ネットワーク接続とAPIの状態を確認してください。
```bash
# ネットワーク確認
ping finance.yahoo.com

# ヘルスチェック実行
daytrade health-check

# デバッグモードで実行
daytrade --debug analyze 7203
```

### Q3: バックテストが遅い

**A**: 期間やデータ量を調整してください。
```bash
# 期間を短縮
daytrade backtest 7203 --start-date 2024-06-01 --end-date 2024-12-31

# 並列処理有効化
daytrade backtest 7203 --parallel

# キャッシュクリア後再実行
daytrade cache clear
daytrade backtest 7203
```

### Q4: ポートフォリオの計算が合わない

**A**: データの整合性を確認してください。
```bash
# ポートフォリオデータの確認
daytrade portfolio show --detailed

# データベースの整合性チェック
python -m day_trade.models.database --check

# 手動での再計算
daytrade portfolio recalculate
```

### Q5: アラートが届かない

**A**: アラート設定と条件を確認してください。
```bash
# アラート設定一覧
daytrade alert list

# テストアラート送信
daytrade alert test

# アラート条件の緩和
daytrade alert price 7203 --above 2500 --below 2700  # 幅を広げる
```

### Q6: メモリ使用量が多い

**A**: キャッシュ設定とデータ処理を最適化してください。
```python
# config/settings.json
{
    "cache": {
        "max_size": 500,        # デフォルト1000から削減
        "ttl": 300             # 5分間キャッシュ
    }
}
```

### Q7: 開発環境でテストが失敗する

**A**: 依存関係とテストデータを確認してください。
```bash
# 依存関係の再インストール
pip install -r requirements-dev.txt

# テストデータの準備
python -m day_trade.models.database --init-test-data

# 個別テスト実行
pytest tests/unit/test_indicators.py -v
```

---

このガイドで解決しない問題がある場合は、以下の情報を含めてIssueを作成してください：

1. 症状の詳細説明
2. エラーメッセージの全文
3. 環境情報（OS、Pythonバージョン、パッケージバージョン）
4. 再現手順
5. 関連するログファイル（該当部分）

迅速なサポートのため、`scripts/diagnose.py`の実行結果も添付してください。
