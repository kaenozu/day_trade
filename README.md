# Day Trade - 自動デイトレード支援システム

[![CI/CD Pipeline](https://github.com/kaenozu/day_trade/actions/workflows/optimized-ci.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/optimized-ci.yml)
[![Pre-commit Checks](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml)
[![Conflict Detection](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml)
[![codecov](https://codecov.io/gh/kaenozu/day_trade/branch/main/graph/badge.svg)](https://codecov.io/gh/kaenozu/day_trade)
[![Test Coverage](https://img.shields.io/badge/coverage-37.5%25-orange.svg)](https://github.com/kaenozu/day_trade/tree/main/reports/coverage)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**高度なアンサンブル戦略を用いた自動デイトレード支援システム**

Day Tradeは、複数のテクニカル分析手法を組み合わせたアンサンブル戦略により、効率的な株式取引の意思決定を支援するPythonアプリケーションです。リアルタイムデータ取得、高度な分析機能、自動化された取引管理を提供します。

## 🚀 主な機能

### 📊 データ取得・管理
- **リアルタイム株価データ取得** (yfinance API統合)
- **銘柄マスター管理** (東証銘柄情報の自動更新)
- **SQLAlchemy基盤のデータベース管理**
- **データキャッシュ機能** (パフォーマンス最適化)

### 🔍 高度なテクニカル分析
- **アンサンブル戦略エンジン** (複数指標の統合判定)
- **テクニカル指標計算** (RSI, MACD, ボリンジャーバンド等)
- **パターン認識** (トレンド分析, サポート/レジスタンス)
- **ボラティリティ分析** (ATR, VIX相関)
- **出来高分析** (VWAP, OBV)

### 🎯 売買判定・管理
- **統合シグナル生成** (複数戦略の重み付け評価)
- **リスク管理機能** (ストップロス, ポジションサイジング)
- **ポートフォリオ最適化** (分散投資, リバランシング)
- **取引履歴管理** (パフォーマンス追跡)

### 🖥️ ユーザーインターフェース
- **インタラクティブCLI** (rich/prompt_toolkit使用)
- **リアルタイムダッシュボード** (価格監視, アラート)
- **詳細レポート生成** (HTML/JSON/CSV出力)
- **カスタマイズ可能なアラート**

### 🤖 自動化機能
- **バックテスト実行** (戦略検証)
- **自動スクリーニング** (投資機会発見)
- **定期レポート生成**
- **アラート通知システム**

## 📦 インストール

### 必要条件
- Python 3.8以上
- pip (パッケージ管理)
- Git

### クイックスタート

```bash
# リポジトリのクローン
git clone https://github.com/kaenozu/day_trade.git
cd day_trade

# 仮想環境の作成（推奨）
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 依存関係のインストール（推奨）
pip install -e .[dev]

# または、requirementsファイルを使用
pip install -r requirements.txt -r requirements-dev.txt
```

### 開発者向けセットアップ

```bash
# pre-commitフックのインストール
pre-commit install

# データベースの初期化
python -m day_trade.models.database --init

# 動作確認
pytest
python -m day_trade.cli.main --help
```

## 🚀 使用方法

### 基本的な使い方

```bash
# アプリケーションの起動（インタラクティブモード）
python -m day_trade.cli.main

# または
daytrade

# ヘルプの表示
daytrade --help
```

### コマンドライン使用例

```bash
# 特定銘柄の分析
daytrade analyze 7203  # トヨタ自動車

# ウォッチリスト管理
daytrade watchlist add 7203 6758 4755
daytrade watchlist show

# ポートフォリオ分析
daytrade portfolio analyze

# バックテスト実行
daytrade backtest --start-date 2024-01-01 --end-date 2024-12-31

# スクリーニング実行
daytrade screen --strategy momentum --min-volume 1000000

# レポート生成
daytrade report --type portfolio --format html
```

### インタラクティブモードの機能

インタラクティブモードでは以下の機能が利用できます：

- 📈 **リアルタイム価格監視**
- 🔍 **銘柄検索・分析**
- 📊 **ポートフォリオ管理**
- ⚡ **クイック売買判定**
- 📋 **ウォッチリスト操作**
- 🎯 **アラート設定**

## 📁 プロジェクト構造

```
day_trade/
├── src/day_trade/                    # メインアプリケーション
│   ├── analysis/                     # 分析エンジン
│   │   ├── ensemble.py               # アンサンブル戦略コア
│   │   ├── enhanced_ensemble.py      # 強化されたアンサンブル機能
│   │   ├── indicators.py             # テクニカル指標計算
│   │   ├── patterns.py               # チャートパターン認識
│   │   ├── signals.py                # 売買シグナル生成
│   │   ├── backtest.py               # バックテスト機能
│   │   ├── screener.py               # 銘柄スクリーニング
│   │   └── ml_models.py              # 機械学習モデル
│   ├── automation/                   # 自動化・オーケストレーション
│   │   ├── orchestrator.py           # 自動化オーケストレータ
│   │   └── optimized_orchestrator.py # 最適化された自動実行
│   ├── cli/                          # コマンドラインインターフェース
│   │   ├── main.py                   # メインCLIエントリポイント
│   │   ├── enhanced_main.py          # 拡張CLIコマンド
│   │   ├── interactive.py            # インタラクティブモード
│   │   └── watchlist_commands.py     # ウォッチリスト操作
│   ├── core/                         # コア機能・ビジネスロジック
│   │   ├── portfolio.py              # ポートフォリオ管理
│   │   ├── trade_manager.py          # 取引管理・実行
│   │   ├── trade_operations.py       # 取引操作詳細
│   │   ├── watchlist.py              # ウォッチリスト管理
│   │   ├── alerts.py                 # アラート・通知システム
│   │   └── config.py                 # 設定管理
│   ├── data/                         # データ取得・管理
│   │   ├── stock_fetcher.py          # 株価データ取得
│   │   ├── enhanced_stock_fetcher.py # 強化データ取得機能
│   │   └── stock_master.py           # 銘柄マスター管理
│   ├── models/                       # データモデル・永続化
│   │   ├── database.py               # データベース管理
│   │   ├── optimized_database.py     # 最適化DB操作
│   │   ├── stock.py                  # 株式データモデル
│   │   └── base.py                   # ベースモデル定義
│   ├── utils/                        # ユーティリティ・共通機能
│   │   ├── logging_config.py         # ログ設定
│   │   ├── cache_utils.py            # キャッシュ機能
│   │   ├── validators.py             # データ検証
│   │   ├── formatters.py             # 出力フォーマット
│   │   ├── exceptions.py             # カスタム例外
│   │   ├── performance_analyzer.py   # パフォーマンス分析
│   │   ├── transaction_manager.py    # トランザクション管理
│   │   └── api_resilience.py         # API耐性機能
│   └── config/                       # 設定管理
│       └── config_manager.py         # 設定ファイル管理
├── tests/                           # テストスイート
│   ├── test_*.py                    # ユニットテスト
│   └── integration/                 # 統合テスト
│       ├── test_end_to_end_workflow.py # E2Eテスト
│       └── conftest.py              # テスト設定
├── docs/                            # ドキュメント
│   ├── api_resilience.md            # API耐性ドキュメント
│   ├── ensemble_strategy.md         # アンサンブル戦略
│   ├── implementation_plan.md       # 実装計画
│   ├── interactive_mode.md          # インタラクティブモード
│   ├── structured_logging.md        # ログ管理
│   └── transaction_management.md    # トランザクション管理
├── config/                          # 設定ファイル
│   ├── settings.json                # アプリケーション全般設定
│   ├── indicators_config.json       # テクニカル指標パラメータ
│   ├── signal_rules.json            # シグナル生成ルール
│   └── test_settings.json           # テスト環境設定
├── scripts/                         # 運用・開発スクリプト
│   ├── coverage_goals.py            # カバレッジ目標管理
│   ├── coverage_report.py           # カバレッジレポート生成
│   └── dependency_manager.py        # 依存関係管理
├── reports/                         # 生成レポート
│   ├── coverage/                    # カバレッジレポート
│   ├── dependencies/                # 依存関係レポート
│   └── logging/                     # ログ分析結果
├── .github/workflows/               # CI/CDワークフロー
│   ├── optimized-ci.yml             # 最適化CIパイプライン
│   ├── pre-commit.yml               # Pre-commitチェック
│   └── conflict-detection.yml       # コンフリクト検出
└── alembic/                         # データベースマイグレーション
    ├── versions/                    # マイグレーションファイル
    └── env.py                       # Alembic設定
```

### モジュール詳細説明

#### 📊 Analysis モジュール
- **ensemble.py**: 複数の指標を統合したアンサンブル戦略の実装
- **indicators.py**: RSI, MACD, ボリンジャーバンド等のテクニカル指標
- **patterns.py**: チャートパターン（三角持ち合い、ヘッドアンドショルダー等）
- **signals.py**: 買い/売りシグナルの生成と評価
- **backtest.py**: 戦略の過去データでの検証機能

#### 🔄 Core モジュール
- **portfolio.py**: ポートフォリオの構築・管理・最適化
- **trade_manager.py**: 注文管理、リスク管理、ポジション追跡
- **watchlist.py**: 監視銘柄の管理とアラート設定

#### 💾 Data モジュール
- **stock_fetcher.py**: yfinance等からのリアルタイムデータ取得
- **stock_master.py**: 銘柄マスター情報の管理と更新

#### 🛠️ Utils モジュール
- **transaction_manager.py**: 高度なトランザクション管理と最適化
- **api_resilience.py**: API障害時の耐性・リトライ機能
- **cache_utils.py**: パフォーマンス向上のためのキャッシュ機能

## 🏃‍♂️ 開発者向けクイックスタート

### 1. 環境構築（5分）
```bash
# リポジトリクローン
git clone https://github.com/kaenozu/day_trade.git
cd day_trade

# 仮想環境作成・有効化
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 依存関係インストール
pip install -e .[dev]

# Pre-commitセットアップ
pre-commit install
```

### 2. 動作確認（3分）
```bash
# テスト実行
pytest

# カバレッジ確認
pytest --cov=src/day_trade --cov-report=html

# アプリケーション起動
python -m day_trade.cli.main --help
```

### 3. 開発開始（即座に）
```bash
# 新機能ブランチ作成
git checkout -b feature/your-feature

# 開発作業
# ... コード変更 ...

# 品質チェック
ruff check . --fix
pytest

# コミット（pre-commitが自動実行）
git add .
git commit -m "feat: 新機能の説明"

# プッシュ・プルリクエスト
git push origin feature/your-feature
gh pr create
```

### 4. 主要な開発タスク

#### 新しいテクニカル指標の追加
```python
# src/day_trade/analysis/indicators.py に追加
def your_indicator(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """新しいテクニカル指標の実装"""
    # 実装コード
    return result

# tests/test_indicators.py にテスト追加
def test_your_indicator():
    # テストコード
    pass
```

#### 新しい戦略の実装
```python
# src/day_trade/analysis/ensemble.py に追加
class YourStrategy(BaseStrategy):
    """新しい戦略の実装"""

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        # シグナル生成ロジック
        pass
```

#### CLI コマンドの追加
```python
# src/day_trade/cli/main.py に追加
@cli.command()
@click.option('--param', help='パラメータの説明')
def your_command(param):
    """新しいコマンドの説明"""
    # コマンド実装
    pass
```

### 5. デバッグ・トラブルシューティング

#### ログ確認
```bash
# ログファイル確認
tail -f daytrade_$(date +%Y%m%d).log

# デバッグモードで実行
LOG_LEVEL=DEBUG python -m day_trade.cli.main
```

#### データベース操作
```bash
# データベース初期化
python -m day_trade.models.database --reset

# マイグレーション実行
alembic upgrade head

# テストデータ投入
python scripts/setup_test_data.py
```

#### パフォーマンス分析
```bash
# カバレッジ目標確認
python scripts/coverage_goals.py

# 依存関係分析
python scripts/dependency_manager.py --check
```

## ⚙️ 設定とカスタマイズ

### 設定ファイル

プロジェクトでは以下の設定ファイルを使用します：

- `config/settings.json`: アプリケーション全般設定
- `config/indicators_config.json`: テクニカル指標パラメータ
- `config/signal_rules.json`: シグナル生成ルール

### 環境変数

```bash
# データベース設定
DATABASE_URL=sqlite:///day_trade.db

# APIキー（必要に応じて）
ALPHA_VANTAGE_API_KEY=your_api_key

# ログレベル
LOG_LEVEL=INFO
```

## 🧪 テスト実行

```bash
# 全テスト実行
pytest

# カバレッジ付きテスト
pytest --cov=src/day_trade --cov-report=html

# 特定のテストファイル
pytest tests/test_ensemble.py

# 詳細出力
pytest -v
```

テストカバレッジレポートは `htmlcov/index.html` で確認できます。

## 📊 パフォーマンス分析

### バックテスト結果例

```python
# バックテスト設定
config = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 1000000,
    "strategies": ["momentum", "mean_reversion", "volatility"]
}

# 結果（例）
total_return: 15.3%
sharpe_ratio: 1.42
max_drawdown: -8.7%
win_rate: 64.2%
```

### 戦略パフォーマンス

| 戦略 | 年間リターン | シャープレシオ | 最大ドローダウン |
|------|-------------|---------------|----------------|
| アンサンブル | 15.3% | 1.42 | -8.7% |
| モメンタム | 12.1% | 1.18 | -12.3% |
| 平均回帰 | 8.9% | 0.95 | -6.4% |
| ボラティリティ | 11.7% | 1.24 | -9.8% |

## 🔧 開発者向け情報

### システムアーキテクチャ

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Web Interface  │    │   API Gateway   │
│  (interactive)  │    │   (future)      │    │   (future)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                        Core Engine                                │
├─────────────────┬───────────────┼───────────────┬─────────────────┤
│  Data Manager   │ Analysis Engine│ Portfolio Mgr │  Trade Manager  │
│                 │               │               │                 │
│ • Stock Fetcher │ • Indicators  │ • Positions   │ • Order Mgmt    │
│ • Cache System  │ • Patterns    │ • Risk Mgmt   │ • Execution     │
│ • DB Management │ • Ensemble    │ • Rebalancing │ • History       │
└─────────────────┴───────────────┴───────────────┴─────────────────┘
```

### 詳細アーキテクチャ構成

#### 1. ユーザーインターフェース層
```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   CLI Module    │  Interactive    │      Future Extensions      │
│                 │     Mode        │                             │
│ • main.py       │ • rich UI       │ • Web Dashboard             │
│ • commands      │ • prompt_toolkit│ • REST API                  │
│ • formatters    │ • real-time     │ • WebSocket                 │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

#### 2. ビジネスロジック層
```
┌─────────────────────────────────────────────────────────────────┐
│                      Business Logic Layer                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Core Module   │ Analysis Module │    Automation Module        │
│                 │                 │                             │
│ • Portfolio     │ • Ensemble      │ • Orchestrator              │
│ • TradeManager  │ • Indicators    │ • Scheduler                 │
│ • Watchlist     │ • Patterns      │ • Report Generator          │
│ • Alerts        │ • Backtest      │ • Auto Screening            │
│ • Config        │ • Signals       │ • Notification System       │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

#### 3. データアクセス層
```
┌─────────────────────────────────────────────────────────────────┐
│                       Data Access Layer                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Data Module   │ Models Module   │       Utils Module          │
│                 │                 │                             │
│ • StockFetcher  │ • Database      │ • Cache Utils               │
│ • StockMaster   │ • Stock Models  │ • Validators                │
│ • API Clients   │ • SQLAlchemy    │ • Error Handlers            │
│ • Rate Limiters │ • Migrations    │ • Performance Analyzers     │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

#### 4. 外部システム連携
```
┌─────────────────────────────────────────────────────────────────┐
│                    External Systems                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Data Sources  │   Databases     │       Monitoring            │
│                 │                 │                             │
│ • yfinance API  │ • SQLite        │ • Logging System            │
│ • Yahoo Finance │ • PostgreSQL    │ • Performance Metrics       │
│ • Alpha Vantage │ • Redis Cache   │ • Error Tracking            │
│ • Future APIs   │ • File Storage  │ • Health Checks             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### データフロー図

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    User     │───▶│     CLI     │───▶│    Core     │───▶│  Database   │
│  Commands   │    │  Interface  │    │   Engine    │    │   Storage   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                               │
                                               ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  External   │◀───│    Data     │◀───│  Analysis   │───▶│   Reports   │
│  APIs       │    │  Fetcher    │    │   Engine    │    │  & Alerts   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 技術スタック詳細

#### コア技術
- **言語**: Python 3.8+
- **フレームワーク**: SQLAlchemy (ORM), Click (CLI), Rich (UI)
- **データベース**: SQLite (開発), PostgreSQL (本番対応)
- **API**: yfinance, pandas-datareader

#### 開発・運用
- **テスト**: pytest, pytest-cov, unittest.mock
- **品質**: ruff (lint), mypy (type), bandit (security)
- **CI/CD**: GitHub Actions (最適化済み)
- **依存管理**: pyproject.toml, pip-tools

### 貢献方法

1. **フォーク**: このリポジトリをフォーク
2. **ブランチ作成**: `git checkout -b feature/amazing-feature`
3. **コミット**: `git commit -m 'feat: Add amazing feature'`
4. **プッシュ**: `git push origin feature/amazing-feature`
5. **プルリクエスト**: プルリクエストを作成

詳細は [CONTRIBUTING.md](CONTRIBUTING.md) をご覧ください。

### コード品質

このプロジェクトでは以下のツールで品質を保証しています：

- **Ruff**: リンターとフォーマッター
- **MyPy**: 型チェック
- **Bandit**: セキュリティ検査
- **pytest**: テストフレームワーク
- **pre-commit**: Git フック

## 📚 ドキュメント

### 🎯 新規参画者向け学習パス

#### ステップ1: 基礎理解（30分）
1. **README.md** - プロジェクト全体概要
2. **[CONTRIBUTING.md](CONTRIBUTING.md)** - 開発ガイドライン
3. **[DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)** - 依存関係の理解

#### ステップ2: アーキテクチャ理解（45分）
1. **[docs/implementation_plan.md](docs/implementation_plan.md)** - 設計思想
2. **[docs/ensemble_strategy.md](docs/ensemble_strategy.md)** - コア戦略
3. **プロジェクト構造** - 上記の詳細説明を参照

#### ステップ3: 実践開発（60分）
1. **開発者向けクイックスタート** - 上記ガイドに従って環境構築
2. **[docs/interactive_mode.md](docs/interactive_mode.md)** - UI機能理解
3. **[TESTING.md](TESTING.md)** - テスト手法の習得

#### ステップ4: 高度な機能（90分）
1. **[docs/transaction_management.md](docs/transaction_management.md)** - DB操作
2. **[docs/structured_logging.md](docs/structured_logging.md)** - ログ管理
3. **[docs/api_resilience.md](docs/api_resilience.md)** - 耐性設計

### 📖 詳細ドキュメント

#### 技術ドキュメント
- **[アンサンブル戦略](docs/ensemble_strategy.md)** - 複数指標統合戦略
- **[実装計画](docs/implementation_plan.md)** - アーキテクチャ設計
- **[トランザクション管理](docs/transaction_management.md)** - DB最適化
- **[API耐性機能](docs/api_resilience.md)** - 外部API障害対応

#### 運用ドキュメント
- **[インタラクティブモード](docs/interactive_mode.md)** - CLI操作ガイド
- **[構造化ログ](docs/structured_logging.md)** - ログ分析・監視
- **[テスト実行ガイド](TESTING.md)** - テスト戦略
- **[依存関係管理](DEPENDENCY_MANAGEMENT.md)** - パッケージ管理

#### 開発支援
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - 貢献ガイドライン
- **[トラブルシューティング](docs/troubleshooting.md)** - 問題解決
- **[開発者ガイド](docs/developer_guide.md)** - 開発ベストプラクティス

### 🎓 学習リソース

#### 初心者向け
- **Python基礎**: [Python公式チュートリアル](https://docs.python.org/ja/3/tutorial/)
- **SQLAlchemy**: [SQLAlchemy Tutorial](https://docs.sqlalchemy.org/en/14/tutorial/)
- **pytest**: [pytest Documentation](https://docs.pytest.org/en/stable/)

#### 中級者向け
- **テクニカル分析**: [TA-Lib Documentation](https://ta-lib.org/)
- **パンダス**: [pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- **非同期処理**: [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)

#### 上級者向け
- **機械学習**: [scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)
- **パフォーマンス最適化**: [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed)
- **アーキテクチャパターン**: [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

## 🐛 トラブルシューティング

### よくある問題

1. **データ取得エラー**
   ```bash
   # ネットワーク接続を確認
   ping finance.yahoo.com

   # APIキーを確認（必要な場合）
   echo $ALPHA_VANTAGE_API_KEY
   ```

2. **データベースエラー**
   ```bash
   # データベースの再初期化
   python -m day_trade.models.database --reset
   ```

3. **依存関係エラー**
   ```bash
   # 仮想環境の再作成
   del venv  # Windows: rmdir /s venv
   python -m venv venv
   venv\Scripts\activate
   pip install -e .[dev]
   ```

### ログファイル

アプリケーションログは以下の場所に保存されます：
- `daytrade_YYYYMMDD.log`: 日次ログファイル
- `logs/`: 詳細ログディレクトリ

## 🤝 コミュニティ

- **GitHub Issues**: バグ報告・機能要求
- **GitHub Discussions**: 質問・議論
- **Pull Requests**: コード貢献

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

## 🙏 謝辞

- **yfinance**: 株価データAPIの提供
- **pandas**: データ分析基盤
- **SQLAlchemy**: データベースORM
- **rich**: 美しいCLI出力
- **pytest**: テストフレームワーク

---

**⚠️ 免責事項**: このソフトウェアは教育・研究目的で提供されています。投資判断は自己責任で行ってください。開発者は投資結果に対する責任を負いません。

**📈 Happy Trading!** 質問や提案があれば、お気軽にIssueを作成してください。
## 最新の更新履歴

- マージコンフリクトの解消と新機能の統合
  - 影響ファイル: `src/day_trade/analysis/enhanced_ensemble.py`, `src/day_trade/analysis/patterns.py`, `src/day_trade/analysis/signals.py`, `src/day_trade/automation/orchestrator.py`