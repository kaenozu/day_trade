# Day Trade - 自動デイトレード支援システム

[![CI/CD Pipeline](https://github.com/kaenozu/day_trade/actions/workflows/optimized-ci.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/optimized-ci.yml)
[![Pre-commit Checks](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml)
[![Conflict Detection](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml)
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
├── src/day_trade/           # メインアプリケーション
│   ├── analysis/            # 分析エンジン
│   │   ├── ensemble.py      # アンサンブル戦略
│   │   ├── indicators.py    # テクニカル指標
│   │   ├── patterns.py      # パターン認識
│   │   └── signals.py       # シグナル生成
│   ├── automation/          # 自動化機能
│   ├── cli/                 # コマンドラインインターフェース
│   ├── core/                # コア機能
│   │   ├── portfolio.py     # ポートフォリオ管理
│   │   ├── trade_manager.py # 取引管理
│   │   └── watchlist.py     # ウォッチリスト
│   ├── data/                # データ取得・管理
│   ├── models/              # データモデル
│   └── utils/               # ユーティリティ
├── tests/                   # テストスイート
├── docs/                    # ドキュメント
├── config/                  # 設定ファイル
└── reports/                 # 生成レポート
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

### アーキテクチャ

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Web Interface  │    │   API Gateway   │
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

詳細なドキュメントは `docs/` ディレクトリにあります：

- [アンサンブル戦略](docs/ensemble_strategy.md)
- [実装計画](docs/implementation_plan.md)
- [インタラクティブモード](docs/interactive_mode.md)
- [構造化ログ](docs/structured_logging.md)
- [トランザクション管理](docs/transaction_management.md)
- [テスト実行ガイド](TESTING.md)
- [依存関係管理](DEPENDENCY_MANAGEMENT.md)

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
