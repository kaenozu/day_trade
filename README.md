# Day Trade - デイトレード支援アプリ

[![CI/CD Pipeline](https://github.com/kaenozu/day_trade/actions/workflows/ci.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/ci.yml)
[![Pre-commit Checks](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml)
[![Conflict Detection](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml)

CUIベースのデイトレード支援アプリケーション

## 機能

- リアルタイム株価データ取得
- テクニカル分析指標の計算
- 売買記録の管理
- ポートフォリオ分析
- アラート機能

## インストール

```bash
# 依存関係のインストール
pip install -r requirements.txt

# 開発モードでインストール
pip install -e .
```

## 使用方法

```bash
# アプリケーションの起動
daytrade

# ヘルプの表示
daytrade --help

# 特定の銘柄の情報を表示
daytrade stock 7203  # トヨタ自動車

# ウォッチリストの表示
daytrade watchlist
```

## 開発

### 開発環境のセットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt

# 開発モードでインストール
pip install -e .

# pre-commitフックのインストール
pip install pre-commit
pre-commit install
```

### コード品質チェック

```bash
# pre-commitフックを手動実行
pre-commit run --all-files

# Ruffによるリンターとフォーマッター
ruff check . --fix
ruff format .

# 型チェック
mypy src/

# セキュリティチェック
bandit -r src/

# テストの実行
pytest
```

### CI/CD

このプロジェクトはGitHub Actionsを使用して以下の自動化を行っています：

- **Pre-commit Checks**: コード品質、フォーマット、型チェック、セキュリティスキャン
- **CI/CD Pipeline**: テスト実行、ビルド、デプロイメント
- **Conflict Detection**: プルリクエストでのマージコンフリクト検出

すべてのプルリクエストは自動的にこれらのチェックが実行され、すべて通過する必要があります。

## ライセンス

MIT License
