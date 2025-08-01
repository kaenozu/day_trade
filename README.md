# Day Trade - デイトレード支援アプリ

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

```bash
# テストの実行
pytest

# コードフォーマット
black src/

# 型チェック
mypy src/
```

## ライセンス

MIT License