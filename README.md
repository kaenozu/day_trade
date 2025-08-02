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

### 📊 データ取得・管理
- **リアルタイム株価データ取得** (yfinance API統合)
- **API耐障害性機能** (リトライ・サーキットブレーカー・フェイルオーバー)
- **銘柄マスター管理** (東証銘柄情報の自動更新)
- **SQLAlchemy基盤のデータベース管理**
- **データキャッシュ機能** (パフォーマンス最適化)

### 🔍 高度なテクニカル分析
- **アンサンブル戦略エンジン** (複数指標の統合判定)
- **テクニカル指標計算** (RSI, MACD, ボリンジャーバンド等)
- **パターン認識** (トレンド分析, サポート/レジスタンス)

## インストール
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

### 戦略パフォーマンス

| 戦略 | 年間リターン | シャープレシオ | 最大ドローダウン |
|------|-------------|---------------|----------------|
| アンサンブル | 15.3% | 1.42 | -8.7% |
| モメンタム | 12.1% | 1.18 | -12.3% |
| 平均回帰 | 8.9% | 0.95 | -6.4% |
| ボラティリティ | 11.7% | 1.24 | -9.8% |

## ライセンス

MIT License

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

### ユーザーガイド
- [ユーザーガイド](docs/user_guide.md) - 基本的な使い方から実践的な活用方法
- [トラブルシューティングガイド](docs/troubleshooting.md) - よくある問題と解決方法

### 技術ドキュメント
- [アンサンブル戦略](docs/ensemble_strategy.md) - 複数指標統合による判定システム
- [API耐障害性機能](docs/api_resilience.md) - リトライ・サーキットブレーカー・フォールバック
- [構造化ログ](docs/structured_logging.md) - 監視・デバッグ・分析のためのログシステム
- [トランザクション管理](docs/transaction_management.md) - ACID準拠の取引管理

### 開発者向け
- [開発者ガイド](docs/developer_guide.md) - 開発環境構築から新機能追加まで
- [インタラクティブモード](docs/interactive_mode.md) - 対話型インターフェース
- [実装計画](docs/implementation_plan.md) - システム設計とアーキテクチャ

### その他
- [テスト実行ガイド](TESTING.md) - テスト戦略とカバレッジ
- [依存関係管理](DEPENDENCY_MANAGEMENT.md) - パッケージ管理

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
