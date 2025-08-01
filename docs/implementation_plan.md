# デイトレード支援アプリ CUI版 実装計画

## 実装機能一覧

### Phase 1: 基礎機能（優先度：高）

#### 1. データ取得・管理機能
- [ ] **株価データ取得モジュール** (`src/day_trade/data/stock_fetcher.py`)
  - yfinanceを使用したリアルタイム株価取得
  - 日本株対応（.T suffix処理）
  - ヒストリカルデータ取得（日足、分足）
  - データキャッシュ機能

- [ ] **銘柄マスタ管理** (`src/day_trade/data/stock_master.py`)
  - 東証上場銘柄リストの管理
  - 証券コード・銘柄名での検索
  - セクター・業種分類
  - SQLiteでのローカル保存

- [ ] **データベース基盤** (`src/day_trade/models/database.py`)
  - SQLAlchemyモデル定義
  - マイグレーション管理
  - 接続管理

### Phase 2: 分析機能（優先度：高）

#### 2. テクニカル分析機能
- [ ] **指標計算エンジン** (`src/day_trade/analysis/indicators.py`)
  - 移動平均線（SMA、EMA）
  - ボリンジャーバンド
  - MACD
  - RSI
  - ストキャスティクス
  - 出来高分析

- [ ] **チャートパターン認識** (`src/day_trade/analysis/patterns.py`)
  - ゴールデンクロス・デッドクロス検出
  - サポート・レジスタンスライン
  - ブレイクアウト検知

- [ ] **シグナル生成** (`src/day_trade/analysis/signals.py`)
  - 買いシグナル・売りシグナル
  - 複合条件によるシグナル
  - シグナル強度の評価

### Phase 3: ポートフォリオ管理（優先度：中）

#### 3. 取引・ポートフォリオ機能
- [ ] **取引記録管理** (`src/day_trade/core/trade_manager.py`)
  - 売買履歴の記録
  - 損益計算
  - 手数料・税金考慮
  - CSVエクスポート機能

- [ ] **ポートフォリオ分析** (`src/day_trade/core/portfolio.py`)
  - 保有銘柄管理
  - リアルタイム損益表示
  - パフォーマンス指標計算
  - リスク分析

- [ ] **ウォッチリスト** (`src/day_trade/core/watchlist.py`)
  - お気に入り銘柄登録
  - グループ管理
  - 条件付きアラート設定

### Phase 4: ユーザーインターフェース（優先度：中）

#### 4. CUIインターフェース
- [ ] **メインCLI** (`src/day_trade/cli/main.py`)
  - Clickベースのコマンド体系
  - サブコマンド構成
  - 設定ファイル管理

- [ ] **インタラクティブモード** (`src/day_trade/cli/interactive.py`)
  - Richを使用したTUIダッシュボード
  - リアルタイム更新表示
  - キーボードショートカット

- [ ] **表示フォーマッタ** (`src/day_trade/cli/formatters.py`)
  - テーブル表示
  - チャート表示（ASCII/Unicode）
  - カラーコーディング

### Phase 5: 高度な機能（優先度：低）

#### 5. 自動化・拡張機能
- [ ] **バックテスト機能** (`src/day_trade/analysis/backtest.py`)
  - 売買戦略の検証
  - パフォーマンス評価
  - 最適化機能

- [ ] **アラート機能** (`src/day_trade/core/alerts.py`)
  - 価格アラート
  - テクニカルアラート
  - メール/デスクトップ通知

- [ ] **スクリーニング機能** (`src/day_trade/analysis/screener.py`)
  - 銘柄スクリーニング
  - カスタム条件設定
  - 結果のランキング表示

## コマンド体系案

```bash
# 基本コマンド
daytrade --help                          # ヘルプ表示
daytrade config                          # 設定管理
daytrade interactive                     # インタラクティブモード起動

# 株価データ関連
daytrade stock <code>                    # 個別銘柄情報表示
daytrade stock <code> --chart            # チャート表示
daytrade stock <code> --history 30d      # 過去データ表示

# 分析関連
daytrade analyze <code>                  # テクニカル分析実行
daytrade analyze <code> --indicators all # 全指標表示
daytrade signal scan                     # シグナル検出スキャン

# ポートフォリオ関連
daytrade portfolio show                  # ポートフォリオ表示
daytrade portfolio add <code> <qty>      # 銘柄追加
daytrade trade record <買/売> <code>     # 取引記録

# ウォッチリスト関連
daytrade watchlist show                  # ウォッチリスト表示
daytrade watchlist add <code>            # 銘柄追加
daytrade watchlist remove <code>         # 銘柄削除

# スクリーニング
daytrade screen --rsi-oversold           # RSI売られすぎ銘柄
daytrade screen --breakout               # ブレイクアウト銘柄
daytrade screen --custom "rsi<30 and volume>1000000"
```

## データベーススキーマ概要

```sql
-- 銘柄マスタ
stocks (
  code VARCHAR PRIMARY KEY,
  name VARCHAR,
  sector VARCHAR,
  market VARCHAR
)

-- 価格データ
price_data (
  id INTEGER PRIMARY KEY,
  code VARCHAR,
  datetime TIMESTAMP,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume INTEGER
)

-- 取引履歴
trades (
  id INTEGER PRIMARY KEY,
  code VARCHAR,
  trade_type VARCHAR,  -- 'buy' or 'sell'
  quantity INTEGER,
  price REAL,
  datetime TIMESTAMP,
  commission REAL
)

-- ウォッチリスト
watchlist (
  id INTEGER PRIMARY KEY,
  code VARCHAR,
  group_name VARCHAR,
  added_date DATE
)

-- アラート設定
alerts (
  id INTEGER PRIMARY KEY,
  code VARCHAR,
  condition_type VARCHAR,
  threshold REAL,
  is_active BOOLEAN
)
```

## 実装優先順位

1. **最優先**：株価データ取得とデータベース基盤
2. **高優先**：基本的なテクニカル分析とCLI基本機能
3. **中優先**：ポートフォリオ管理とインタラクティブUI
4. **低優先**：バックテストや高度な分析機能
