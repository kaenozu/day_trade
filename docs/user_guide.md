# Day Trade ユーザーガイド

## 目次

1. [はじめに](#はじめに)
2. [基本的な使い方](#基本的な使い方)
3. [主要機能詳細](#主要機能詳細)
4. [実践的な使用例](#実践的な使用例)
5. [設定とカスタマイズ](#設定とカスタマイズ)
6. [トラブルシューティング](#トラブルシューティング)

## はじめに

Day Tradeは、高度なアンサンブル戦略を用いた自動デイトレード支援システムです。複数のテクニカル分析手法を組み合わせて、効率的な株式取引の意思決定をサポートします。

### 対象ユーザー
- 個人投資家
- デイトレーダー
- システムトレーダー
- 投資研究者

### 必要な知識
- 基本的な株式投資の知識
- テクニカル分析の基礎理解
- コマンドライン操作の基本

## 基本的な使い方

### インストール後の初期設定

```bash
# データベースの初期化
python -m day_trade.models.database --init

# 設定ファイルの確認
cat config/settings.json

# 動作確認
python -m day_trade.cli.main --help
```

### アプリケーションの起動

#### インタラクティブモード（推奨）
```bash
# 基本起動
python -m day_trade.cli.main

# または
daytrade

# ヘルプ表示
daytrade --help
```

#### コマンドラインモード
```bash
# 特定銘柄の分析
daytrade analyze 7203

# ウォッチリスト表示
daytrade watchlist show

# ポートフォリオ分析
daytrade portfolio analyze
```

### 初回使用の流れ

1. **銘柄の追加**
   ```bash
   # 主要銘柄をウォッチリストに追加
   daytrade watchlist add 7203 6758 4755 8411
   ```

2. **ポートフォリオの設定**
   ```bash
   # 保有銘柄の登録
   daytrade portfolio add 7203 --quantity 100 --price 2500
   ```

3. **分析の実行**
   ```bash
   # シグナル生成
   daytrade analyze 7203
   ```

## 主要機能詳細

### 1. 銘柄分析

#### リアルタイム価格取得
```bash
# 現在価格の取得
daytrade price 7203

# 複数銘柄の一括取得
daytrade price 7203 6758 4755
```

#### テクニカル分析
```bash
# 基本的な分析
daytrade analyze 7203

# 詳細分析（複数指標）
daytrade analyze 7203 --indicators RSI,MACD,BB --period 30

# チャート表示
daytrade chart 7203 --period 30d --interval 1h
```

#### 分析結果の例
```
トヨタ自動車 (7203.T)
現在価格: ¥2,724
前日比: +24 (+0.89%)

テクニカル指標:
  RSI (14): 58.3 [中立]
  MACD: 上向きクロス [買いシグナル]
  ボリンジャーバンド: 中央線付近 [中立]

アンサンブル判定: 弱い買い (信頼度: 72%)
```

### 2. ウォッチリスト管理

#### 基本操作
```bash
# 銘柄追加
daytrade watchlist add 7203 6758 4755

# 銘柄削除
daytrade watchlist remove 7203

# 一覧表示
daytrade watchlist show

# 詳細表示（価格・シグナル付き）
daytrade watchlist show --details
```

#### 自動監視
```bash
# アラート設定
daytrade watchlist alert 7203 --price-above 2800 --price-below 2600

# 定期更新（5分間隔）
daytrade watchlist monitor --interval 5m
```

### 3. ポートフォリオ管理

#### 保有銘柄の管理
```bash
# 銘柄追加
daytrade portfolio add 7203 --quantity 100 --price 2500

# 銘柄売却
daytrade portfolio sell 7203 --quantity 50 --price 2750

# 現在のポートフォリオ表示
daytrade portfolio show

# パフォーマンス分析
daytrade portfolio analyze --period 30d
```

#### リバランシング
```bash
# 目標配分の設定
daytrade portfolio rebalance --target-weights 7203:40,6758:30,4755:30

# リバランシング実行
daytrade portfolio rebalance --execute
```

### 4. バックテスト

#### 基本的なバックテスト
```bash
# 単一銘柄のバックテスト
daytrade backtest 7203 --start-date 2024-01-01 --end-date 2024-12-31

# 複数戦略の比較
daytrade backtest 7203 --strategies momentum,mean_reversion,volatility
```

#### 詳細設定
```bash
# 初期資金・手数料設定
daytrade backtest 7203 \
  --initial-capital 1000000 \
  --commission-rate 0.001 \
  --slippage 0.0005
```

#### 結果の出力
```bash
# HTML形式でレポート生成
daytrade backtest 7203 --output-format html --output-file backtest_report.html

# JSON形式でエクスポート
daytrade backtest 7203 --output-format json --output-file results.json
```

### 5. スクリーニング

#### 条件指定による銘柄抽出
```bash
# 高成長銘柄のスクリーニング
daytrade screen --strategy momentum --min-volume 1000000 --min-price 500

# 割安銘柄のスクリーニング
daytrade screen --strategy value --max-per 15 --min-dividend-yield 3.0

# テクニカル条件
daytrade screen --rsi-range 30,70 --macd-bullish --above-sma200
```

#### 結果の表示・保存
```bash
# 画面に表示
daytrade screen --strategy momentum --limit 20

# CSVファイルに保存
daytrade screen --strategy momentum --output screening_results.csv
```

### 6. アラート機能

#### 価格アラート
```bash
# 単一銘柄の価格アラート
daytrade alert price 7203 --above 2800 --below 2600

# 複数銘柄の一括設定
daytrade alert price-batch --file price_alerts.csv
```

#### シグナルアラート
```bash
# 買いシグナル発生時
daytrade alert signal 7203 --signal-type buy --confidence 70

# 強いシグナル発生時
daytrade alert signal 7203 --signal-strength strong
```

#### アラート履歴
```bash
# 履歴表示
daytrade alert history --last 7d

# アラート設定一覧
daytrade alert list
```

## 実践的な使用例

### 例1: 日々のトレーディングルーチン

```bash
# 1. 朝の準備（9:00頃）
# ウォッチリストの状況確認
daytrade watchlist show --details

# ポートフォリオの状況確認
daytrade portfolio show

# 2. 寄り付き後の分析（9:30頃）
# 主要銘柄の分析
daytrade analyze 7203 6758 4755

# 新しい投資機会のスクリーニング
daytrade screen --strategy momentum --limit 10

# 3. 昼休み中の確認（12:00頃）
# アラート履歴チェック
daytrade alert history --last 4h

# ポートフォリオパフォーマンス確認
daytrade portfolio analyze --period 1d

# 4. 引け前の最終確認（14:30頃）
# ウォッチリストの最終確認
daytrade watchlist monitor --once

# 必要に応じてポジション調整
daytrade portfolio rebalance --dry-run
```

### 例2: 週末の戦略見直し

```bash
# 1. 週次パフォーマンス分析
daytrade portfolio analyze --period 7d --detailed

# 2. バックテスト実行
daytrade backtest portfolio --period 1m --strategies all

# 3. 新しい銘柄の発掘
daytrade screen --strategy growth --min-volume 5000000 --limit 50

# 4. ウォッチリストの整理
daytrade watchlist cleanup --remove-inactive --days 30

# 5. レポート生成
daytrade report --type weekly --format html --output weekly_report.html
```

### 例3: 月次の包括的レビュー

```bash
# 1. 月次パフォーマンス詳細分析
daytrade portfolio analyze --period 30d --include-benchmark

# 2. 戦略別パフォーマンス比較
daytrade backtest portfolio --period 3m --strategy-comparison

# 3. リスク分析
daytrade portfolio risk-analysis --var-confidence 0.95

# 4. セクター分析
daytrade portfolio sector-analysis --rebalance-suggestions

# 5. 月次レポート生成
daytrade report --type monthly --format pdf --output monthly_report.pdf
```

## 設定とカスタマイズ

### 設定ファイルの編集

#### メイン設定ファイル（config/settings.json）
```json
{
  "database": {
    "url": "sqlite:///day_trade.db",
    "echo": false
  },
  "data": {
    "cache_ttl": 300,
    "max_cache_size": 1000
  },
  "analysis": {
    "default_period": 20,
    "confidence_threshold": 70
  },
  "alerts": {
    "email_enabled": false,
    "sound_enabled": true
  }
}
```

#### テクニカル指標設定（config/indicators_config.json）
```json
{
  "RSI": {
    "period": 14,
    "overbought": 70,
    "oversold": 30
  },
  "MACD": {
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9
  },
  "BollingerBands": {
    "period": 20,
    "std_dev": 2
  }
}
```

#### シグナル生成ルール（config/signal_rules.json）
```json
{
  "buy_conditions": [
    {"indicator": "RSI", "condition": "below", "value": 30},
    {"indicator": "MACD", "condition": "cross_above", "target": "signal"}
  ],
  "sell_conditions": [
    {"indicator": "RSI", "condition": "above", "value": 70},
    {"indicator": "MACD", "condition": "cross_below", "target": "signal"}
  ],
  "weights": {
    "momentum": 0.4,
    "mean_reversion": 0.3,
    "volatility": 0.2,
    "volume": 0.1
  }
}
```

### 環境変数

```bash
# データベース設定
export DATABASE_URL="sqlite:///day_trade.db"

# APIキー（必要に応じて）
export ALPHA_VANTAGE_API_KEY="your_api_key"

# ログレベル
export LOG_LEVEL="INFO"

# キャッシュ設定
export CACHE_TTL="300"
export MAX_CACHE_SIZE="1000"

# アラート設定
export ALERT_EMAIL_ENABLED="false"
export ALERT_SOUND_ENABLED="true"
```

### カスタム戦略の作成

#### 簡単なカスタム戦略例
```python
# config/custom_strategies.py
def custom_momentum_strategy(data, **kwargs):
    """カスタムモメンタム戦略"""
    rsi = calculate_rsi(data['Close'], period=14)
    macd_line, signal_line = calculate_macd(data['Close'])

    # 買いシグナル条件
    if rsi[-1] > 50 and macd_line[-1] > signal_line[-1]:
        return {"signal": "BUY", "confidence": 75}

    # 売りシグナル条件
    elif rsi[-1] < 50 and macd_line[-1] < signal_line[-1]:
        return {"signal": "SELL", "confidence": 75}

    # 中立
    else:
        return {"signal": "HOLD", "confidence": 50}
```

#### 戦略の登録・使用
```bash
# カスタム戦略で分析
daytrade analyze 7203 --custom-strategy custom_momentum_strategy

# バックテストでの使用
daytrade backtest 7203 --custom-strategy custom_momentum_strategy
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. データ取得エラー
```bash
# 症状: "データを取得できません"エラー
# 原因: ネットワーク接続問題、API制限

# 対処方法:
# ネットワーク接続確認
ping finance.yahoo.com

# キャッシュクリア
daytrade cache clear

# 手動でのデータ取得テスト
daytrade test-connection

# API制限の確認
daytrade api-status
```

#### 2. データベースエラー
```bash
# 症状: "データベース接続エラー"
# 原因: データベースファイルの破損、権限問題

# 対処方法:
# データベースの再初期化
python -m day_trade.models.database --reset

# バックアップからの復元
daytrade restore-database --backup-file backup.db

# 権限確認（Linux/macOS）
ls -la day_trade.db
chmod 644 day_trade.db
```

#### 3. パフォーマンス問題
```bash
# 症状: 分析処理が遅い
# 原因: 大量データ、メモリ不足

# 対処方法:
# キャッシュサイズの調整
export MAX_CACHE_SIZE="2000"

# 分析期間の短縮
daytrade analyze 7203 --period 10  # デフォルト20から10に

# 並列処理の有効化
daytrade analyze 7203 --parallel
```

#### 4. アラートが届かない
```bash
# 症状: 設定したアラートが発火しない
# 原因: 設定ミス、条件未達

# 対処方法:
# アラート設定の確認
daytrade alert list --verbose

# アラート履歴の確認
daytrade alert history --all

# テストアラートの送信
daytrade alert test --type price

# ログファイルの確認
tail -f daytrade_$(date +%Y%m%d).log
```

### ログファイルの確認

```bash
# 日次ログファイル
tail -f daytrade_$(date +%Y%m%d).log

# エラーログのみ抽出
grep ERROR daytrade_$(date +%Y%m%d).log

# 特定の銘柄のログ抽出
grep "7203" daytrade_$(date +%Y%m%d).log

# 詳細ログの有効化
export LOG_LEVEL="DEBUG"
daytrade analyze 7203
```

### デバッグモード

```bash
# デバッグ情報付きで実行
daytrade --debug analyze 7203

# より詳細な情報
daytrade --verbose --debug analyze 7203

# テストモード（実際の取引は行わない）
daytrade --test-mode portfolio rebalance --execute
```

### サポートとコミュニティ

#### 問題報告
- **GitHub Issues**: バグ報告・機能要求
- **GitHub Discussions**: 質問・議論

#### 情報収集
- **README.md**: 基本情報
- **docs/**: 詳細ドキュメント
- **ログファイル**: トラブル時の情報

#### 支援の求め方
1. 問題の詳細説明
2. エラーメッセージの全文
3. 環境情報（OS、Pythonバージョン）
4. 再現手順
5. ログファイル（該当部分）

---

このガイドを参考に、Day Tradeシステムを効果的に活用してください。投資判断は常に自己責任で行い、リスク管理を徹底してください。
