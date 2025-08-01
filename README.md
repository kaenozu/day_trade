
# Day Trade Analysis System

## 概要
このプロジェクトは、様々なテクニカル指標と市場センチメント分析を統合し、日中の取引シグナルを生成するための分析エンジンを提供します。CLIインターフェースを通じて、ユーザーは特定のデータセットと戦略に基づいて市場分析を実行し、最終的な取引推奨を得ることができます。

## 特徴
- **複数の分析戦略**: 移動平均線クロス、RSI、VWAP、ボリンジャーバンド、ATR、出来高分析、市場センチメント分析など、多様なテクニカル指標をサポート。
- **シグナル統合**: 複数の戦略からのシグナルを統合し、重み付けされたスコアに基づいて最終的な売買推奨を決定。
- **CLIインターフェース**: コマンドラインからの簡単な操作で分析を実行。
- **拡張性**: 新しい分析戦略を容易に追加できるモジュール設計。

## セットアップ

### 1. リポジトリのクローン
```bash
git clone https://github.com/kaenozu/day_trade.git
cd day_trade
```

### 2. 仮想環境の作成とアクティベート
Python 3.8以上がインストールされていることを確認してください。
```bash
python -m venv venv
# Windowsの場合
.\venv\Scripts\activate
# macOS/Linuxの場合
source venv/bin/activate
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```
`requirements.txt` が存在しない場合は、以下のコマンドで作成できます。
```bash
pip freeze > requirements.txt
```

## 使用方法

### データ準備
分析にはCSV形式の株価データが必要です。以下のカラムが含まれていることを確認してください。
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Sentiment` (市場センチメント分析を使用する場合、-1:悲観, 0:中立, 1:楽観)

例: `dummy_data.csv`

### 分析の実行
`main.py` を使用して分析を実行します。

```bash
python main.py run_analysis --data_path <データファイルへのパス> --strategies_file <戦略ファイルへのパス>
```

- `<データファイルへのパス>`: 分析に使用するCSVデータファイルへのパスを指定します。
- `<戦略ファイルへのパス>`: 実行する戦略のリストをJSON形式で記述したファイルへのパスを指定します。

**例: `strategies.json` の内容**
```json
["MA Cross", "RSI", "Sentiment Analysis"]
```

**実行例:**
```bash
python main.py run_analysis --data_path dummy_data.csv --strategies_file strategies.json
```

## 貢献
貢献を歓迎します！バグ報告、機能リクエスト、プルリクエストなど、お気軽にお寄せください。

### 開発フロー
1. イシューを作成または選択します。
2. `gh issue develop <issue_number>` を使用して開発ブランチを作成します。
3. 変更を実装し、テストします。
4. 変更をコミットし、リモートブランチにプッシュします。
5. `gh pr create --fill` を使用してプルリクエストを作成します。

## ライセンス
[ライセンス情報があればここに記述]