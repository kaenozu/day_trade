# 📈 DayTrade 全自動取引システム - ユーザーガイド

## 🚀 クイックスタート（3分で開始）

### 1. システム要件
- Python 3.8以上
- Windows/macOS/Linux対応
- インターネット接続必須（株価データ取得）

### 2. インストール
```bash
# リポジトリのクローン
git clone https://github.com/kaenozu/day_trade.git
cd day_trade

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. すぐに実行
```bash
# 最も簡単な実行方法 - TOP3推奨銘柄を自動表示
python daytrade_simple.py

# または引数なしで高速モード実行
python daytrade_simple.py --quick
```

**これだけで推奨銘柄が3分以内に表示されます！**

---

## 📋 基本的な使い方

### シンプルインターフェース

#### 高速モード（推奨）
```bash
# 引数なしまたは--quickで実行（デフォルト）
python daytrade_simple.py
python daytrade_simple.py --quick

# 結果例：
# 今日の推奨銘柄 TOP3
# 1. 7203 (トヨタ自動車) - [強い買い]
#    推奨度: 95点, 信頼度: 87%
```

#### フルモード（全84銘柄分析）
```bash
# 全銘柄を分析してTOP5推奨
python daytrade_simple.py --full

# 実行時間: 約5-10分
# 結果: より精密な分析結果
```

#### 特定銘柄指定
```bash
# 注目している銘柄のみ分析
python daytrade_simple.py --symbols 7203,8306,9984

# カンマ区切りで複数指定可能
# 任意の東証銘柄コードが使用可能
```

#### 安全モード
```bash
# 高リスク銘柄を除外して分析
python daytrade_simple.py --safe

# 初心者や保守的投資家向け
# 大型安定株中心の推奨
```

---

## 📊 出力の読み方

### 推奨銘柄情報の解説

```
1. 7203 (トヨタ自動車)
   推奨度: 95点        # 0-100点の総合評価
   アクション: [強い買い] # 5段階の推奨レベル
   信頼度: 87%         # 予測の確実性
   リスク: 中          # 低/中/高の3段階
   理由: SMAゴールデンクロス, 出来高急増
   目標価格: 2850円     # 利確目安
   ストップロス: 2650円  # 損切り目安
```

### アクションレベル
- **[強い買い]**: 積極的な投資候補（スコア80点以上）
- **[買い]**: 慎重な投資候補（スコア65点以上）
- **[様子見]**: 判断保留（スコア35-65点）
- **[売り]**: 売却検討（スコア20-35点）
- **[強い売り]**: 即座に売却（スコア20点未満）

### リスクレベル
- **低**: 安定した値動き、初心者向け
- **中**: 標準的なリスク、一般投資家向け
- **高**: 高い変動性、上級者向け

---

## 🛠 高度な使用方法

### コマンドライン引数一覧
```bash
python daytrade_simple.py [OPTIONS]

オプション:
  --quick              高速モード（デフォルト）
  --full               フルモード（全銘柄分析）
  --symbols SYMBOLS    特定銘柄指定（カンマ区切り）
  --safe               安全モード（高リスク除外）
  --version            バージョン情報表示
  --help               ヘルプ表示
```

### 実行例とユースケース

#### 日次ルーティン
```bash
# 毎朝の株式チェック
python daytrade_simple.py --quick

# 週末の詳細分析
python daytrade_simple.py --full
```

#### 投資スタイル別
```bash
# デイトレーダー向け（高速・頻繁実行）
python daytrade_simple.py --quick

# スイングトレーダー向け（詳細分析）
python daytrade_simple.py --full --safe

# 個別株フォロワー向け
python daytrade_simple.py --symbols 7203,6758,4689
```

---

## 🔧 設定とカスタマイズ

### 銘柄リストの編集
設定ファイル：`config/settings.json`
```json
{
  "watchlist": {
    "symbols": [
      {"code": "7203", "name": "トヨタ自動車"},
      {"code": "8306", "name": "三菱UFJ銀行"}
    ]
  }
}
```

### パフォーマンス調整
```bash
# 高速実行のための環境変数
export PYTHONPATH=.
export OMP_NUM_THREADS=4  # CPU利用スレッド数

# GPU使用（NVIDIA GPU環境）
# 自動検出されて高速化されます
```

---

## ❗ 重要な注意事項

### 免責事項
- **このシステムは投資の参考情報を提供するものです**
- **投資判断は必ずご自身の責任で行ってください**
- **過去の実績は将来の結果を保証するものではありません**
- **損失が発生する可能性があります**

### 推奨の使用法
1. **複数指標の確認**: システムの推奨を参考の一つとして使用
2. **リスク管理**: 必ずストップロスを設定
3. **分散投資**: 単一銘柄への集中投資は避ける
4. **市場状況確認**: 全体的な市場トレンドも考慮
5. **継続的学習**: 投資知識を継続して習得

---

## 🚨 トラブルシューティング

### よくある問題と解決法

#### 実行エラー
```bash
# モジュールが見つからない
pip install -r requirements.txt

# 権限エラー
python -m pip install --user -r requirements.txt

# Python環境問題
python --version  # 3.8以上であることを確認
```

#### データ取得エラー
```bash
# ネットワーク接続確認
ping yahoo.com

# プロキシ環境の場合
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080
```

#### パフォーマンス問題
```bash
# メモリ不足の場合
python daytrade_simple.py --symbols 7203,8306  # 銘柄数を減らす

# CPU負荷が高い場合
python daytrade_simple.py --quick  # 高速モードを使用
```

### エラーメッセージ別対処法

| エラーメッセージ | 原因 | 解決法 |
|---|---|---|
| `ModuleNotFoundError` | ライブラリ未インストール | `pip install -r requirements.txt` |
| `Network timeout` | ネットワーク問題 | 接続確認、時間をおいて再実行 |
| `Data quality insufficient` | データ品質低下 | 市場時間外の実行、翌日再実行 |
| `No recommendations found` | 推奨銘柄なし | 市場状況により正常、設定確認 |

---

## 🎯 効果的な活用方法

### 投資初心者向け
1. **安全モードから開始**: `--safe`オプションを使用
2. **小額から実践**: 推奨銘柄でも小額投資から
3. **学習と併用**: 推奨理由を理解し投資知識を習得
4. **記録の蓄積**: 実行結果と実際の株価動向を比較

### 経験者向け
1. **詳細分析活用**: `--full`モードで全銘柄チェック
2. **特定銘柄深掘り**: `--symbols`で注目銘柄を集中分析
3. **他の分析と併用**: ファンダメンタル分析等と組み合わせ
4. **パラメータ調整**: 設定ファイルのカスタマイズ

### プロ・デイトレーダー向け
1. **自動化スクリプト**: cronやTask Schedulerで定期実行
2. **API統合**: 結果を取引システムに連携
3. **バックテスト**: 過去データでの戦略検証
4. **リアルタイム監視**: 高頻度での実行と監視

---

## 🔄 アップデートとメンテナンス

### 定期的な更新
```bash
# システムの更新
git pull origin main
pip install -r requirements.txt --upgrade

# 設定の確認
python daytrade_simple.py --version
```

### ログとキャッシュの管理
```bash
# ログファイルの確認
ls -la *.log

# キャッシュのクリア（必要に応じて）
rm -rf __pycache__ src/**/__pycache__
rm -f data/market_cache.db
```

---

## 📞 サポートとコミュニティ

### ヘルプとサポート
- **GitHubイシュー**: [Issues](https://github.com/kaenozu/day_trade/issues)
- **ドキュメント**: 最新の使用方法とFAQ
- **コミュニティ**: ユーザー同士の情報交換

### 機能要望とバグレポート
新機能の要望やバグ報告は、GitHubのIssuesページからお願いします。

---

**Happy Trading! 📈**

*最終更新: 2025年8月*