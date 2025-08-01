# アンサンブル戦略

## 概要

アンサンブル戦略は、複数の取引戦略を組み合わせて、より信頼性の高いシグナルを生成する高度な手法です。機械学習のアンサンブル手法にインスパイアされ、異なる特徴を持つ戦略を組み合わせることで、単一戦略の弱点を補完し、全体的なパフォーマンスを向上させます。

## 主な特徴

### 1. 多様な戦略の組み合わせ
- **保守的RSI戦略**: 極値でのみ反応する慎重なアプローチ
- **積極的モメンタム戦略**: ブレイクアウトと出来高急増を重視
- **トレンドフォロー戦略**: ゴールデンクロス・デッドクロスに基づく
- **平均回帰戦略**: ボリンジャーバンドとRSIの組み合わせ
- **デフォルト統合戦略**: 従来の全ルールを統合

### 2. 高度な投票システム
- **ソフト投票**: 信頼度による重み付け投票（デフォルト）
- **ハード投票**: 多数決による単純投票
- **重み付け平均**: パフォーマンスベースの動的重み付け

### 3. 適応的学習機能
- **パフォーマンス記録**: 各戦略の成功率とシャープレシオを追跡
- **動的重み調整**: 過去のパフォーマンスに基づく自動重み調整
- **メタ特徴量**: 市場状況を考慮した文脈的判断

## アーキテクチャ

```python
EnsembleTradingStrategy
├── 個別戦略群
│   ├── conservative_rsi (保守的RSI戦略)
│   ├── aggressive_momentum (積極的モメンタム戦略)
│   ├── trend_following (トレンドフォロー戦略)
│   ├── mean_reversion (平均回帰戦略)
│   └── default_integrated (デフォルト統合戦略)
├── 投票システム
│   ├── ソフト投票 (信頼度重み付け)
│   ├── ハード投票 (多数決)
│   └── 重み付け平均
├── パフォーマンス管理
│   ├── 戦略別成功率追跡
│   ├── 動的重み調整
│   └── パフォーマンス履歴保存
└── メタ学習
    ├── 市場状況分析
    ├── ボラティリティ計算
    └── 文脈的特徴量抽出
```

## 戦略タイプ

### 1. CONSERVATIVE (保守的)
- **特徴**: 高い合意を重視、偽シグナルを最小化
- **重み配分**: 保守的戦略に高い重み
- **信頼度閾値**: 60%
- **適用場面**: 安定した収益を重視する場合

### 2. AGGRESSIVE (積極的)
- **特徴**: 取引機会を最大化、高いリターンを追求
- **重み配分**: モメンタム戦略に高い重み
- **信頼度閾値**: 30%
- **適用場面**: 高いリスクを許容し、大きな利益を狙う場合

### 3. BALANCED (バランス型)
- **特徴**: リスクとリターンのバランスを重視
- **重み配分**: 全戦略に均等な重み
- **信頼度閾値**: 45%
- **適用場面**: 一般的な取引に最適（デフォルト）

### 4. ADAPTIVE (適応型)
- **特徴**: パフォーマンスに基づく動的調整
- **重み配分**: 成功率に基づく自動調整
- **信頼度閾値**: 動的調整（30-70%）
- **適用場面**: 長期運用で最適化を重視する場合

## 設定例

### config/settings.json
```json
{
  "analysis": {
    "ensemble": {
      "enabled": true,
      "strategy_type": "balanced",
      "voting_type": "soft",
      "performance_file_path": "data/ensemble_performance.json",
      "strategy_weights": {
        "conservative_rsi": 0.2,
        "aggressive_momentum": 0.25,
        "trend_following": 0.25,
        "mean_reversion": 0.2,
        "default_integrated": 0.1
      },
      "confidence_thresholds": {
        "conservative": 60.0,
        "aggressive": 30.0,
        "balanced": 45.0,
        "adaptive": 40.0
      },
      "meta_learning_enabled": true,
      "adaptive_weights_enabled": true
    }
  }
}
```

## 使用方法

### 基本的な使用例

```python
from src.day_trade.analysis.ensemble import (
    EnsembleTradingStrategy,
    EnsembleStrategy,
    EnsembleVotingType
)

# アンサンブル戦略の初期化
ensemble = EnsembleTradingStrategy(
    ensemble_strategy=EnsembleStrategy.BALANCED,
    voting_type=EnsembleVotingType.SOFT_VOTING,
    performance_file="data/ensemble_performance.json"
)

# シグナル生成
ensemble_signal = ensemble.generate_ensemble_signal(
    df=price_data,
    indicators=technical_indicators,
    patterns=chart_patterns
)

if ensemble_signal:
    signal = ensemble_signal.ensemble_signal
    print(f"シグナル: {signal.signal_type.value}")
    print(f"信頼度: {signal.confidence:.1f}%")
    print(f"強度: {signal.strength.value}")
```

### オーケストレーターでの使用

```python
from src.day_trade.automation.orchestrator import DayTradeOrchestrator

# 設定ファイルでアンサンブルを有効化済みの場合
orchestrator = DayTradeOrchestrator("config/settings.json")

# 全自動化実行（アンサンブル戦略が自動適用される）
report = orchestrator.run_full_automation()
```

## メタ特徴量

アンサンブル戦略は以下のメタ特徴量を考慮してシグナルの質を向上させます：

### 市場状況指標
- **ボラティリティ**: 年率ボラティリティで市場の変動性を測定
- **トレンド強度**: SMA20/SMA50の比率でトレンドの強さを判定
- **価格位置**: 過去20日のレンジ内での現在価格の位置

### テクニカル状況
- **RSIレベル**: 現在のRSI値で過買い/過売り状況を判定
- **MACD乖離**: MACDとシグナルラインの乖離でモメンタムを測定

### 出来高情報
- **出来高比率**: 過去10日平均との比較で市場参加度を測定

## パフォーマンス管理

### 戦略評価指標
- **成功率**: シグナル後の価格動向による成功/失敗の判定
- **平均信頼度**: 各戦略の平均的な信頼度スコア
- **シャープレシオ**: リスク調整後リターンによる戦略評価
- **平均リターン**: 戦略による平均的な収益率

### 動的重み調整
適応型戦略では以下の複合スコアで重みを動的調整：
```
スコア = 成功率 × 0.4 + シャープレシオ × 0.3 + 平均リターン × 0.2 + 最新性 × 0.1
```

## 投票メカニズム

### ソフト投票（推奨）
1. 各戦略の信頼度に重みを掛け合わせ
2. シグナルタイプ別に重み付きスコアを集計
3. 最高スコアのシグナルタイプを選択
4. 閾値未満の場合はHOLDに変更

### ハード投票
1. 各戦略から1票ずつ獲得
2. 成功率30%未満の戦略は投票権剥奪
3. 最多得票のシグナルタイプを選択
4. 過半数未満の場合はHOLD

## テスト

```bash
# アンサンブル戦略のテスト実行
pytest tests/test_ensemble.py -v

# 設定管理のテスト実行
pytest tests/test_config_ensemble.py -v

# 全テスト実行
pytest tests/ -k ensemble -v
```

## ログ

アンサンブル戦略は以下のログを出力します：

```
INFO - アンサンブル戦略を有効化: balanced, 投票方式: soft
DEBUG - アンサンブルシグナル生成 (7203): BUY, 信頼度: 0.67
DEBUG - 適応型重み更新: {'conservative_rsi': 0.22, 'aggressive_momentum': 0.28, ...}
```

## パフォーマンス最適化

### 推奨設定
- **CPU使用量重視**: `meta_learning_enabled: false`
- **精度重視**: `voting_type: "soft"`, `strategy_type: "adaptive"`
- **速度重視**: `voting_type: "hard"`, `strategy_type: "balanced"`

### メモリ使用量
- パフォーマンス履歴: 約1MB（1年間の記録）
- 戦略インスタンス: 約5MB（5戦略）
- メタ特徴量: 約1KB（1銘柄あたり）

## トラブルシューティング

### よくある問題

1. **シグナルが生成されない**
   - 信頼度閾値が高すぎる可能性 → `confidence_thresholds`を下げる
   - データ不足の可能性 → 最小50日分のデータを確保

2. **重みの合計が1.0にならない**
   - 設定ファイルの重み設定を確認
   - 自動正規化が機能している場合は問題なし

3. **パフォーマンス履歴が保存されない**
   - `performance_file_path`のディレクトリが存在するか確認
   - 書き込み権限を確認

## 今後の拡張予定

- **強化学習との統合**: Q-learningによる動的戦略選択
- **外部データ連携**: ニュースセンチメント分析の組み込み
- **リアルタイム最適化**: 市場状況に応じたリアルタイム重み調整
- **カスタム戦略**: ユーザー定義戦略の追加サポート
