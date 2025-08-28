# 🚤 Boatrace競艇予想システム

BoatraceOpenAPIを活用した競艇予想・舟券購入支援システムです。

## ✨ 機能概要

### 🎯 予想・分析機能
- **BoatraceOpenAPI統合** - 公式データを活用したリアルタイム情報取得
- **競技場特性分析** - 全国24競技場の水面条件・特徴を考慮した分析
- **選手成績分析** - 勝率、クラス、競技場適性、調子を総合的に評価
- **AI予想エンジン** - 統計分析に基づく確率計算と買い目推奨
- **レース競争力分析** - 激戦度・大荒れ確率の判定

### 💰 投票・収支管理
- **多様な投票戦略** - 保守的・バランス・積極的・AI戦略
- **舟券管理システム** - 購入記録・結果確認・配当計算
- **ポートフォリオ管理** - 投資収支・ROI・リスク分析
- **資金管理** - Kelly基準ベースの最適賭け金計算

### 📊 データ管理
- **自動データ収集** - APIから出走表・結果の自動取得・保存
- **SQLiteデータベース** - 競技場・選手・レース・予想データの永続化
- **パフォーマンス追跡** - 戦略別・期間別の成績分析

## 🏗️ システム構成

```
src/boatrace/
├── core/           # コアシステム
│   ├── api_client.py       # BoatraceOpenAPIクライアント
│   ├── data_models.py      # データモデル定義
│   ├── stadium_manager.py  # 競技場管理
│   └── race_manager.py     # レース管理
├── data/           # データ管理
│   ├── database.py         # SQLAlchemyデータベース
│   ├── data_collector.py   # データ収集
│   └── data_processor.py   # データ前処理・分析
├── prediction/     # 予想エンジン
│   ├── prediction_engine.py  # 予想エンジン本体
│   ├── racer_analyzer.py     # 選手分析
│   └── race_analyzer.py      # レース分析
├── betting/        # 投票管理
│   ├── ticket_manager.py     # 舟券管理
│   ├── betting_strategy.py   # 投票戦略
│   └── portfolio.py          # ポートフォリオ管理
└── web/           # Webインターフェース（今後実装）
```

## 🚀 クイックスタート

### 1. システム実行
```bash
# メインデモの実行
python main_boatrace.py

# テスト実行
python tests/test_boatrace_integration.py
```

### 2. 基本的な使用方法

```python
from src.boatrace.core.api_client import BoatraceAPIClient
from src.boatrace.core.race_manager import RaceManager
from src.boatrace.prediction.prediction_engine import PredictionEngine
from src.boatrace.data.database import init_database

# システム初期化
database = init_database()
api_client = BoatraceAPIClient()
race_manager = RaceManager(api_client)
prediction_engine = PredictionEngine()

# 今日のレース取得
today_races = race_manager.get_today_races()
print(f"今日のレース数: {len(today_races)}")

# 予想実行
if today_races:
    race = today_races[0]
    prediction = prediction_engine.predict_race(race.id)
    
    if prediction:
        print(f"予想信頼度: {float(prediction.confidence):.2f}")
        print("推奨買い目:")
        for bet in prediction.recommended_bets:
            print(f"  {bet['bet_type']}: {bet['numbers']}")
```

### 3. 投票戦略の活用

```python
from src.boatrace.betting.betting_strategy import StrategyManager
from src.boatrace.betting.ticket_manager import TicketManager
from decimal import Decimal

# 戦略管理
strategy_manager = StrategyManager()
ticket_manager = TicketManager()

# 戦略選択
strategy = strategy_manager.get_strategy("balanced")
budget = Decimal('5000')

# 買い目推奨生成
recommendations = strategy.generate_bets(prediction, budget, race_info)

# 舟券購入記録
for rec in recommendations:
    ticket_id = ticket_manager.purchase_ticket(
        race_id=race.id,
        bet_type=rec.bet_type,
        numbers=rec.numbers,
        amount=rec.amount
    )
```

## 📈 主要機能詳細

### BoatraceOpenAPI連携
- **エンドポイント**: プログラム・直前情報・結果API
- **キャッシュ機能**: API呼び出し最適化
- **リトライ機構**: 接続エラー時の自動再試行

### 競技場特性データベース
```python
# 競技場分析例
stadium_manager = StadiumManager()
analysis = stadium_manager.get_stadium_analysis(3)  # 江戸川

print(f"水面タイプ: {analysis['characteristics'].water_type}")
print(f"潮汐影響: {analysis['characteristics'].is_tidal}")
print(f"インコース有利度: {analysis['characteristics'].inside_advantage}")
```

### 予想アルゴリズム
1. **選手成績評価** - 勝率・クラス・適性を数値化
2. **競技場補正** - 水面特性による有利不利を考慮
3. **確率計算** - 統計分析による各艇の勝率算出
4. **買い目推奨** - 期待値とリスクを考慮した投票戦略

### 投票戦略
- **保守的戦略**: 本命中心の手堅い投票
- **バランス戦略**: 本命と穴のバランス型
- **積極戦略**: 高配当狙いの多点勝負
- **AI戦略**: 予想確率に基づく最適配分

## 📊 データベース設計

### 主要テーブル
- `stadiums` - 競技場マスタ（特性情報含む）
- `racers` - 選手マスタ（成績・クラス情報）
- `races` - レースマスタ（日程・条件）
- `race_entries` - 出走情報（選手×レース）
- `predictions` - 予想結果（確率・推奨買い目）
- `betting_tickets` - 舟券購入履歴
- `race_results` - レース結果・配当

## 🛡️ リスク管理機能

### ポートフォリオ分析
```python
portfolio = Portfolio(initial_capital=Decimal('100000'))

# 収支サマリー
balance = portfolio.get_current_balance()
print(f"ROI: {balance['roi']:.2f}%")

# リスク分析
risk_analysis = portfolio.get_risk_analysis()
print(f"リスクレベル: {risk_analysis['overall_risk_level']}")

# 最適賭け金計算
optimal_bet = portfolio.optimize_bet_sizing(
    prediction_confidence=0.75,
    expected_odds=3.0
)
```

## 🎯 特徴・優位性

1. **公式データ活用** - BoatraceOpenAPIから最新の正確なデータを取得
2. **競技場特性重視** - 全24競技場の水面条件を詳細に考慮
3. **統計分析ベース** - 客観的データに基づく予想生成
4. **戦略的投票** - 多様な投票戦略と資金管理機能
5. **包括的分析** - 選手・レース・収支を統合的に管理

## 📝 今後の拡張予定

### Phase 2 (進行中)
- 舟券購入・管理システムの完成
- Webダッシュボードの実装

### Phase 3 (予定)
- 機械学習モデルの追加
- リアルタイム予想更新
- 自動投票システム（TELEBOAT連携）

### Phase 4 (構想)
- モバイルアプリ対応
- クラウドデプロイメント
- API提供

## ⚠️ 注意事項

- **投資は自己責任**: 本システムは予想支援ツールであり、投資結果を保証するものではありません
- **適度な投資**: ギャンブル依存に注意し、余剰資金での投資を推奨します
- **法的制限**: 未成年者の利用や法的制限のある地域での利用は禁止です

## 🤝 コントリビューション

プルリクエスト・イシュー報告を歓迎します。

## 📄 ライセンス

MIT License

---

**🚤 競艇予想の新しいスタンダードを目指して**