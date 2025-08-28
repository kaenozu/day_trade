# Boatrace Open API Complete Rebuild Plan

## プロジェクト概要
全システムをBoatraceOpenAPI（競艇Open API）に特化した予想・舟券購入支援システムに全面的に作り直します。

## API仕様調査結果

### 利用可能なAPI
1. **Programs API (出走表)**: `https://boatraceopenapi.github.io/programs/v2/{日付}.json`
2. **Previews API (直前情報)**: `https://boatraceopenapi.github.io/previews/v2/{日付}.json`
3. **Results API (結果)**: `https://boatraceopenapi.github.io/results/v2/{日付}.json`

### データ構造（Programs API）
```json
{
  "programs": [
    {
      "race_date": "2025-08-21",
      "race_stadium_number": 3,
      "race_number": 1,
      "race_closed_at": "08:55:00",
      "race_grade_number": 1,
      "race_title": "レースタイトル",
      "race_subtitle": "レースサブタイトル", 
      "race_distance": 1800,
      "boats": [
        {
          "racer_boat_number": 1,
          "racer_name": "選手名",
          "racer_number": 4123,
          "racer_class_number": 1,
          "racer_age": 28,
          "racer_weight": 52.0,
          "racer_local_winning_rate": 65.22,
          "racer_local_quinella_rate": 78.26,
          "racer_national_winning_rate": 58.33,
          "racer_national_quinella_rate": 75.00,
          // その他多数の選手統計データ
        }
      ]
    }
  ]
}
```

## 新システム設計

### アーキテクチャ
```
boatrace_system/
├── src/
│   ├── boatrace/
│   │   ├── core/
│   │   │   ├── api_client.py        # BoatraceOpenAPIクライアント
│   │   │   ├── data_models.py       # レース・選手・結果データモデル
│   │   │   ├── stadium_manager.py   # 競技場情報管理
│   │   │   └── race_manager.py      # レース情報管理
│   │   ├── prediction/
│   │   │   ├── prediction_engine.py # 予想エンジン
│   │   │   ├── racer_analyzer.py    # 選手分析
│   │   │   ├── race_analyzer.py     # レース分析
│   │   │   └── ml_models.py         # 機械学習モデル
│   │   ├── betting/
│   │   │   ├── ticket_manager.py    # 舟券管理
│   │   │   ├── betting_strategy.py  # 投票戦略
│   │   │   ├── odds_analyzer.py     # オッズ分析
│   │   │   └── portfolio.py         # 投票履歴・収支管理
│   │   ├── data/
│   │   │   ├── data_collector.py    # データ収集
│   │   │   ├── data_processor.py    # データ前処理
│   │   │   └── database.py          # データベース操作
│   │   ├── web/
│   │   │   ├── dashboard.py         # Webダッシュボード
│   │   │   ├── api_routes.py        # REST API
│   │   │   └── templates/           # HTMLテンプレート
│   │   └── utils/
│   │       ├── config.py            # 設定管理
│   │       ├── logger.py            # ログ管理
│   │       └── exceptions.py        # 例外定義
├── config/
│   ├── stadiums.yaml               # 競技場情報
│   ├── prediction_models.yaml     # 予想モデル設定
│   └── betting_strategies.yaml    # 投票戦略設定
├── data/
│   ├── databases/
│   │   ├── boatrace.db            # メインデータベース
│   │   └── predictions.db         # 予想データベース
│   └── cache/                     # APIキャッシュ
├── tests/
└── docs/
```

### 主要機能

#### 1. データ収集・管理
- BoatraceOpenAPIからの自動データ取得
- 出走表、直前情報、結果データの収集・保存
- 24競技場の管理
- 選手データベースの構築・更新

#### 2. 予想エンジン
- 選手成績分析（勝率、連対率、複勝率）
- レース条件分析（コース、天候、風向き）
- 機械学習による予想モデル
- 複数の予想アルゴリズム（統計、ML、複合）

#### 3. 舟券購入支援
- 購入戦略シミュレーション
- リスク管理機能
- 投票履歴・収支管理
- オッズ分析・期待値計算

#### 4. Webダッシュボード
- リアルタイム出走表表示
- 予想結果表示
- 収支グラフ・統計
- レース結果確認

## 技術スタック

### Backend
- **Python 3.11+**
- **FastAPI** - Web API フレームワーク
- **SQLAlchemy** - ORM
- **Alembic** - データベースマイグレーション
- **Pandas** - データ分析
- **Scikit-learn** - 機械学習
- **Requests** - HTTP クライアント
- **SQLite/PostgreSQL** - データベース

### Frontend  
- **HTML/CSS/JavaScript**
- **Bootstrap** - UIフレームワーク
- **Chart.js** - グラフ描画
- **Alpine.js** - 軽量JavaScript フレームワーク

### データベース設計

#### テーブル構成
1. **stadiums** - 競技場マスタ
2. **racers** - 選手マスタ
3. **races** - レースマスタ
4. **race_entries** - 出走情報
5. **race_results** - レース結果
6. **predictions** - 予想データ
7. **betting_tickets** - 購入舟券
8. **betting_results** - 投票結果

## 実装フェーズ

### Phase 1: 基盤構築
- [x] プロジェクト構造設計
- [ ] APIクライアント実装
- [ ] データモデル定義
- [ ] データベース設計・マイグレーション

### Phase 2: データ収集
- [ ] BoatraceOpenAPI統合
- [ ] データ収集バッチ処理
- [ ] 競技場・選手マスタ構築

### Phase 3: 予想エンジン
- [ ] 基本統計分析
- [ ] 機械学習モデル実装
- [ ] 予想アルゴリズム開発

### Phase 4: 舟券管理
- [ ] 投票戦略実装
- [ ] 収支管理機能
- [ ] ポートフォリオ分析

### Phase 5: Web UI
- [ ] ダッシュボード実装
- [ ] レスポンシブデザイン
- [ ] リアルタイム更新

### Phase 6: 高度機能
- [ ] 自動投票システム
- [ ] リアルタイム予想更新
- [ ] レポート機能

## 期待効果
- 競艇予想の精度向上
- 投票戦略の最適化
- 収支管理の効率化
- データドリブンな意思決定支援