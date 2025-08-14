# Issue #900: 商用機能除去実行計画

## 🎯 除去対象ファイル・ディレクトリ詳細

### 🗂️ 完全削除対象ディレクトリ

#### 1. API Gateway関連
```bash
# 完全削除
rm -rf api_gateway/
├── docker-compose.kong.yml
├── kong/kong.yml
├── monitoring/logstash/kong-logstash.conf
└── [その他Kong設定ファイル]
```

#### 2. マイクロサービス関連
```bash
# 完全削除
rm -rf microservices/
├── data_service/app.py
├── ml_service/app.py
├── integration_tests/
└── scripts/run_integration_tests.sh
```

#### 3. 複雑なインフラ設定
```bash
# 完全削除
rm -rf infrastructure/
├── terraform/
├── helm/
└── kubernetes/

rm -rf service_mesh/
├── istio/
└── security-policies.yaml
```

#### 4. エンタープライズ監視
```bash
# 完全削除
rm -rf monitoring/elasticsearch/
rm -rf monitoring/sla_reporting/
rm -rf monitoring/alerts/
└── [複雑な監視設定]
```

#### 5. 高度なセキュリティ
```bash
# 完全削除または簡素化
rm -rf security/auth/api_key_manager.py
rm -rf security/monitoring/intrusion_detection.py
rm -rf security/crypto/encryption_manager.py
```

### 📝 修正・簡素化対象ファイル

#### 1. メイン実行ファイル統合
```python
# daytrade.py - シンプル化版
#!/usr/bin/env python3
"""
Day Trade ML System - 個人利用版
Simple Personal Trading System
"""

import asyncio
from src.day_trade.core.simple_engine import SimpleTradingEngine

async def main():
    """シンプル実行"""
    engine = SimpleTradingEngine()
    await engine.run_analysis()
    engine.display_results()

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. 設定ファイル統合
```yaml
# config.yaml - 統一設定
app:
  name: "Day Trade Personal"
  log_level: "INFO"

data:
  source: "yfinance"
  cache_enabled: true
  sqlite_path: "./data/day_trade.db"

prediction:
  model_path: "./models/"
  ensemble_enabled: true
  accuracy_target: 0.93

output:
  format: "console"  # console, file
  save_results: true
  results_path: "./results/"
```

## 🔄 機能統合・簡素化計画

### 1. モノリス化統合
```
現在: マイクロサービス分散
├── ml-service (port 8000)
├── data-service (port 8001)  
├── symbol-service (port 8002)
├── execution-service (port 8003)
└── notification-service (port 8004)

変更後: 単一プロセス
└── daytrade.py (統合実行)
    ├── MLエンジン
    ├── データ取得
    ├── 銘柄選択
    ├── 実行管理
    └── 結果出力
```

### 2. データベース簡素化
```
現在: PostgreSQL + Redis + 複雑なORM
変更後: SQLite + シンプルORM
```

### 3. 監視簡素化
```
現在: Prometheus + Grafana + ELK
変更後: コンソールログ + ファイル出力
```

### 4. 認証簡素化
```
現在: JWT + RBAC + APIキー + レート制限
変更後: 認証なし（個人利用）
```

## 📦 新しいディレクトリ構造

### 🎯 個人利用向け構造
```
day_trade/
├── daytrade.py              # メイン実行ファイル
├── config.yaml              # 統一設定
├── requirements.txt         # 必要最小限の依存関係
├── README.md               # 個人向け簡単ガイド
├── src/day_trade/
│   ├── core/               # コア機能（93%精度エンジン）
│   │   ├── ensemble_system.py
│   │   ├── prediction_engine.py
│   │   └── simple_engine.py  # 新規：統合エンジン
│   ├── data/               # データ処理
│   │   ├── fetcher.py
│   │   ├── cache.py
│   │   └── sqlite_manager.py
│   ├── analysis/           # 分析機能
│   │   ├── technical_indicators.py
│   │   └── ensemble_models.py
│   └── utils/              # ユーティリティ
│       ├── logging.py      # シンプルログ
│       └── config.py       # 設定管理
├── data/                   # ローカルデータ
│   ├── day_trade.db       # SQLiteデータベース
│   └── cache/             # キャッシュファイル
├── models/                 # 訓練済みモデル
├── results/               # 結果出力
└── tests/                 # 基本テスト
    ├── test_core.py
    └── test_integration.py
```

## 🛠️ 実装手順

### Phase 1: ファイル削除・移動
```bash
# 1. 商用ディレクトリ削除
rm -rf api_gateway/ microservices/ infrastructure/ service_mesh/
rm -rf monitoring/elasticsearch/ monitoring/sla_reporting/
rm -rf security/auth/ security/monitoring/

# 2. 必要機能をcoreに統合
mkdir -p src/day_trade/core/
# ML関連機能を統合

# 3. 設定ファイル統合
# 複数設定 → config.yaml
```

### Phase 2: コード統合・簡素化
```python
# 新規: src/day_trade/core/simple_engine.py
class SimpleTradingEngine:
    """個人利用向け統合エンジン"""

    def __init__(self, config_path="config.yaml"):
        self.config = self.load_config(config_path)
        self.ensemble_system = EnsembleSystem()  # Issue #762
        self.feature_engine = RealTimeFeatureEngine()  # Issue #763
        self.inference_engine = OptimizedInferenceEngine()  # Issue #761

    async def run_analysis(self):
        """シンプル分析実行"""
        # 93%精度予測を実行
        # 複雑な分散処理なし
        pass
```

### Phase 3: ドキュメント更新
```markdown
# README.md - 個人向け
## 3分でスタート
git clone [repo]
cd day_trade
pip install -r requirements.txt
python daytrade.py

## これだけで93%精度の予測開始！
```

## ✅ 検証項目

### 🎯 機能維持確認
- [ ] 93%予測精度維持
- [ ] リアルタイム処理機能
- [ ] アンサンブルシステム動作
- [ ] データ取得・保存機能

### 🚀 簡素化確認
- [ ] インストール3ステップ以下
- [ ] 設定ファイル5個以下
- [ ] メモリ使用量50%削減
- [ ] 実行時間短縮

### 📚 ドキュメント確認
- [ ] 個人向けREADME
- [ ] 複雑な企業ドキュメント除去
- [ ] 簡単トラブルシューティング

## 🎉 期待される最終状態

### 👤 理想的な個人利用体験
```bash
# ユーザーの体験
$ git clone [repo]
$ cd day_trade
$ pip install -r requirements.txt
$ python daytrade.py

🚀 Day Trade ML System (個人版) 起動中...
📊 データ取得中... ✅
🧠 93%精度AI分析中... ✅
📈 推奨銘柄:
   1. 7203 (トヨタ) - BUY (93.2%信頼度)
   2. 8306 (MUFG) - HOLD (91.8%信頼度)
   3. 9984 (ソフトバンク) - SELL (94.1%信頼度)
```

---

**実装優先度**: 🔥 **最高**
**影響範囲**: 📦 **システム全体**
**複雑度**: ⚠️ **大規模リファクタリング**