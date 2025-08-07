# Day Trade - 投資分析支援システム

[![CI/CD Pipeline](https://github.com/kaenozu/day_trade/actions/workflows/optimized-ci.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/optimized-ci.yml)
[![Pre-commit Checks](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/pre-commit.yml)
[![Conflict Detection](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml/badge.svg)](https://github.com/kaenozu/day_trade/actions/workflows/conflict-detection.yml)
[![codecov](https://codecov.io/gh/kaenozu/day_trade/branch/main/graph/badge.svg)](https://codecov.io/gh/kaenozu/day_trade)
[![Test Coverage](https://img.shields.io/badge/coverage-37.5%25-orange.svg)](https://github.com/kaenozu/day_trade/tree/main/reports/coverage)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**高度な投資分析・手動取引支援システム**

## ⚠️ 重要な変更通知

**このシステムは自動取引機能を完全に無効化しています。**

- ✅ **安全な分析専用システム** - 実際の取引は一切実行されません
- ✅ **教育・学習目的に最適** - 投資理論とプログラミングの学習
- ✅ **手動取引支援** - 分析情報の提供と意思決定支援
- ❌ **自動取引なし** - 注文実行機能は完全に無効化済み

## 🎯 プロジェクトの目的

このプロジェクトは**教育・研究目的**で開発された投資分析学習システムです。

### 学習価値
- 📚 **投資理論の理解** - テクニカル分析・ファンダメンタル分析
- 💻 **プログラミングスキル** - Python、非同期処理、システム設計
- 📊 **データ分析手法** - 金融データ処理、統計分析
- 🛠️ **システム設計** - 大規模システムのアーキテクチャ設計

### 実用価値
- 🔍 **市場分析** - 包括的な株式市場分析
- 📈 **投資判断支援** - データに基づく意思決定支援
- 🎯 **手動取引支援** - 取引タイミングと戦略の提案
- 📊 **パフォーマンス追跡** - 投資成果の分析・評価

## 🚀 主な機能

### 📊 市場データ分析
- **リアルタイム株価データ取得** (yfinance API統合)
- **銘柄マスター管理** (東証銘柄情報の自動更新)
- **包括的市場分析** (トレンド、ボラティリティ、相関分析)
- **データキャッシュ機能** (パフォーマンス最適化)

### 🔍 高度なテクニカル分析
- **アンサンブル戦略エンジン** (複数指標の統合判定)
- **テクニカル指標計算** (RSI, MACD, ボリンジャーバンド等)
- **パターン認識** (トレンド分析, サポート/レジスタンス)
- **ボラティリティ分析** (ATR, VIX相関)
- **出来高分析** (VWAP, OBV)

### 🎯 投資判断支援
- **統合シグナル生成** (複数戦略の重み付け評価)
- **リスク分析・警告** (ポートフォリオリスク評価)
- **手動取引支援** (タイミング・ポジションサイズの提案)
- **パフォーマンス追跡** (投資成果分析)

### 🖥️ ユーザーインターフェース
- **インタラクティブCLI** (rich/prompt_toolkit使用)
- **リアルタイム分析ダッシュボード** (価格監視, アラート)
- **詳細レポート生成** (HTML/JSON/CSV出力)
- **カスタマイズ可能なアラート**

### 🤖 分析機能
- **バックテスト実行** (戦略検証・学習)
- **自動スクリーニング** (投資機会発見)
- **定期分析レポート生成**
- **リスクアラートシステム**

## 📦 インストール

### 前提条件
- Python 3.8以上
- pip (パッケージマネージャー)

### セットアップ
```bash
# リポジトリのクローン
git clone https://github.com/kaenozu/day_trade.git
cd day_trade

# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt

# 開発依存関係（オプション）
pip install -r requirements-dev.txt
```

## 🏃 使用方法

### 1. 安全設定確認
```bash
# システム設定の確認
python -c "from src.day_trade.config.trading_mode_config import log_current_configuration; log_current_configuration()"
```

### 2. 分析システムテスト
```bash
# 基本動作テスト
python test_analysis_system.py
```

### 3. 統合分析システム起動
```python
import asyncio
from src.day_trade.core.integrated_analysis_system import IntegratedAnalysisSystem

async def main():
    # 監視したい銘柄を指定
    symbols = ["7203", "6758", "9984"]  # トヨタ、ソニー、ソフトバンク

    # システム初期化
    system = IntegratedAnalysisSystem(symbols)

    # 包括的分析開始（30秒間隔）
    await system.start_comprehensive_analysis(analysis_interval=30.0)

# 実行
asyncio.run(main())
```

### 4. 個別分析実行
```python
from src.day_trade.analysis.market_analysis_system import MarketAnalysisSystem

# 市場分析システム
symbols = ["7203", "6758"]
analysis_system = MarketAnalysisSystem(symbols)

# サンプル市場データで分析実行
market_data = {
    "7203": {"current_price": 2500, "price_change_pct": 1.5, "volume": 1000000},
    "6758": {"current_price": 12000, "price_change_pct": -0.8, "volume": 500000}
}

# 包括的分析実行
analysis_result = await analysis_system.perform_comprehensive_market_analysis(market_data)
print(analysis_result)
```

## 🏗️ システム構成

### 核心モジュール
```
src/day_trade/
├── config/                 # 設定管理（安全性確保）
│   └── trading_mode_config.py  # セーフモード設定
├── core/                   # コアシステム
│   └── integrated_analysis_system.py  # 統合分析システム
├── analysis/               # 分析エンジン
│   ├── market_analysis_system.py      # 市場分析
│   └── signals.py                     # シグナル生成
├── automation/             # 分析エンジン（旧：自動取引）
│   ├── risk_aware_trading_engine.py   # 分析エンジン
│   └── advanced_order_manager.py      # 注文分析（無効化済み）
├── data/                   # データ管理
│   └── stock_fetcher.py    # 株価データ取得
└── utils/                  # ユーティリティ
    └── logging_config.py   # ログ設定
```

### 安全性機能
- **セーフモード強制** - 自動取引の完全無効化
- **設定検証** - 起動時の安全確認
- **実行制限** - 注文API呼び出しの禁止
- **監査ログ** - 全活動の記録

## 🔒 安全性について

### 自動取引無効化
このシステムは以下により自動取引を完全に無効化しています：

1. **設定レベル**: `trading_mode_config.py`でセーフモード強制
2. **コードレベル**: 全注文実行機能の無効化
3. **実行レベル**: API呼び出しの禁止
4. **検証レベル**: 起動時の安全確認

### 利用時の注意
- ✅ 分析・学習目的での利用
- ✅ 手動取引の意思決定支援
- ❌ 実際の自動取引への使用禁止
- ❌ 投資判断の完全依存禁止

## 📚 学習・開発ガイド

### アーキテクチャ学習
- **システム設計パターン** - MVCアーキテクチャ、依存性注入
- **非同期処理** - asyncio、並行処理の実装
- **エラーハンドリング** - 例外処理、ログ管理
- **テスト設計** - 単体テスト、統合テスト

### 投資理論学習
- **テクニカル分析** - 各種指標の計算と解釈
- **ファンダメンタル分析** - 企業価値評価
- **リスク管理** - ポートフォリオ理論、VaR計算
- **行動経済学** - 市場心理の分析

### 拡張開発
1. **新分析手法の追加**
2. **UI/UXの改善**
3. **データソースの拡張**
4. **レポート機能の強化**

## 🧪 テスト

```bash
# 全テスト実行
pytest

# カバレッジ付きテスト
pytest --cov=src/day_trade

# 特定モジュールテスト
pytest tests/test_analysis_system.py

# システム統合テスト
python test_analysis_system.py
```

## 📈 パフォーマンス

### 分析処理能力
- **リアルタイム分析**: 複数銘柄の同時監視
- **データ処理**: 大容量履歴データの高速処理
- **レスポンス時間**: 分析結果の即座提供
- **メモリ効率**: 最適化されたデータ構造

### スケーラビリティ
- **並行処理**: 非同期処理による高速化
- **キャッシュ機能**: データ取得の最適化
- **バッチ処理**: 大量データの効率的処理
- **リソース管理**: メモリ・CPU使用量の最適化

## 🤝 貢献

### 開発参加
1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

### コードスタイル
- **Black**: コードフォーマット
- **isort**: インポート整理
- **flake8**: リンター
- **mypy**: 型チェック

## 📋 ロードマップ

### バージョン 2.0
- [ ] Web UIダッシュボード
- [ ] 機械学習による予測モデル
- [ ] 高度なバックテスト機能
- [ ] モバイルアプリ連携

### バージョン 2.1
- [ ] 仮想通貨対応
- [ ] 海外市場データ統合
- [ ] ソーシャル取引機能
- [ ] API提供機能

## ⚖️ 免責事項

### 重要な注意事項
- **投資判断の責任**: 本システムの分析結果に基づく投資判断は自己責任
- **損失の責任**: 投資による損失について開発者は一切責任を負いません  
- **教育目的**: このシステムは教育・学習目的で開発されています
- **専門家相談**: 重要な投資判断前には専門家にご相談ください

### システムの限界
- **予測の不確実性**: 市場予測は100%正確ではありません
- **データの制約**: 利用可能なデータに基づく分析です
- **技術的制約**: システムの技術的制限があります
- **市場変動**: 急激な市場変化への対応に制限があります

## 📄 ライセンス

MITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご確認ください。

## 📞 サポート・問い合わせ

- **Issue**: [GitHub Issues](https://github.com/kaenozu/day_trade/issues)
- **Discussion**: [GitHub Discussions](https://github.com/kaenozu/day_trade/discussions)
- **Email**: [プロジェクト管理者へのメール](mailto:project@example.com)

---

**Day Trade - 投資分析支援システム**
学習・研究・分析のためのオープンソース投資支援ツール

*Built with ❤️ for educational and research purposes*
