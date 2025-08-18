# Day Trade 開発者ガイド

## 目次

1. [開発環境の構築](#開発環境の構築)
2. [アーキテクチャ概要](#アーキテクチャ概要)
3. [コード構造とモジュール](#コード構造とモジュール)
4. [開発ワークフロー](#開発ワークフロー)
5. [テスト戦略](#テスト戦略)
6. [パフォーマンス最適化](#パフォーマンス最適化)
7. [新機能の追加](#新機能の追加)
8. [コード品質管理](#コード品質管理)

## 開発環境の構築

### 前提条件

- Python 3.8以上
- Git
- Visual Studio Code（推奨）または任意のIDE

### 開発環境セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/kaenozu/day_trade.git
cd day_trade

# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 開発用依存関係のインストール
pip install -e .[dev]

# pre-commitフックのインストール
pre-commit install

# データベースの初期化
python -m day_trade.models.database --init

# 動作確認
pytest
```

### IDE設定（VS Code）

#### 必要な拡張機能
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-python.pytest"
  ]
}
```

#### workspace設定（.vscode/settings.json）
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.typeChecking": "strict"
}
```

## アーキテクチャ概要

### 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│                        Day Trade System                      │
├─────────────────┬───────────────┬───────────────┬─────────────┤
│   Presentation  │   Business    │    Data       │   Utilities │
│     Layer       │    Logic      │    Layer      │    Layer    │
│                 │               │               │             │
│ • CLI Interface │ • Analysis    │ • Data Models │ • Logging   │
│ • Interactive   │   Engine      │ • Stock       │ • Caching   │
│   Mode          │ • Trading     │   Fetcher     │ • Exceptions│
│ • Reports       │   Manager     │ • Database    │ • Config    │
│                 │ • Portfolio   │   Manager     │ • Utils     │
│                 │   Manager     │               │             │
└─────────────────┴───────────────┴───────────────┴─────────────┘
```

### 主要コンポーネント

#### 1. Presentation Layer（プレゼンテーション層）
- **CLI Interface**: コマンドライン操作
- **Interactive Mode**: 対話型モード
- **Reports**: レポート生成

#### 2. Business Logic Layer（ビジネスロジック層）
- **Analysis Engine**: テクニカル分析・シグナル生成
- **Trading Manager**: 取引管理・実行
- **Portfolio Manager**: ポートフォリオ管理

#### 3. Data Layer（データ層）
- **Data Models**: データベーススキーマ
- **Stock Fetcher**: 外部API連携
- **Database Manager**: データ永続化

#### 4. Utilities Layer（ユーティリティ層）
- **Logging**: 構造化ログ
- **Caching**: データキャッシュ
- **Configuration**: 設定管理

### デザインパターン

#### Repository Pattern
```python
class StockRepository:
    """株価データのアクセス抽象化"""

    def get_price_data(self, symbol: str) -> PriceData:
        pass

    def save_price_data(self, data: PriceData) -> None:
        pass

class SQLAlchemyStockRepository(StockRepository):
    """SQLAlchemy実装"""

    def get_price_data(self, symbol: str) -> PriceData:
        # データベースから取得
        pass
```

#### Strategy Pattern
```python
class TradingStrategy:
    """取引戦略の抽象クラス"""

    def generate_signal(self, data: MarketData) -> TradingSignal:
        pass

class MomentumStrategy(TradingStrategy):
    """モメンタム戦略"""

    def generate_signal(self, data: MarketData) -> TradingSignal:
        # モメンタム分析ロジック
        pass
```

#### Observer Pattern
```python
class AlertManager:
    """アラート管理"""

    def __init__(self):
        self.observers = []

    def add_observer(self, observer: AlertObserver):
        self.observers.append(observer)

    def notify_alert(self, alert: Alert):
        for observer in self.observers:
            observer.handle_alert(alert)
```

## コード構造とモジュール

### ディレクトリ構造詳細

```
src/day_trade/
├── analysis/              # 分析エンジン
│   ├── __init__.py
│   ├── ensemble.py        # アンサンブル戦略
│   ├── indicators.py      # テクニカル指標
│   ├── patterns.py        # パターン認識
│   ├── signals.py         # シグナル生成
│   ├── backtest.py        # バックテスト
│   └── screener.py        # スクリーニング
│
├── automation/            # 自動化機能
│   ├── __init__.py
│   ├── orchestrator.py    # 自動実行管理
│   └── scheduler.py       # スケジューラー
│
├── cli/                   # コマンドライン
│   ├── __init__.py
│   ├── main.py           # メインCLI
│   ├── interactive.py    # 対話型モード
│   └── commands/         # 個別コマンド
│
├── core/                  # コア機能
│   ├── __init__.py
│   ├── portfolio.py      # ポートフォリオ管理
│   ├── trade_manager.py  # 取引管理
│   ├── watchlist.py      # ウォッチリスト
│   ├── alerts.py         # アラート機能
│   └── config.py         # 設定管理
│
├── data/                  # データ取得・管理
│   ├── __init__.py
│   ├── stock_fetcher.py  # 株価データ取得
│   ├── enhanced_stock_fetcher.py  # 耐障害性版
│   └── stock_master.py   # 銘柄マスター
│
├── models/                # データモデル
│   ├── __init__.py
│   ├── database.py       # データベース定義
│   └── schemas.py        # データスキーマ
│
└── utils/                 # ユーティリティ
    ├── __init__.py
    ├── logging_config.py  # ログ設定
    ├── exceptions.py      # 例外定義
    ├── cache_utils.py     # キャッシュ
    ├── api_resilience.py  # API耐障害性
    └── performance_analyzer.py  # パフォーマンス分析
```

### 主要モジュールの詳細

#### analysis/ensemble.py
```python
class EnsembleTradingStrategy:
    """アンサンブル取引戦略"""

    def __init__(self, strategies: List[TradingStrategy], weights: Dict[str, float]):
        self.strategies = strategies
        self.weights = weights

    def generate_ensemble_signal(self, data: MarketData) -> EnsembleSignal:
        """複数戦略の結果を統合してシグナル生成"""
        signals = []
        for strategy in self.strategies:
            signal = strategy.generate_signal(data)
            signals.append(signal)

        # 重み付き投票による統合
        ensemble_signal = self._combine_signals(signals)
        return ensemble_signal
```

#### core/portfolio.py
```python
class PortfolioManager:
    """ポートフォリオ管理クラス"""

    def __init__(self, db_session):
        self.db_session = db_session

    def add_position(self, symbol: str, quantity: int, price: float) -> Position:
        """ポジション追加"""
        position = Position(symbol=symbol, quantity=quantity, price=price)
        self.db_session.add(position)
        self.db_session.commit()
        return position

    def calculate_performance(self, period: str = "1M") -> PerformanceMetrics:
        """パフォーマンス計算"""
        # 期間収益率、シャープレシオ等の計算
        pass
```

#### data/enhanced_stock_fetcher.py
```python
class EnhancedStockFetcher(StockFetcher):
    """耐障害性強化版データ取得"""

    def __init__(self, enable_fallback=True, enable_circuit_breaker=True):
        super().__init__()
        self.enable_fallback = enable_fallback
        self.enable_circuit_breaker = enable_circuit_breaker
        if enable_circuit_breaker:
            self._setup_resilience()

    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """フォールバック機能付き価格取得"""
        return self._enhanced_retry_on_error(super().get_current_price, symbol)
```

## 開発ワークフロー

### Git Flow

#### ブランチ戦略
```bash
# 機能開発
git checkout -b feature/new-indicator
git commit -m "feat: Add new technical indicator"
git push origin feature/new-indicator

# バグ修正
git checkout -b fix/portfolio-calculation
git commit -m "fix: Correct portfolio performance calculation"
git push origin fix/portfolio-calculation

# ホットフィックス
git checkout -b hotfix/critical-data-issue
git commit -m "fix: Critical data fetching issue"
git push origin hotfix/critical-data-issue
```

#### コミットメッセージ規約
```bash
# 形式: <type>(<scope>): <description>

feat(analysis): Add RSI divergence detection
fix(portfolio): Fix position calculation bug
docs(api): Update API documentation
style(cli): Format code with black
refactor(data): Improve data fetching performance
test(backtest): Add comprehensive backtest tests
chore(deps): Update dependencies
```

### プルリクエストワークフロー

#### 1. 作業前の準備
```bash
# 最新のmainブランチに同期
git checkout main
git pull origin main

# 新しいブランチ作成
git checkout -b feature/your-feature-name
```

#### 2. 開発・テスト
```bash
# コード修正
# ...

# テスト実行
pytest

# リンター実行
ruff check .
black .

# 型チェック
mypy src/
```

#### 3. プルリクエスト作成
```bash
# コミット・プッシュ
git add .
git commit -m "feat: Add your feature"
git push origin feature/your-feature-name

# GitHub でプルリクエスト作成
```

### コードレビューガイドライン

#### レビュー観点
1. **機能性**: 要件を満たしているか
2. **可読性**: コードが理解しやすいか
3. **保守性**: 将来の変更が容易か
4. **パフォーマンス**: 効率的な実装か
5. **テスト**: 適切なテストカバレッジか
6. **文書化**: 必要な文書は更新されているか

#### レビューテンプレート
```markdown
## 変更内容
- [ ] 新機能追加
- [ ] バグ修正
- [ ] パフォーマンス改善
- [ ] リファクタリング
- [ ] ドキュメント更新

## テスト
- [ ] ユニットテストが追加・更新されている
- [ ] 統合テストが通る
- [ ] 手動テストが完了

## 影響範囲
- [ ] 既存機能に影響なし
- [ ] データベーススキーマ変更あり
- [ ] 設定ファイル変更あり

## その他
- [ ] ドキュメントが更新されている
- [ ] CHANGELOG.mdが更新されている
```

## テスト戦略

### テスト構成

```
tests/
├── unit/                  # ユニットテスト
│   ├── test_indicators.py
│   ├── test_portfolio.py
│   └── test_signals.py
├── integration/           # 統合テスト
│   ├── test_data_flow.py
│   └── test_api_integration.py
├── performance/           # パフォーマンステスト
│   └── test_backtest_performance.py
└── fixtures/              # テストデータ
    ├── sample_data.csv
    └── test_config.json
```

### テスト実行

```bash
# 全テスト実行
pytest

# 特定のテストファイル
pytest tests/unit/test_indicators.py

# カバレッジ付き
pytest --cov=src/day_trade --cov-report=html

# 並列実行
pytest -n auto

# 詳細出力
pytest -v

# 失敗したテストのみ再実行
pytest --lf

# 新しく追加されたテストのみ
pytest --nf
```

### テスト作成ガイドライン

#### ユニットテストの例
```python
import pytest
from src.day_trade.analysis.indicators import RSI

class TestRSI:
    """RSI指標のテストクラス"""

    def test_rsi_calculation_basic(self):
        """基本的なRSI計算のテスト"""
        prices = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]
        rsi = RSI(period=5)
        result = rsi.calculate(prices)

        assert len(result) == len(prices) - 4  # 期間分短くなる
        assert 0 <= result[-1] <= 100  # RSIは0-100の範囲

    def test_rsi_extreme_values(self):
        """極端な値でのテスト"""
        # 連続上昇
        rising_prices = list(range(1, 21))
        rsi = RSI(period=14)
        result = rsi.calculate(rising_prices)
        assert result[-1] == 100  # 連続上昇なのでRSI=100

        # 連続下降
        falling_prices = list(range(20, 0, -1))
        result = rsi.calculate(falling_prices)
        assert result[-1] == 0  # 連続下降なのでRSI=0

    @pytest.mark.parametrize("period", [5, 10, 14, 20])
    def test_rsi_different_periods(self, period):
        """異なる期間でのテスト"""
        prices = [10 + i * 0.1 for i in range(50)]
        rsi = RSI(period=period)
        result = rsi.calculate(prices)

        assert len(result) == len(prices) - period + 1
        assert all(0 <= val <= 100 for val in result)
```

#### 統合テストの例
```python
import pytest
from src.day_trade.core.portfolio import PortfolioManager
from src.day_trade.data.stock_fetcher import StockFetcher

class TestPortfolioIntegration:
    """ポートフォリオ統合テスト"""

    @pytest.fixture
    def portfolio_manager(self, db_session):
        return PortfolioManager(db_session)

    @pytest.fixture
    def stock_fetcher(self):
        return StockFetcher()

    def test_portfolio_with_real_data(self, portfolio_manager, stock_fetcher):
        """実際のデータを使ったポートフォリオテスト"""
        # ポートフォリオに銘柄追加
        portfolio_manager.add_position("7203", 100, 2500)

        # 現在価格取得
        current_data = stock_fetcher.get_current_price("7203")

        # パフォーマンス計算
        performance = portfolio_manager.calculate_performance()

        assert performance.total_value > 0
        assert performance.unrealized_pnl is not None
```

### モック・フィクスチャの活用

```python
import pytest
from unittest.mock import Mock, patch
import pandas as pd

@pytest.fixture
def sample_price_data():
    """テスト用価格データ"""
    return pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Open': [100 + i * 0.5 for i in range(100)],
        'High': [101 + i * 0.5 for i in range(100)],
        'Low': [99 + i * 0.5 for i in range(100)],
        'Close': [100.5 + i * 0.5 for i in range(100)],
        'Volume': [1000000] * 100
    })

@pytest.fixture
def mock_stock_fetcher():
    """モック化されたStockFetcher"""
    mock = Mock()
    mock.get_current_price.return_value = {
        'symbol': '7203.T',
        'current_price': 2500.0,
        'change': 25.0,
        'change_percent': 1.0
    }
    return mock

def test_with_mock(mock_stock_fetcher):
    """モックを使ったテスト"""
    price = mock_stock_fetcher.get_current_price("7203")
    assert price['current_price'] == 2500.0
```

## パフォーマンス最適化

### パフォーマンス分析

#### プロファイリング
```python
import cProfile
import pstats
from src.day_trade.analysis.backtest import BacktestEngine

def profile_backtest():
    """バックテストのプロファイリング"""
    profiler = cProfile.Profile()
    profiler.enable()

    # バックテスト実行
    engine = BacktestEngine()
    result = engine.run_backtest("7203", "2024-01-01", "2024-12-31")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == "__main__":
    profile_backtest()
```

#### メモリ使用量監視
```python
import tracemalloc
from src.day_trade.utils.performance_analyzer import MemoryAnalyzer

def analyze_memory_usage():
    """メモリ使用量分析"""
    tracemalloc.start()

    # 重い処理の実行
    analyzer = MemoryAnalyzer()
    result = analyzer.analyze_large_dataset()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

    tracemalloc.stop()
```

### 最適化テクニック

#### データ処理の最適化
```python
# 悪い例：ループでの逐次処理
def calculate_sma_slow(prices, period):
    sma = []
    for i in range(period - 1, len(prices)):
        avg = sum(prices[i - period + 1:i + 1]) / period
        sma.append(avg)
    return sma

# 良い例：pandas/numpyのベクトル化
def calculate_sma_fast(prices, period):
    return prices.rolling(window=period).mean()
```

#### キャッシュの効果的利用
```python
from functools import lru_cache
from src.day_trade.utils.cache_utils import cache_with_ttl

class OptimizedAnalyzer:
    @lru_cache(maxsize=128)
    def get_indicator_config(self, indicator_name: str):
        """設定情報のキャッシュ"""
        return self._load_indicator_config(indicator_name)

    @cache_with_ttl(300)  # 5分間キャッシュ
    def get_market_data(self, symbol: str):
        """市場データのキャッシュ"""
        return self._fetch_market_data(symbol)
```

#### 並列処理の活用
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

class ParallelAnalyzer:
    def analyze_multiple_stocks(self, symbols: List[str]):
        """複数銘柄の並列分析"""
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.analyze_stock, symbol): symbol
                for symbol in symbols
            }

            results = {}
            for future in futures:
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")

            return results

    def parallel_backtest(self, configurations: List[dict]):
        """並列バックテスト"""
        cpu_count = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [
                executor.submit(self.run_single_backtest, config)
                for config in configurations
            ]

            return [future.result() for future in futures]
```

## 新機能の追加

### 新しいテクニカル指標の追加

#### 1. 指標クラスの作成
```python
# src/day_trade/analysis/indicators.py
class StochasticOscillator:
    """ストキャスティクス指標"""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period

    def calculate(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """ストキャスティクス計算"""
        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=self.d_period).mean()

        return {
            'K': k_percent,
            'D': d_percent
        }
```

#### 2. テストの追加
```python
# tests/unit/test_indicators.py
class TestStochasticOscillator:
    def test_stochastic_calculation(self):
        """ストキャスティクス計算テスト"""
        # テストデータ準備
        data = pd.DataFrame({
            'High': [110, 115, 120, 118, 116],
            'Low': [105, 108, 110, 112, 110],
            'Close': [108, 112, 118, 115, 114]
        })

        stoch = StochasticOscillator(k_period=3, d_period=2)
        result = stoch.calculate(data['High'], data['Low'], data['Close'])

        assert 'K' in result
        assert 'D' in result
        assert all(0 <= val <= 100 for val in result['K'].dropna())
```

#### 3. ドキュメントの更新
```python
# docs/technical_indicators.md に追加
"""
## ストキャスティクス (Stochastic Oscillator)

### 概要
ストキャスティクスは、価格の勢いを測定するモメンタム指標です。

### 計算式
%K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
%D = %Kの移動平均

### 使用方法
```python
from src.day_trade.analysis.indicators import StochasticOscillator

stoch = StochasticOscillator(k_period=14, d_period=3)
result = stoch.calculate(high, low, close)
```
"""
```

### 新しい取引戦略の追加

#### 1. 戦略クラスの作成
```python
# src/day_trade/analysis/strategies/pairs_trading.py
class PairsTradingStrategy(TradingStrategy):
    """ペアトレード戦略"""

    def __init__(self, symbol1: str, symbol2: str, lookback: int = 60):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.lookback = lookback

    def generate_signal(self, data1: pd.DataFrame, data2: pd.DataFrame) -> TradingSignal:
        """ペアトレードシグナル生成"""
        # 価格比率の計算
        ratio = data1['Close'] / data2['Close']

        # 移動平均からの乖離
        mean_ratio = ratio.rolling(self.lookback).mean()
        std_ratio = ratio.rolling(self.lookback).std()
        z_score = (ratio.iloc[-1] - mean_ratio.iloc[-1]) / std_ratio.iloc[-1]

        # シグナル判定
        if z_score > 2:
            return TradingSignal(
                signal_type=SignalType.SELL,
                strength=SignalStrength.STRONG,
                confidence=abs(z_score) * 25,
                pair_action=(SignalType.SELL, SignalType.BUY)
            )
        elif z_score < -2:
            return TradingSignal(
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=abs(z_score) * 25,
                pair_action=(SignalType.BUY, SignalType.SELL)
            )
        else:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=50
            )
```

#### 2. 統合とテスト
```python
# tests/integration/test_pairs_trading.py
class TestPairsTradingIntegration:
    def test_pairs_trading_with_real_data(self):
        """実データでのペアトレードテスト"""
        strategy = PairsTradingStrategy("7203", "6758")  # トヨタ vs ソニー

        # データ取得
        fetcher = StockFetcher()
        data1 = fetcher.get_historical_data("7203", period="1y")
        data2 = fetcher.get_historical_data("6758", period="1y")

        # シグナル生成
        signal = strategy.generate_signal(data1, data2)

        assert signal is not None
        assert hasattr(signal, 'pair_action')
```

## コード品質管理

### 品質チェックツール

#### Ruff（リンター・フォーマッター）
```bash
# リント実行
ruff check .

# 自動修正
ruff check --fix .

# 設定ファイル（pyproject.toml）
[tool.ruff]
target-version = "py38"
line-length = 88
select = ["E", "F", "W", "C90", "I", "N", "UP", "YTT", "S", "BLE", "B", "A", "COM", "C4", "DTZ", "T10", "EM", "EXE", "ISC", "ICN", "G", "INP", "PIE", "T20", "PYI", "PT", "Q", "RSE", "RET", "SLF", "SIM", "TID", "TCH", "ARG", "PTH", "ERA", "PGH", "PL", "TRY", "NPY", "RUF"]
ignore = ["E501", "S101", "S311"]
```

#### MyPy（型チェック）
```bash
# 型チェック実行
mypy src/

# 設定ファイル（pyproject.toml）
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
```

#### Bandit（セキュリティチェック）
```bash
# セキュリティチェック実行
bandit -r src/

# 設定ファイル（pyproject.toml）
[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101", "B601"]
```

### コード品質指標

#### カバレッジ目標
- **ユニットテスト**: 80%以上
- **統合テスト**: 60%以上
- **全体**: 75%以上

#### 複雑度指標
- **循環的複雑度**: 10以下
- **認知的複雑度**: 15以下
- **ネスト階層**: 4以下

#### パフォーマンス指標
- **レスポンス時間**: 1秒以下
- **メモリ使用量**: 100MB以下
- **CPU使用率**: 70%以下

### CI/CDパイプライン

#### GitHub Actions設定例
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Lint with ruff
      run: ruff check .

    - name: Type check with mypy
      run: mypy src/

    - name: Security check with bandit
      run: bandit -r src/

    - name: Test with pytest
      run: pytest --cov=src/day_trade --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

---

このガイドに従って開発を進めることで、高品質で保守性の高いコードを維持できます。質問や不明点があれば、Issueや Discussionでお気軽にお聞きください。
