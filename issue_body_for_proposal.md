### 概要

本イシューは、既存のバックテストエンジンをさらに強化し、より現実的で包括的な分析機能を提供することを目的とします。具体的には、`src/day_trade/backtesting/backtest_engine.py`の機能を、イベント駆動型である`src/day_trade/analysis/advanced_backtest.py`に統合し、統一された高性能なバックテストエンジンを構築します。これにより、コードの重複を排除し、詳細な取引レベルのPnL分析を洗練させます。

### 現状分析

*   **`src/day_trade/analysis/advanced_backtest.py`**:
    *   イベント駆動型アーキテクチャを採用しており、`MarketDataEvent`や`SignalEvent`などのイベントキューベースの処理ロジックを持っています。
    *   `Order`, `OrderType`, `OrderStatus`などのコアデータ構造は`src/day_trade/analysis/events.py`に集約されつつあります。
    *   複数シンボルデータに対応する柔軟性を持っています。
*   **`src/day_trade/backtesting/backtest_engine.py`**:
    *   日次ベースのバックテストエンジンであり、より詳細な取引コスト計算（手数料、スリッページ）や多様なパフォーマンス指標の計算機能を有しています。
    *   `Order`, `Position`, `Portfolio`などの独自のデータ構造定義を持っています。
    *   `yfinance`を用いた過去データ取得機能が含まれています。

### 課題

1.  **機能の重複と不整合**: `advanced_backtest.py`と`backtest_engine.py`間で、バックテスト機能とデータ構造の定義に重複と不整合があります。これはコードの保守性と拡張性を低下させます。
2.  **取引レベルPnL分析の不足**: `advanced_backtest.py`はイベント駆動型の基盤を持つものの、`backtest_engine.py`にあるような個々の取引の詳細なPnL計算や、それを基にした「勝率」「平均利益/損失」などの分析機能が不足しています。

### 目標

1.  **バックテストエンジンの統一**: `src/day_trade/analysis/advanced_backtest.py`をプロジェクトの主要なバックテストエンジンとして確立し、`backtest_engine.py`の価値ある機能をすべて統合する。
2.  **包括的なPnL分析の実現**: 取引レベルでの詳細な損益（PnL）追跡、コスト計算、およびそれを基にしたパフォーマンス分析機能を`advanced_backtest.py`に完全に統合する。
3.  **データ構造の標準化**: `Order`, `Position`など、全ての取引関連データ構造を`src/day_trade/analysis/events.py`に集約し、システム全体で一貫した利用を保証する。

### 提案される変更とタスク

#### 1. データ構造の標準化と一元化

*   **`src/day_trade/analysis/events.py`**:
    *   `backtest_engine.py`で定義されている`OrderType`, `OrderStatus` (`Enum`) を`events.py`に移動し、既存の`OrderType`, `OrderStatus` (`Enum`) と統合する。
    *   `backtest_engine.py`で定義されている`Order`, `Position`, `Portfolio` (`dataclass`) を`events.py`に移動し、既存のクラスと統合または置き換える。これにより、`advanced_backtest.py`と`backtest_engine.py`の両方で`events.py`からこれらのデータ構造を参照できるようにする。
    *   必要に応じて、`BacktestResults` (`dataclass`) も`events.py`または専用の`results.py`のようなファイルに移動し、標準化を推進する。

#### 2. `backtest_engine.py`機能の`advanced_backtest.py`への統合

*   **`src/day_trade/analysis/advanced_backtest.py`**:
    *   `backtest_engine.py`の`TradingCosts`クラスの定義を`advanced_backtest.py`に統合し、既存のものと整合させる。
    *   `backtest_engine.py`の`_execute_buy_order`と`_execute_sell_order`のロジック（特に取引コスト計算とPnL計算）を、`advanced_backtest.py`の`_handle_fill`メソッド内に統合し、`FillEvent`に基づいて正確なPnLとコストを計算・記録できるようにする。
    *   `backtest_engine.py`の`_analyze_results`メソッドで計算される詳細なパフォーマンス指標（例: `annualized_return`, `volatility`, `sharpe_ratio`, `max_drawdown`, `win_rate`, `avg_trade_return`, `best_trade`, `worst_trade`）を、`advanced_backtest.py`の`_calculate_performance_metrics`に統合・拡張する。`TradeRecord`の情報を活用し、取引レベルでの分析精度を向上させる。
    *   `backtest_engine.py`の`load_historical_data`メソッドのようなデータロード機能を、`advanced_backtest.py`の`run_backtest`または別途のデータハンドリングモジュールに統合・参照させる。これにより、`run_backtest`が直接YFデータに依存せず、より抽象化されたデータ入力インターフェースを持つようにする。
    *   `advanced_backtest.py`内のコメント（`# OrderType と OrderStatus は events.py に移動しました`など）を整理し、コードの整合性を保つ。

#### 3. `src/day_trade/backtesting/backtest_engine.py`の廃止またはリファクタリング

*   上記の統合が完了した後、`backtest_engine.py`は役割を終えるため、削除するか、または`advanced_backtest.py`のサブモジュールとして整理・リファクタリングを検討する。

#### 4. デモスクリプトの更新

*   `run_backtest_demo.py`を更新し、統合された`advanced_backtest.py`の新しいインターフェースと機能（特に複数アセットと詳細なPnL分析）を示す例を記述する。

### 実施順序と優先度

1.  **データ構造の標準化と一元化**: 複数のモジュールに影響するため、最初に実施。
2.  **`backtest_engine.py`機能の`advanced_backtest.py`への統合**: メインとなる機能統合。
3.  **`backtest_engine.py`の廃止またはリファクタリング**: 統合完了後に実施。
4.  **デモスクリプトの更新**: 全ての機能が統合された後に、動作確認とデモンストレーションのために実施。