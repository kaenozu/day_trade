### 概要

現在のバックテストエンジンはイベント駆動型アーキテクチャへの移行が進んでおり、そのモジュール性と拡張性は向上しています。本イシューの目的は、この堅牢な基盤をさらに発展させ、現実の市場環境をより正確にシミュレートし、戦略のパフォーマンスをより詳細に分析できるようにすることです。具体的には、複数アセットの同時バックテストを可能にし、個々の取引レベルでの損益（PnL）追跡機能を包括的に実装します。

### 現在の状況と評価

*   `AdvancedBacktestEngine`はイベントキューとイベントハンドラを導入し、イベント駆動型の処理ロジックへと移行しています。
*   基本的なポートフォリオ管理（ポジション、資本、パフォーマンス指標）は実装されています。
*   ロギングとデータクラスの利用は適切です。

### 課題と目標

1.  **複数アセットバックテストの完全な実現**:
    *   **課題**: 現在のエンジンは内部的に複数シンボルを扱える構造を持ちながらも、イベント生成部分が単一シンボルにハードコードされており、真のポートフォリオバックテストを阻害しています。
    *   **目標**: 複数の金融商品を同時に、かつ効率的にバックテストできるようにする。

2.  **包括的な取引レベルPnL追跡**:
    *   **課題**: 個々の取引（エントリーからエグジットまで）の損益（P&L）が明確に追跡されておらず、詳細な取引分析が困難です。
    *   **目標**: 各取引の実現損益、手数料、スリッページを正確に計算・記録し、戦略の強みと弱みを詳細に分析できるようにする。

3.  **イベントキュー処理の効率化（二次的な目標）**:
    *   **課題**: イベントのソートに非効率なプロセスが含まれている。
    *   **目標**: 大規模データセットにおけるバックテストのパフォーマンスを向上させる。

4.  **より現実的な注文約定モデル（二次的な目標）**:
    *   **課題**: 現在の約定モデルは比較的単純で、実世界の複雑な約定シナリオを十分にシミュレートできていない。
    *   **目標**: バックテスト結果の現実性を高め、実運用に近い評価を可能にする。

### 提案される変更とタスク

#### 1. 複数アセットバックテストの完全実装

*   **`src/day_trade/analysis/advanced_backtest.py`**:
    *   `run_backtest`メソッド内で、`data`および`strategy_signals`DataFrameがマルチインデックス（例: `('シンボル', '属性')`）であることを想定し、各シンボルに対して`MarketDataEvent`と`SignalEvent`を動的に生成するように修正する。
    *   **（関連する既存関数への影響）**: `_handle_market_data`、`_handle_signal`、`_update_positions_from_market_data`、`_process_pending_orders`、`_process_strategy_signal`など、シンボルに依存する処理が、複数シンボルに対応できるよう適切にハンドリングされていることを確認する。
*   **`run_backtest_demo.py` (新規作成)**:
    *   複数のシンボルデータを`yfinance`などで取得し、これらをマルチインデックスのDataFrameとして`data`および`signals`を構築するデモコードを作成する。
    *   新しい`AdvancedBacktestEngine`が複数シンボルを処理できることを示す例を追加する。

#### 2. 包括的な取引レベルPnL追跡

*   **`src/day_trade/analysis/events.py`**:
    *   `TradeRecord` dataclassを新規追加する。これには、`trade_id`、`symbol`、`entry_time`、`entry_price`、`entry_quantity`、`entry_commission`、`entry_slippage`、`exit_time`、`exit_price`、`exit_quantity`、`exit_commission`、`exit_slippage`、`realized_pnl`、`total_commission`、`total_slippage`、`is_closed`などのフィールドを含める。
*   **`src/day_trade/analysis/advanced_backtest.py`**:
    *   `AdvancedBacktestEngine`クラスの`__init__`および`_reset_backtest`メソッドで、未決済の取引を追跡するための`self.open_trades: Dict[str, TradeRecord]`のような辞書を導入する。
    *   `self.trade_history`の型を`List[TradeRecord]`に変更する。
    *   `_handle_fill`メソッドを修正し、`FillEvent`を受け取った際に、以下のロジックを実装する:
        *   もし買い（または空売りカバー）約定で、そのシンボルに未決済の`TradeRecord`がない場合、新しい`TradeRecord`を作成し`self.open_trades`に追加する。
        *   もし既存の`TradeRecord`がある場合、その`entry_quantity`、`entry_commission`、`entry_slippage`などを更新する。
        *   もし売り（または買い戻し）約定で、対応する未決済の`TradeRecord`がある場合、その`TradeRecord`を`exit_time`、`exit_price`、`exit_quantity`、`exit_commission`、`exit_slippage`で更新し、`realized_pnl`、`total_commission`、`total_slippage`を計算して設定する。
        *   決済された`TradeRecord`を`self.trade_history`に追加し、`self.open_trades`から削除する。
    *   `_update_position_from_order`メソッドが、約定によって発生した**その時点の**実現損益（P&L）を返すように修正し、`_handle_fill`でPnLを取得して`TradeRecord`に設定できるようにする。
    *   `_calculate_performance_metrics`メソッドを修正し、`self.trade_history`内の`TradeRecord`オブジェクトを使用して、`win_rate`、`profit_factor`、`avg_win`、`avg_loss`、`total_commission`、`total_slippage`などを正確に計算するようにする。特に、P&Lは`TradeRecord`の`realized_pnl`を直接利用する。
    *   `_get_cost_basis_for_sold_trade`関数は、`TradeRecord`の導入により不要となるか、`TradeRecord`の`realized_pnl`計算ロジックに統合される。

#### 3. イベントキュー処理の効率化 (必要に応じて後から着手)

*   `src/day_trade/analysis/advanced_backtest.py`の`run_backtest`メソッド内でイベントをキューに投入する際に、`deque(sorted(list(self.events)...))`のような非効率なソートを避け、最初から順序が保証されるか、またはイベントタイプに応じた適切な優先度キュー（例: `heapq`）の使用を検討する。

#### 4. より現実的な注文約定モデル (必要に応じて後から着手)

*   `src/day_trade/analysis/advanced_backtest.py`の`_should_fill_order`および`_calculate_fill_price`メソッドを拡張し、部分約定、より複雑なスリッページ（例: 出来高ベース）、および市場インパクトをシミュレートするロジックを導入する。
*   `OrderType` Enumに、OCO、OTO、トレイリングストップなどの新しい注文タイプを追加し、それらを処理するロジックを実装する。

### 実施順序と優先度

1.  **複数アセットバックテストの完全実装**: エンジンの根本的な能力を向上させるため、最優先で着手。
2.  **包括的な取引レベルPnL追跡**: 戦略分析の質を大幅に向上させるため、次点での着手。
3.  残りの項目は、上記の主要なタスクが完了し、安定性が確認された後に検討。