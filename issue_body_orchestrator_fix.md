**タイトル:** `DayTradeOrchestrator._generate_ensemble_signals` で引数の不整合によりエラーが発生

**再現手順:**
1. `daytrade.py` または `demo_automation.py` などを実行し、`DayTradeOrchestrator` を経由してアンサンブル戦略によるシグナル生成を行う。
2. ログに以下のエラーが出力されることを確認する。
   ```
   ERROR - シグナル生成エラー (xxxx): DayTradeOrchestrator._generate_ensemble_signals() takes 4 positional arguments but 5 were given
   ```

**問題:**
`src/day_trade/automation/orchestrator.py` の `DayTradeOrchestrator` クラスにおいて、`_generate_signals_batch` メソッドから `_generate_ensemble_signals` メソッドを呼び出す際に、引数が一つ多く渡されている。

- `_generate_ensemble_signals` の定義： `(self, symbol: str, analysis: Dict, patterns: Dict)`
- `_generate_ensemble_signals` の呼び出し： `(symbol, analysis, patterns, symbol_stock_data)`

また、`_generate_ensemble_signals` メソッド内では、渡されるべき `stock_data` を使用せず、ハードコードされたダミーの `DataFrame` (`dummy_df`) を使用してシグナル生成を行っている。

**原因:**
- メソッド呼び出しと定義の引数の数が一致していない。
- `_generate_ensemble_signals` の実装が不完全で、実際の株価データではなくダミーデータを使用している。

**修正案:**
1. `_generate_ensemble_signals` メソッドの定義を以下のように修正し、`stock_data` を受け取れるようにする。
   ```python
   def _generate_ensemble_signals(
       self, symbol: str, analysis: Dict, patterns: Dict, stock_data: Dict[str, Any]
   ) -> List[Dict[str, Any]]:
   ```
2. `_generate_ensemble_signals` メソッド内で `dummy_df` を作成している箇所を削除し、引数で受け取った `stock_data` から履歴データ (`historical`) を取得して `DataFrame` として利用するように修正する。
   ```python
   if not stock_data or "historical" not in stock_data or stock_data["historical"] is None:
       logger.warning(f"アンサンブルシグナル生成のための履歴データがありません ({symbol})")
       return []

   historical_df = stock_data["historical"]

   # ...

   ensemble_signal = self.ensemble_strategy.generate_ensemble_signal(
       historical_df, indicators_df, patterns
   )
   ```

**期待される結果:**
- `_generate_ensemble_signals` 実行時に引数エラーが発生しなくなる。
- アンサンブル戦略のシグナル生成が、実際の株価データを用いて正しく行われるようになる。
