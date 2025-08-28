from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from ..models.enums import TradeType
from ..utils.logging_config import get_context_logger
from .trade_models import BuyLot, Position, RealizedPnL, Trade


class PortfolioManager:
    def __init__(self, trade_manager):
        self.tm = trade_manager
        self.logger = get_context_logger(__name__)

    def _update_position(self, trade: Trade) -> None:
        """
        ポジション更新（FIFO会計原則・強化版）

        買いロットキューを使用して厳密なFIFO管理を実現
        """
        symbol = trade.symbol

        if trade.trade_type == TradeType.BUY:
            if symbol in self.tm.positions:
                # 既存ポジションに買いロットを追加
                position = self.tm.positions[symbol]

                # 新しい買いロットを作成
                new_lot = BuyLot(
                    quantity=trade.quantity,
                    price=trade.price,
                    commission=trade.commission,
                    timestamp=trade.timestamp,
                    trade_id=trade.id,
                )

                # 買いロットキューに追加（FIFO）
                position.buy_lots.append(new_lot)

                # ポジション全体を更新
                total_cost = (
                    position.total_cost
                    + (trade.price * Decimal(trade.quantity))
                    + trade.commission
                )
                total_quantity = position.quantity + trade.quantity
                average_price = total_cost / Decimal(total_quantity)

                position.quantity = total_quantity
                position.average_price = average_price
                position.total_cost = total_cost
            else:
                # 新規ポジション
                total_cost = (trade.price * Decimal(trade.quantity)) + trade.commission

                # 初期買いロット作成
                initial_lot = BuyLot(
                    quantity=trade.quantity,
                    price=trade.price,
                    commission=trade.commission,
                    timestamp=trade.timestamp,
                    trade_id=trade.id,
                )

                new_position = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    average_price=total_cost / Decimal(trade.quantity),
                    total_cost=total_cost,
                )
                new_position.buy_lots = deque([initial_lot])

                self.tm.positions[symbol] = new_position

        elif trade.trade_type == TradeType.SELL:
            if symbol in self.tm.positions:
                position = self.tm.positions[symbol]

                if position.quantity >= trade.quantity:
                    # 厳密なFIFO会計による実現損益計算
                    self._process_sell_fifo(position, trade)
                else:
                    self.logger.warning(
                        f"銘柄 '{symbol}' の売却数量が保有数量 ({position.quantity}) を超過しています。売却数量: {trade.quantity}。取引は処理されません。",
                        extra={
                            "symbol": symbol,
                            "available_quantity": position.quantity,
                            "requested_quantity": trade.quantity,
                        },
                    )
            else:
                self.logger.warning(
                    f"ポジションを保有していない銘柄 '{symbol}' の売却を試みました。取引は無視されます。",
                    extra={"symbol": symbol, "trade_type": "SELL"},
                )

    def _process_sell_fifo(self, position: Position, sell_trade: Trade) -> None:
        """
        FIFO原則による売却処理（厳密版）

        買いロットキューから順次売却し、正確な実現損益を計算
        """
        remaining_sell_quantity = sell_trade.quantity
        sell_price = sell_trade.price
        sell_commission = sell_trade.commission
        symbol = sell_trade.symbol

        # 売却処理による実現損益を累積
        total_buy_cost = Decimal("0")
        total_buy_commission = Decimal("0")
        total_sold_quantity = 0
        earliest_buy_date = None

        while remaining_sell_quantity > 0 and position.buy_lots:
            # 最古のロット（FIFO）を取得
            oldest_lot = position.buy_lots.popleft()

            if earliest_buy_date is None:
                earliest_buy_date = oldest_lot.timestamp

            # 売却数量の決定
            quantity_to_sell = min(remaining_sell_quantity, oldest_lot.quantity)

            # 売却分のコスト按分
            lot_cost_per_share = oldest_lot.total_cost_per_share()
            buy_cost_for_this_sale = lot_cost_per_share * Decimal(quantity_to_sell)
            buy_commission_for_this_sale = (
                oldest_lot.commission
                * Decimal(quantity_to_sell)
                / Decimal(oldest_lot.quantity)
            )

            total_buy_cost += buy_cost_for_this_sale
            total_buy_commission += buy_commission_for_this_sale
            total_sold_quantity += quantity_to_sell

            # ロットを部分的に消費
            if oldest_lot.quantity > quantity_to_sell:
                # ロットの残りを戻す
                remaining_lot = BuyLot(
                    quantity=oldest_lot.quantity - quantity_to_sell,
                    price=oldest_lot.price,
                    commission=oldest_lot.commission
                    * Decimal(oldest_lot.quantity - quantity_to_sell)
                    / Decimal(oldest_lot.quantity),
                    timestamp=oldest_lot.timestamp,
                    trade_id=oldest_lot.trade_id,
                )
                position.buy_lots.appendleft(remaining_lot)

            remaining_sell_quantity -= quantity_to_sell

        if total_sold_quantity > 0:
            # 実現損益計算
            gross_proceeds = sell_price * Decimal(total_sold_quantity) - sell_commission
            cost_basis = total_buy_cost
            pnl_before_tax = gross_proceeds - cost_basis

            # 税金計算（利益が出た場合のみ）
            tax = Decimal("0")
            if pnl_before_tax > 0:
                tax = pnl_before_tax * self.tm.tax_rate

            # 最終的な実現損益（税引き後）
            pnl = pnl_before_tax - tax

            # 収益率計算
            pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else Decimal("0")

            # 平均買い価格の計算
            average_buy_price = (
                cost_basis / Decimal(total_sold_quantity)
                if total_sold_quantity > 0
                else Decimal("0")
            )

            # 実現損益を記録
            realized_pnl = RealizedPnL(
                symbol=symbol,
                quantity=total_sold_quantity,
                buy_price=average_buy_price,
                sell_price=sell_price,
                buy_commission=total_buy_commission,
                sell_commission=sell_commission,
                pnl=pnl,
                pnl_percent=pnl_percent,
                buy_date=earliest_buy_date or sell_trade.timestamp,
                sell_date=sell_trade.timestamp,
            )

            self.tm.realized_pnl.append(realized_pnl)

            # ポジション情報を更新
            position.quantity -= total_sold_quantity
            position.total_cost -= cost_basis

            if position.quantity > 0:
                # 平均価格の再計算
                position.average_price = position.total_cost / Decimal(
                    position.quantity
                )
            else:
                # ポジション完全クローズ
                del self.tm.positions[symbol]

    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジション情報を取得"""
        return self.tm.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジション情報を取得"""
        return self.tm.positions.copy()

    def update_current_prices(self, prices: Dict[str, Decimal]) -> None:
        """現在価格を更新"""
        for symbol, price in prices.items():
            if symbol in self.tm.positions:
                self.tm.positions[symbol].current_price = price

    def _get_earliest_buy_date(self, symbol: str) -> datetime:
        """
        最も古い買い取引の日付を取得（FIFO会計原則・最適化版）

        買いロットキューから直接取得することで効率化
        """
        if symbol not in self.tm.positions:
            self.logger.warning(f"銘柄 {symbol} のポジションが見つかりません")
            return datetime.now()

        position = self.tm.positions[symbol]

        # 買いロットキューが空でない場合、最古のロット（先頭）の日付を返す
        if position.buy_lots:
            earliest_date = position.buy_lots[0].timestamp
            self.logger.debug(
                f"銘柄 {symbol} の最早買い取引日（ロットキューから取得）: {earliest_date}"
            )
            return earliest_date

        # フォールバック: 全取引から検索（互換性維持）
        buy_trades = [
            trade
            for trade in self.tm.trades
            if trade.symbol == symbol and trade.trade_type == TradeType.BUY
        ]

        if not buy_trades:
            self.logger.warning(f"銘柄 {symbol} の買い取引が見つかりません")
            return datetime.now()

        # 最早の取引を効率的に検索
        earliest_date = min(trade.timestamp for trade in buy_trades)
        self.logger.debug(
            f"銘柄 {symbol} の最早買い取引日（フォールバック検索）: {earliest_date}"
        )
        return earliest_date

    def _load_trades_from_db(self) -> None:
        """データベースから取引履歴を読み込み（トランザクション保護版）"""
        load_logger = self.logger
        load_logger.info("データベースから取引履歴読み込み開始")

        try:
            # トランザクション内で一括処理
            with db_manager.transaction_scope() as session:
                # データベースから全取引を取得
                db_trades = (
                    session.query(DBTrade).order_by(DBTrade.trade_datetime).all()
                )

                load_logger.info("DB取引データ取得", extra={"count": len(db_trades)})

                # メモリ内データ構造を一旦クリア（原子性保証）
                trades_backup = self.tm.trades.copy()
                positions_backup = self.tm.positions.copy()
                realized_pnl_backup = self.tm.realized_pnl.copy()
                counter_backup = self.tm._trade_counter

                try:
                    # メモリ内データクリア
                    self.tm.trades.clear()
                    self.tm.positions.clear()
                    self.tm.realized_pnl.clear()
                    self.tm._trade_counter = 0

                    for db_trade in db_trades:
                        # セッションから切り離す前に必要な属性を読み込み
                        trade_id = db_trade.id
                        stock_code = db_trade.stock_code
                        trade_type_str = db_trade.trade_type
                        quantity = db_trade.quantity
                        price = db_trade.price
                        trade_datetime = db_trade.trade_datetime
                        commission = db_trade.commission or Decimal("0")
                        memo = db_trade.memo or ""

                        # メモリ内形式に変換
                        if isinstance(trade_type_str, TradeType):
                            trade_type = trade_type_str
                        else:
                            trade_type_str_upper = str(trade_type_str).upper()
                            trade_type = (
                                TradeType.BUY
                                if trade_type_str_upper in ["BUY", "buy"]
                                else TradeType.SELL
                            )

                        memory_trade = Trade(
                            id=f"DB_{trade_id}",  # DBから読み込んだことを示すプレフィックス
                            symbol=stock_code,
                            trade_type=trade_type,
                            quantity=quantity,
                            price=Decimal(str(price)),
                            timestamp=trade_datetime,
                            commission=Decimal(str(commission)),
                            status=TradeStatus.EXECUTED,
                            notes=memo,
                        )

                        self.tm.trades.append(memory_trade)
                        self._update_position(memory_trade)

                    # 取引カウンターを最大値+1に設定
                    if db_trades:
                        max_id = max(db_trade.id for db_trade in db_trades)
                        self.tm._trade_counter = max_id + 1

                    load_logger.info(
                        "データベース読み込み完了",
                        extra={"loaded_trades": len(db_trades)},
                        trade_counter=self.tm._trade_counter,
                    )

                except Exception as restore_error:
                    # メモリ内データの復元
                    self.tm.trades = trades_backup
                    self.tm.positions = positions_backup
                    self.tm.realized_pnl = realized_pnl_backup
                    self.tm._trade_counter = counter_backup
                    load_logger.error(
                        "読み込み処理失敗、メモリ内データを復元",
                        extra={"error": str(restore_error)},
                    )
                    raise restore_error

        except Exception as e:
            log_error_with_context(e, {"operation": "load_trades_from_db"})
            load_logger.error("データベース読み込み失敗", extra={"error": str(e)})
            raise

    def clear_all_data(self, persist_to_db: bool = True) -> None:
        """
        すべての取引データを削除（トランザクション保護）

        Args:
            persist_to_db: データベースからも削除するかどうか

        Warning:
            この操作は取引履歴、ポジション、実現損益をすべて削除します
        """
        clear_logger = self.logger.bind(
            operation="clear_all_data", persist_to_db=persist_to_db
        )
        clear_logger.warning("全データ削除処理開始")

        # メモリ内データのバックアップ
        trades_backup = self.tm.trades.copy()
        positions_backup = self.tm.positions.copy()
        realized_pnl_backup = self.tm.realized_pnl.copy()
        counter_backup = self.tm._trade_counter

        try:
            if persist_to_db:
                # データベースとメモリ両方をクリア
                with db_manager.transaction_scope() as session:
                    # データベースの取引データを削除
                    deleted_count = session.query(DBTrade).delete()
                    clear_logger.info(
                        "データベース取引データ削除",
                        extra={"deleted_count": deleted_count},
                    )

                    # メモリ内データクリア
                    self.tm.trades.clear()
                    self.tm.positions.clear()
                    self.tm.realized_pnl.clear()
                    self.tm._trade_counter = 0

                    log_business_event(
                        "all_data_cleared",
                        deleted_db_records=deleted_count,
                        persisted=True,
                    )

                    clear_logger.warning("全データ削除完了（DB + メモリ）")
            else:
                # メモリ内のみクリア
                self.tm.trades.clear()
                self.tm.positions.clear()
                self.tm.realized_pnl.clear()
                self.tm._trade_counter = 0

                log_business_event(
                    "all_data_cleared", deleted_db_records=0, persisted=False
                )

                clear_logger.warning("全データ削除完了（メモリのみ）")

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.tm.trades = trades_backup
            self.tm.positions = positions_backup
            self.tm.realized_pnl = realized_pnl_backup
            self.tm._trade_counter = counter_backup

            log_error_with_context(
                e, {"operation": "clear_all_data", "persist_to_db": persist_to_db}
            )
            clear_logger.error(
                "全データ削除失敗、メモリ内データを復元", extra={"error": str(e)}
            )
            raise
