"""
取引実行機能

trade_manager.py からのリファクタリング抽出
取引実行、ポジション更新、FIFO売却処理
"""

from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from .trade_manager_core import TradeManagerCore
from ..models.trade_models import Trade, Position, BuyLot, RealizedPnL, TradeStatus
from ..models.trade_utils import safe_decimal_conversion, validate_positive_decimal, mask_sensitive_info
from ...models.database import db_manager
from ...models.enums import TradeType
from ...models.stock import Stock, Trade as DBTrade
from ...utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)


class TradeManagerExecution(TradeManagerCore):
    """
    取引実行機能
    
    取引の実行、ポジション更新、FIFO会計処理を担当
    TradeManagerCore を継承して実行機能を追加
    """

    def add_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        commission: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> str:
        """
        取引を追加（データベース永続化対応）

        Args:
            symbol: 銘柄コード
            trade_type: 取引タイプ
            quantity: 数量
            price: 価格
            timestamp: 取引日時
            commission: 手数料（Noneの場合は自動計算）
            notes: メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引ID
        """
        try:
            # 安全なDecimal変換
            safe_price = safe_decimal_conversion(price, "取引価格")
            validate_positive_decimal(safe_price, "取引価格")

            # 数量検証
            if not isinstance(quantity, int) or quantity <= 0:
                raise ValueError(f"数量は正の整数である必要があります: {quantity}")

            # コンテキスト情報をログに含める（価格情報はマスク）
            context_info = {
                "operation": "add_trade",
                "symbol": symbol,
                "trade_type": trade_type.value,
                "quantity": quantity,
                "price_masked": mask_sensitive_info(str(safe_price)),
                "persist_to_db": persist_to_db,
            }

            self.logger.info("取引追加処理開始", extra=context_info)

            if timestamp is None:
                timestamp = datetime.now()

            if commission is None:
                commission = self._calculate_commission(safe_price, quantity)
            else:
                commission = safe_decimal_conversion(commission, "手数料")
                validate_positive_decimal(commission, "手数料", allow_zero=True)

            # メモリ内データのバックアップ（原子性保証）
            trades_backup = self.trades.copy()
            positions_backup = self.positions.copy()
            realized_pnl_backup = self.realized_pnl.copy()
            counter_backup = self._trade_counter

            try:
                if persist_to_db:
                    # データベース永続化
                    with db_manager.transaction_scope() as session:
                        db_trade = DBTrade(
                            stock_code=symbol,
                            trade_type=trade_type,
                            quantity=quantity,
                            price=safe_price,
                            commission=commission,
                            trade_datetime=timestamp,
                            memo=notes,
                        )
                        session.add(db_trade)
                        session.flush()  # IDを取得

                        # メモリ内取引記録作成
                        trade_id = f"DB_{db_trade.id}"
                        memory_trade = Trade(
                            id=trade_id,
                            symbol=symbol,
                            trade_type=trade_type,
                            quantity=quantity,
                            price=safe_price,
                            timestamp=timestamp,
                            commission=commission,
                            status=TradeStatus.EXECUTED,
                            notes=notes,
                        )

                        self.trades.append(memory_trade)
                        self._update_position(memory_trade)
                        session.commit()

                        self.logger.info(
                            "取引追加完了（DB永続化）",
                            extra={
                                "trade_id": trade_id,
                                "db_id": db_trade.id,
                            },
                        )
                else:
                    # メモリ内のみ
                    trade_id = self._generate_trade_id()
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=trade_type,
                        quantity=quantity,
                        price=safe_price,
                        timestamp=timestamp,
                        commission=commission,
                        status=TradeStatus.EXECUTED,
                        notes=notes,
                    )

                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)

                    self.logger.info(
                        "取引追加完了（メモリのみ）",
                        extra={"trade_id": trade_id},
                    )

                log_business_event("trade_added")

                return trade_id

            except Exception as e:
                # エラー時はメモリ内データを復元
                self.trades = trades_backup
                self.positions = positions_backup
                self.realized_pnl = realized_pnl_backup
                self._trade_counter = counter_backup
                raise e

        except Exception as e:
            log_error_with_context(e, context_info)
            self.logger.error("取引追加失敗", extra={"error": str(e)})
            raise

    def _update_position(self, trade: Trade) -> None:
        """
        ポジション更新（FIFO会計原則・強化版）

        買いロットキューを使用して厳密なFIFO管理を実現
        """
        symbol = trade.symbol

        if trade.trade_type == TradeType.BUY:
            if symbol in self.positions:
                # 既存ポジションに買いロットを追加
                position = self.positions[symbol]

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

                self.positions[symbol] = new_position

        elif trade.trade_type == TradeType.SELL:
            if symbol in self.positions:
                position = self.positions[symbol]

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
                tax = pnl_before_tax * self.tax_rate

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

            self.realized_pnl.append(realized_pnl)

            # ポジション情報を更新
            position.quantity -= total_sold_quantity
            position.total_cost -= cost_basis

            # ポジションが空になった場合は削除
            if position.quantity == 0:
                del self.positions[symbol]
            else:
                # 平均価格を再計算
                if position.quantity > 0:
                    position.average_price = position.total_cost / Decimal(position.quantity)

    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, any]:
        """
        株式買い注文実行（包括的エラーハンドリング・強化版）

        Args:
            symbol: 銘柄コード
            quantity: 数量
            price: 買い価格
            current_market_price: 現在の市場価格（時価評価用）
            notes: メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            実行結果の詳細情報
        """
        try:
            # 入力検証
            safe_price = safe_decimal_conversion(price, "買い価格")
            validate_positive_decimal(safe_price, "買い価格")

            if not isinstance(quantity, int) or quantity <= 0:
                raise ValueError(f"数量は正の整数である必要があります: {quantity}")

            safe_current_market_price = None
            if current_market_price is not None:
                safe_current_market_price = safe_decimal_conversion(
                    current_market_price, "現在価格"
                )
                validate_positive_decimal(safe_current_market_price, "現在価格")

            # 手数料計算
            commission = self._calculate_commission(safe_price, quantity)
            timestamp = datetime.now()

            # メモリ内データのバックアップ
            trades_backup = self.trades.copy()
            positions_backup = self.positions.copy()
            counter_backup = self._trade_counter

            # 取引追加
            trade_id = self.add_trade(
                symbol=symbol,
                trade_type=TradeType.BUY,
                quantity=quantity,
                price=safe_price,
                timestamp=timestamp,
                commission=commission,
                notes=notes,
                persist_to_db=persist_to_db,
            )

            # 現在価格を設定（時価評価用）
            if safe_current_market_price and symbol in self.positions:
                self.positions[symbol].current_price = safe_current_market_price

            # 結果データ作成
            result = {
                "success": True,
                "trade_id": trade_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": str(safe_price),
                "commission": str(commission),
                "timestamp": timestamp.isoformat(),
                "position": (
                    self.positions[symbol].to_dict()
                    if symbol in self.positions
                    else None
                ),
                "total_cost": str(safe_price * quantity + commission),
            }

            return result

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "buy_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": mask_sensitive_info(str(price)),
                },
            )
            raise

    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, any]:
        """
        株式売り注文実行（FIFO会計・包括的エラーハンドリング強化版）

        Args:
            symbol: 銘柄コード
            quantity: 数量
            price: 売り価格
            current_market_price: 現在の市場価格（時価評価用）
            notes: メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            実行結果の詳細情報（実現損益含む）
        """
        try:
            # 入力検証
            safe_price = safe_decimal_conversion(price, "売り価格")
            validate_positive_decimal(safe_price, "売り価格")

            if not isinstance(quantity, int) or quantity <= 0:
                raise ValueError(f"数量は正の整数である必要があります: {quantity}")

            # ポジション存在確認
            if symbol not in self.positions:
                raise ValueError(f"銘柄 '{symbol}' のポジションを保有していません")

            position = self.positions[symbol]
            if position.quantity < quantity:
                raise ValueError(
                    f"売却数量({quantity})が保有数量({position.quantity})を超えています"
                )

            safe_current_market_price = None
            if current_market_price is not None:
                safe_current_market_price = safe_decimal_conversion(
                    current_market_price, "現在価格"
                )
                validate_positive_decimal(safe_current_market_price, "現在価格")

            # 手数料計算
            commission = self._calculate_commission(safe_price, quantity)
            timestamp = datetime.now()

            # メモリ内データのバックアップ
            trades_backup = self.trades.copy()
            positions_backup = self.positions.copy()
            realized_pnl_backup = self.realized_pnl.copy()
            counter_backup = self._trade_counter

            # 売却前の実現損益件数
            initial_realized_count = len(self.realized_pnl)

            # 取引追加（FIFO処理も実行される）
            trade_id = self.add_trade(
                symbol=symbol,
                trade_type=TradeType.SELL,
                quantity=quantity,
                price=safe_price,
                timestamp=timestamp,
                commission=commission,
                notes=notes,
                persist_to_db=persist_to_db,
            )

            # 今回の売却で生成された実現損益を取得
            new_realized_pnl = self.realized_pnl[initial_realized_count:]

            # 現在価格を設定（残りポジションがある場合）
            if safe_current_market_price and symbol in self.positions:
                self.positions[symbol].current_price = safe_current_market_price

            # 結果データ作成
            result = {
                "success": True,
                "trade_id": trade_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": str(safe_price),
                "commission": str(commission),
                "timestamp": timestamp.isoformat(),
                "remaining_position": (
                    self.positions[symbol].to_dict()
                    if symbol in self.positions
                    else None
                ),
                "realized_pnl": [pnl.to_dict() for pnl in new_realized_pnl],
                "total_proceeds": str(safe_price * quantity - commission),
            }

            return result

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "sell_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": mask_sensitive_info(str(price)),
                },
            )
            raise