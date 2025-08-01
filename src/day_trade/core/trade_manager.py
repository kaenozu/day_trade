"""
取引記録管理機能
売買履歴を記録し、損益計算を行う
"""

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TradeType(Enum):
    """取引タイプ"""

    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "pending"  # 注文中
    EXECUTED = "executed"  # 約定済み
    CANCELLED = "cancelled"  # キャンセル
    PARTIAL = "partial"  # 一部約定


@dataclass
class Trade:
    """取引記録"""

    id: str
    symbol: str
    trade_type: TradeType
    quantity: int
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal("0")
    status: TradeStatus = TradeStatus.EXECUTED
    notes: str = ""

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        data = asdict(self)
        data["trade_type"] = self.trade_type.value
        data["status"] = self.status.value
        data["price"] = str(self.price)
        data["commission"] = str(self.commission)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "Trade":
        """辞書から復元"""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            trade_type=TradeType(data["trade_type"]),
            quantity=data["quantity"],
            price=Decimal(data["price"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            commission=Decimal(data["commission"]),
            status=TradeStatus(data["status"]),
            notes=data.get("notes", ""),
        )


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int
    average_price: Decimal
    total_cost: Decimal
    current_price: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        """時価総額"""
        return self.current_price * Decimal(self.quantity)

    @property
    def unrealized_pnl(self) -> Decimal:
        """含み損益"""
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """含み損益率"""
        if self.total_cost == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * 100

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": str(self.average_price),
            "total_cost": str(self.total_cost),
            "current_price": str(self.current_price),
            "market_value": str(self.market_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_percent": str(
                self.unrealized_pnl_percent.quantize(Decimal("0.01"))
            ),
        }


@dataclass
class RealizedPnL:
    """実現損益"""

    symbol: str
    quantity: int
    buy_price: Decimal
    sell_price: Decimal
    buy_commission: Decimal
    sell_commission: Decimal
    pnl: Decimal
    pnl_percent: Decimal
    buy_date: datetime
    sell_date: datetime

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "buy_price": str(self.buy_price),
            "sell_price": str(self.sell_price),
            "buy_commission": str(self.buy_commission),
            "sell_commission": str(self.sell_commission),
            "pnl": str(self.pnl),
            "pnl_percent": str(self.pnl_percent.quantize(Decimal("0.01"))),
            "buy_date": self.buy_date.isoformat(),
            "sell_date": self.sell_date.isoformat(),
        }


class TradeManager:
    """取引記録管理クラス"""

    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.001"),
        tax_rate: Decimal = Decimal("0.2"),
    ):
        """
        初期化

        Args:
            commission_rate: 手数料率（デフォルト0.1%）
            tax_rate: 税率（デフォルト20%）
        """
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: List[RealizedPnL] = []
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self._trade_counter = 0

    def _generate_trade_id(self) -> str:
        """取引IDを生成"""
        self._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}{self._trade_counter:04d}"

    def _calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """手数料を計算"""
        total_value = price * Decimal(quantity)
        commission = total_value * self.commission_rate
        # 最低100円の手数料
        return max(commission, Decimal("100"))

    def add_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        commission: Optional[Decimal] = None,
        notes: str = "",
    ) -> str:
        """
        取引を追加

        Args:
            symbol: 銘柄コード
            trade_type: 取引タイプ
            quantity: 数量
            price: 価格
            timestamp: 取引日時
            commission: 手数料（Noneの場合は自動計算）
            notes: メモ

        Returns:
            取引ID
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()

            if commission is None:
                commission = self._calculate_commission(price, quantity)

            trade_id = self._generate_trade_id()

            trade = Trade(
                id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                commission=commission,
                notes=notes,
            )

            self.trades.append(trade)
            self._update_position(trade)

            logger.info(
                f"取引追加: {trade_id} - {symbol} {trade_type.value} {quantity}株 @{price}円"
            )
            return trade_id

        except Exception as e:
            logger.error(
                f"取引の追加中に予期せぬエラーが発生しました。入力データを確認してください。詳細: {e}"
            )
            raise

    def _update_position(self, trade: Trade):
        """ポジションを更新"""
        symbol = trade.symbol

        if trade.trade_type == TradeType.BUY:
            if symbol in self.positions:
                # 既存ポジションに追加
                position = self.positions[symbol]
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
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    average_price=total_cost / Decimal(trade.quantity),
                    total_cost=total_cost,
                )

        elif trade.trade_type == TradeType.SELL:
            if symbol in self.positions:
                position = self.positions[symbol]

                if position.quantity >= trade.quantity:
                    # 実現損益を計算
                    buy_price = position.average_price
                    sell_price = trade.price

                    # 手数料を按分
                    buy_commission_per_share = (
                        position.total_cost / Decimal(position.quantity)
                        - position.average_price
                    )
                    buy_commission = buy_commission_per_share * Decimal(trade.quantity)

                    pnl_before_tax = (
                        (sell_price - buy_price) * Decimal(trade.quantity)
                        - buy_commission
                        - trade.commission
                    )

                    # 税金計算（利益が出た場合のみ）
                    tax = Decimal("0")
                    if pnl_before_tax > 0:
                        tax = pnl_before_tax * self.tax_rate

                    pnl = pnl_before_tax - tax
                    pnl_percent = (pnl / (buy_price * Decimal(trade.quantity))) * 100

                    # 実現損益を記録
                    realized_pnl = RealizedPnL(
                        symbol=symbol,
                        quantity=trade.quantity,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        buy_commission=buy_commission,
                        sell_commission=trade.commission,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        buy_date=self._get_earliest_buy_date(symbol),
                        sell_date=trade.timestamp,
                    )

                    self.realized_pnl.append(realized_pnl)

                    # ポジション更新
                    remaining_quantity = position.quantity - trade.quantity
                    if remaining_quantity > 0:
                        # 按分してコストを調整
                        remaining_ratio = Decimal(remaining_quantity) / Decimal(
                            position.quantity
                        )
                        remaining_cost = position.total_cost * remaining_ratio
                        position.quantity = remaining_quantity
                        position.total_cost = remaining_cost
                        position.average_price = remaining_cost / Decimal(
                            remaining_quantity
                        )
                    else:
                        # ポジション完全クローズ
                        del self.positions[symbol]

                else:
                    logger.warning(
                        f"銘柄 '{symbol}' の売却数量が保有数量 ({position.quantity}) を超過しています。売却数量: {trade.quantity}。取引は処理されません。"
                    )
            else:
                logger.warning(
                    f"ポジションを保有していない銘柄 '{symbol}' の売却を試みました。取引は無視されます。"
                )

    def _get_earliest_buy_date(self, symbol: str) -> datetime:
        """最も古い買い取引の日付を取得"""
        buy_trades = [
            t
            for t in self.trades
            if t.symbol == symbol and t.trade_type == TradeType.BUY
        ]
        if buy_trades:
            return min(trade.timestamp for trade in buy_trades)
        return datetime.now()

    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジション情報を取得"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジション情報を取得"""
        return self.positions.copy()

    def update_current_prices(self, prices: Dict[str, Decimal]):
        """現在価格を更新"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """取引履歴を取得"""
        if symbol:
            return [trade for trade in self.trades if trade.symbol == symbol]
        return self.trades.copy()

    def get_realized_pnl_history(
        self, symbol: Optional[str] = None
    ) -> List[RealizedPnL]:
        """実現損益履歴を取得"""
        if symbol:
            return [pnl for pnl in self.realized_pnl if pnl.symbol == symbol]
        return self.realized_pnl.copy()

    def get_portfolio_summary(self) -> Dict:
        """ポートフォリオサマリーを取得"""
        total_cost = sum(pos.total_cost for pos in self.positions.values())
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = total_market_value - total_cost
        total_realized_pnl = sum(pnl.pnl for pnl in self.realized_pnl)

        return {
            "total_positions": len(self.positions),
            "total_cost": str(total_cost),
            "total_market_value": str(total_market_value),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_realized_pnl": str(total_realized_pnl),
            "total_pnl": str(total_unrealized_pnl + total_realized_pnl),
            "total_trades": len(self.trades),
            "winning_trades": len([pnl for pnl in self.realized_pnl if pnl.pnl > 0]),
            "losing_trades": len([pnl for pnl in self.realized_pnl if pnl.pnl < 0]),
            "win_rate": (
                f"{(len([pnl for pnl in self.realized_pnl if pnl.pnl > 0]) / max(len(self.realized_pnl), 1) * 100):.1f}%"
                if self.realized_pnl
                else "0.0%"
            ),
        }

    def export_to_csv(self, filepath: str, data_type: str = "trades"):
        """
        CSVファイルにエクスポート

        Args:
            filepath: 出力ファイルパス
            data_type: データタイプ ('trades', 'positions', 'realized_pnl')
        """
        try:
            if data_type == "trades":
                data = [trade.to_dict() for trade in self.trades]
                fieldnames = [
                    "id",
                    "symbol",
                    "trade_type",
                    "quantity",
                    "price",
                    "timestamp",
                    "commission",
                    "status",
                    "notes",
                ]

            elif data_type == "positions":
                data = [pos.to_dict() for pos in self.positions.values()]
                fieldnames = [
                    "symbol",
                    "quantity",
                    "average_price",
                    "total_cost",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "unrealized_pnl_percent",
                ]

            elif data_type == "realized_pnl":
                data = [pnl.to_dict() for pnl in self.realized_pnl]
                fieldnames = [
                    "symbol",
                    "quantity",
                    "buy_price",
                    "sell_price",
                    "buy_commission",
                    "sell_commission",
                    "pnl",
                    "pnl_percent",
                    "buy_date",
                    "sell_date",
                ]

            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"CSV出力完了: {filepath} ({len(data)}件)")

        except Exception as e:
            logger.error(
                f"データのエクスポート中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}"
            )
            raise

    def save_to_json(self, filepath: str):
        """JSON形式で保存"""
        try:
            data = {
                "trades": [trade.to_dict() for trade in self.trades],
                "positions": {
                    symbol: pos.to_dict() for symbol, pos in self.positions.items()
                },
                "realized_pnl": [pnl.to_dict() for pnl in self.realized_pnl],
                "settings": {
                    "commission_rate": str(self.commission_rate),
                    "tax_rate": str(self.tax_rate),
                    "trade_counter": self._trade_counter,
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"JSON保存完了: {filepath}")

        except Exception as e:
            logger.error(
                f"データの保存中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}"
            )
            raise

    def load_from_json(self, filepath: str):
        """JSON形式から読み込み"""
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # 取引履歴復元
            self.trades = [
                Trade.from_dict(trade_data) for trade_data in data.get("trades", [])
            ]

            # ポジション復元
            self.positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.positions[symbol] = Position(
                    symbol=pos_data["symbol"],
                    quantity=pos_data["quantity"],
                    average_price=Decimal(pos_data["average_price"]),
                    total_cost=Decimal(pos_data["total_cost"]),
                    current_price=Decimal(pos_data["current_price"]),
                )

            # 実現損益復元
            self.realized_pnl = []
            for pnl_data in data.get("realized_pnl", []):
                self.realized_pnl.append(
                    RealizedPnL(
                        symbol=pnl_data["symbol"],
                        quantity=pnl_data["quantity"],
                        buy_price=Decimal(pnl_data["buy_price"]),
                        sell_price=Decimal(pnl_data["sell_price"]),
                        buy_commission=Decimal(pnl_data["buy_commission"]),
                        sell_commission=Decimal(pnl_data["sell_commission"]),
                        pnl=Decimal(pnl_data["pnl"]),
                        pnl_percent=Decimal(pnl_data["pnl_percent"]),
                        buy_date=datetime.fromisoformat(pnl_data["buy_date"]),
                        sell_date=datetime.fromisoformat(pnl_data["sell_date"]),
                    )
                )

            # 設定復元
            settings = data.get("settings", {})
            if "commission_rate" in settings:
                self.commission_rate = Decimal(settings["commission_rate"])
            if "tax_rate" in settings:
                self.tax_rate = Decimal(settings["tax_rate"])
            if "trade_counter" in settings:
                self._trade_counter = settings["trade_counter"]

            logger.info(f"JSON読み込み完了: {filepath}")

        except Exception as e:
            logger.error(
                f"データの読み込み中にエラーが発生しました。ファイル形式が正しいか、破損していないか確認してください。詳細: {e}"
            )
            raise

    def calculate_tax_implications(self, year: int) -> Dict:
        """税務計算"""
        try:
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31, 23, 59, 59)

            # 年内の実現損益
            year_pnl = [
                pnl
                for pnl in self.realized_pnl
                if year_start <= pnl.sell_date <= year_end
            ]

            total_gain = (
                sum(pnl.pnl for pnl in year_pnl if pnl.pnl > 0)
                if year_pnl
                else Decimal("0")
            )
            total_loss = (
                sum(abs(pnl.pnl) for pnl in year_pnl if pnl.pnl < 0)
                if year_pnl
                else Decimal("0")
            )
            net_gain = total_gain - total_loss

            # 税額計算
            tax_due = Decimal("0")
            if net_gain > 0:
                tax_due = net_gain * self.tax_rate

            return {
                "year": year,
                "total_trades": len(year_pnl),
                "total_gain": str(total_gain),
                "total_loss": str(total_loss),
                "net_gain": str(net_gain),
                "tax_due": str(tax_due),
                "winning_trades": len([pnl for pnl in year_pnl if pnl.pnl > 0]),
                "losing_trades": len([pnl for pnl in year_pnl if pnl.pnl < 0]),
            }

        except Exception as e:
            logger.error(
                f"税務計算中に予期せぬエラーが発生しました。入力データまたは計算ロジックを確認してください。詳細: {e}"
            )
            raise


# 使用例
if __name__ == "__main__":
    from datetime import datetime, timedelta
    from decimal import Decimal

    # 取引管理システムを初期化
    tm = TradeManager(commission_rate=Decimal("0.001"), tax_rate=Decimal("0.2"))

    # サンプル取引を追加
    base_date = datetime.now() - timedelta(days=30)

    # トヨタ株の取引例
    tm.add_trade("7203", TradeType.BUY, 100, Decimal("2500"), base_date)
    tm.add_trade(
        "7203", TradeType.BUY, 200, Decimal("2450"), base_date + timedelta(days=1)
    )

    # 現在価格を更新
    tm.update_current_prices({"7203": Decimal("2600")})

    # ポジション情報表示
    position = tm.get_position("7203")
    if position:
        print("=== ポジション情報 ===")
        print(f"銘柄: {position.symbol}")
        print(f"数量: {position.quantity}株")
        print(f"平均単価: {position.average_price}円")
        print(f"総コスト: {position.total_cost}円")
        print(f"現在価格: {position.current_price}円")
        print(f"時価総額: {position.market_value}円")
        print(
            f"含み損益: {position.unrealized_pnl}円 ({position.unrealized_pnl_percent}%)"
        )

    # 一部売却
    tm.add_trade(
        "7203", TradeType.SELL, 100, Decimal("2650"), base_date + timedelta(days=5)
    )

    # 実現損益表示
    realized_pnl = tm.get_realized_pnl_history("7203")
    if realized_pnl:
        print("\n=== 実現損益 ===")
        for pnl in realized_pnl:
            print(f"銘柄: {pnl.symbol}")
            print(f"数量: {pnl.quantity}株")
            print(f"買値: {pnl.buy_price}円")
            print(f"売値: {pnl.sell_price}円")
            print(f"損益: {pnl.pnl}円 ({pnl.pnl_percent}%)")

    # ポートフォリオサマリー
    summary = tm.get_portfolio_summary()
    print("\n=== ポートフォリオサマリー ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # CSV出力例
    print("\n=== CSV出力例 ===")
    try:
        tm.export_to_csv("trades.csv", "trades")
        tm.export_to_csv("positions.csv", "positions")
        tm.export_to_csv("realized_pnl.csv", "realized_pnl")
        print("CSV出力完了")
    except Exception as e:
        print(f"CSV出力エラー: {e}")
