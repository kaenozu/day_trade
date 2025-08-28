#!/usr/bin/env python3
"""
Immutable Data Structures for Trading Systems
取引システム向け不変データ構造
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar, Iterator, Optional, Union, Dict, List, Tuple, Hashable
from dataclasses import dataclass, field
from functools import reduce
import copy
from decimal import Decimal
from datetime import datetime

T = TypeVar('T')
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class Persistent(ABC, Generic[T]):
    """永続データ構造基底クラス"""
    
    @abstractmethod
    def size(self) -> int:
        """サイズ取得"""
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        """空判定"""
        pass
    
    @abstractmethod
    def to_list(self) -> List[T]:
        """リスト変換"""
        pass


class ImmutableList(Persistent[T]):
    """不変リスト実装"""
    
    def __init__(self, items: List[T] = None):
        self._items = tuple(items) if items else tuple()
    
    @classmethod
    def empty(cls) -> 'ImmutableList[T]':
        """空リスト作成"""
        return cls([])
    
    @classmethod
    def of(cls, *items: T) -> 'ImmutableList[T]':
        """要素からリスト作成"""
        return cls(list(items))
    
    @classmethod
    def from_iterable(cls, iterable) -> 'ImmutableList[T]':
        """イテラブルからリスト作成"""
        return cls(list(iterable))
    
    def size(self) -> int:
        """サイズ取得"""
        return len(self._items)
    
    def is_empty(self) -> bool:
        """空判定"""
        return len(self._items) == 0
    
    def head(self) -> T:
        """先頭要素取得"""
        if self.is_empty():
            raise IndexError("head of empty list")
        return self._items[0]
    
    def tail(self) -> 'ImmutableList[T]':
        """末尾リスト取得"""
        if self.is_empty():
            return ImmutableList.empty()
        return ImmutableList(list(self._items[1:]))
    
    def cons(self, item: T) -> 'ImmutableList[T]':
        """先頭に要素追加"""
        return ImmutableList([item] + list(self._items))
    
    def append(self, item: T) -> 'ImmutableList[T]':
        """末尾に要素追加"""
        return ImmutableList(list(self._items) + [item])
    
    def prepend(self, item: T) -> 'ImmutableList[T]':
        """先頭に要素追加（consのエイリアス）"""
        return self.cons(item)
    
    def concat(self, other: 'ImmutableList[T]') -> 'ImmutableList[T]':
        """リスト結合"""
        return ImmutableList(list(self._items) + list(other._items))
    
    def map(self, func: Callable[[T], V]) -> 'ImmutableList[V]':
        """マップ"""
        return ImmutableList([func(item) for item in self._items])
    
    def filter(self, predicate: Callable[[T], bool]) -> 'ImmutableList[T]':
        """フィルター"""
        return ImmutableList([item for item in self._items if predicate(item)])
    
    def fold_left(self, initial: V, func: Callable[[V, T], V]) -> V:
        """左畳み込み"""
        return reduce(func, self._items, initial)
    
    def fold_right(self, initial: V, func: Callable[[T, V], V]) -> V:
        """右畳み込み"""
        result = initial
        for item in reversed(self._items):
            result = func(item, result)
        return result
    
    def reduce(self, func: Callable[[T, T], T]) -> T:
        """リデュース"""
        if self.is_empty():
            raise ValueError("reduce of empty list")
        return reduce(func, self._items)
    
    def take(self, n: int) -> 'ImmutableList[T]':
        """先頭n個取得"""
        return ImmutableList(list(self._items[:n]))
    
    def drop(self, n: int) -> 'ImmutableList[T]':
        """先頭n個削除"""
        return ImmutableList(list(self._items[n:]))
    
    def reverse(self) -> 'ImmutableList[T]':
        """逆順"""
        return ImmutableList(list(reversed(self._items)))
    
    def sort(self, key: Callable[[T], Any] = None, reverse: bool = False) -> 'ImmutableList[T]':
        """ソート"""
        sorted_items = sorted(self._items, key=key, reverse=reverse)
        return ImmutableList(sorted_items)
    
    def find(self, predicate: Callable[[T], bool]) -> Optional[T]:
        """検索"""
        for item in self._items:
            if predicate(item):
                return item
        return None
    
    def exists(self, predicate: Callable[[T], bool]) -> bool:
        """存在確認"""
        return any(predicate(item) for item in self._items)
    
    def for_all(self, predicate: Callable[[T], bool]) -> bool:
        """全要素条件確認"""
        return all(predicate(item) for item in self._items)
    
    def get(self, index: int) -> T:
        """インデックスアクセス"""
        return self._items[index]
    
    def updated(self, index: int, value: T) -> 'ImmutableList[T]':
        """要素更新"""
        if index < 0 or index >= len(self._items):
            raise IndexError("index out of bounds")
        
        new_items = list(self._items)
        new_items[index] = value
        return ImmutableList(new_items)
    
    def inserted(self, index: int, value: T) -> 'ImmutableList[T]':
        """要素挿入"""
        new_items = list(self._items)
        new_items.insert(index, value)
        return ImmutableList(new_items)
    
    def removed(self, index: int) -> 'ImmutableList[T]':
        """要素削除"""
        if index < 0 or index >= len(self._items):
            raise IndexError("index out of bounds")
        
        new_items = list(self._items)
        del new_items[index]
        return ImmutableList(new_items)
    
    def slice(self, start: int, end: int) -> 'ImmutableList[T]':
        """スライス"""
        return ImmutableList(list(self._items[start:end]))
    
    def to_list(self) -> List[T]:
        """リスト変換"""
        return list(self._items)
    
    def to_tuple(self) -> Tuple[T, ...]:
        """タプル変換"""
        return self._items
    
    def __iter__(self) -> Iterator[T]:
        """イテレーター"""
        return iter(self._items)
    
    def __len__(self) -> int:
        """長さ"""
        return len(self._items)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[T, 'ImmutableList[T]']:
        """インデックスアクセス"""
        if isinstance(index, slice):
            return ImmutableList(list(self._items[index]))
        return self._items[index]
    
    def __add__(self, other: 'ImmutableList[T]') -> 'ImmutableList[T]':
        """加算（結合）"""
        return self.concat(other)
    
    def __eq__(self, other) -> bool:
        """等価判定"""
        if not isinstance(other, ImmutableList):
            return False
        return self._items == other._items
    
    def __hash__(self) -> int:
        """ハッシュ値"""
        return hash(self._items)
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"ImmutableList({list(self._items)})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ImmutableDict(Persistent[V]):
    """不変辞書実装"""
    
    def __init__(self, items: Dict[K, V] = None):
        self._items = dict(items) if items else {}
    
    @classmethod
    def empty(cls) -> 'ImmutableDict[K, V]':
        """空辞書作成"""
        return cls({})
    
    @classmethod
    def of(cls, **kwargs) -> 'ImmutableDict[str, Any]':
        """キーワード引数から辞書作成"""
        return cls(kwargs)
    
    @classmethod
    def from_pairs(cls, pairs: List[Tuple[K, V]]) -> 'ImmutableDict[K, V]':
        """ペアリストから辞書作成"""
        return cls(dict(pairs))
    
    def size(self) -> int:
        """サイズ取得"""
        return len(self._items)
    
    def is_empty(self) -> bool:
        """空判定"""
        return len(self._items) == 0
    
    def contains_key(self, key: K) -> bool:
        """キー存在確認"""
        return key in self._items
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """値取得"""
        return self._items.get(key, default)
    
    def get_or_else(self, key: K, default: V) -> V:
        """値取得またはデフォルト"""
        return self._items.get(key, default)
    
    def updated(self, key: K, value: V) -> 'ImmutableDict[K, V]':
        """値更新"""
        new_items = self._items.copy()
        new_items[key] = value
        return ImmutableDict(new_items)
    
    def removed(self, key: K) -> 'ImmutableDict[K, V]':
        """キー削除"""
        if key not in self._items:
            return self
        
        new_items = self._items.copy()
        del new_items[key]
        return ImmutableDict(new_items)
    
    def merge(self, other: 'ImmutableDict[K, V]') -> 'ImmutableDict[K, V]':
        """辞書マージ"""
        new_items = self._items.copy()
        new_items.update(other._items)
        return ImmutableDict(new_items)
    
    def keys(self) -> ImmutableList[K]:
        """キーリスト取得"""
        return ImmutableList(list(self._items.keys()))
    
    def values(self) -> ImmutableList[V]:
        """値リスト取得"""
        return ImmutableList(list(self._items.values()))
    
    def items(self) -> ImmutableList[Tuple[K, V]]:
        """アイテムリスト取得"""
        return ImmutableList(list(self._items.items()))
    
    def map_values(self, func: Callable[[V], T]) -> 'ImmutableDict[K, T]':
        """値マップ"""
        new_items = {k: func(v) for k, v in self._items.items()}
        return ImmutableDict(new_items)
    
    def filter(self, predicate: Callable[[Tuple[K, V]], bool]) -> 'ImmutableDict[K, V]':
        """フィルター"""
        new_items = {k: v for k, v in self._items.items() if predicate((k, v))}
        return ImmutableDict(new_items)
    
    def to_list(self) -> List[Tuple[K, V]]:
        """リスト変換"""
        return list(self._items.items())
    
    def to_dict(self) -> Dict[K, V]:
        """辞書変換"""
        return self._items.copy()
    
    def __getitem__(self, key: K) -> V:
        """インデックスアクセス"""
        return self._items[key]
    
    def __contains__(self, key: K) -> bool:
        """包含確認"""
        return key in self._items
    
    def __iter__(self) -> Iterator[K]:
        """イテレーター（キー）"""
        return iter(self._items)
    
    def __len__(self) -> int:
        """長さ"""
        return len(self._items)
    
    def __eq__(self, other) -> bool:
        """等価判定"""
        if not isinstance(other, ImmutableDict):
            return False
        return self._items == other._items
    
    def __hash__(self) -> int:
        """ハッシュ値"""
        return hash(tuple(sorted(self._items.items())))
    
    def __str__(self) -> str:
        return f"ImmutableDict({self._items})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class FrozenRecord:
    """不変レコード基底クラス"""
    
    def updated(self, **kwargs) -> 'FrozenRecord':
        """フィールド更新"""
        current_values = {}
        for field_info in self.__dataclass_fields__.values():
            current_values[field_info.name] = getattr(self, field_info.name)
        
        current_values.update(kwargs)
        return self.__class__(**current_values)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書変換"""
        result = {}
        for field_info in self.__dataclass_fields__.values():
            result[field_info.name] = getattr(self, field_info.name)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrozenRecord':
        """辞書から作成"""
        return cls(**data)


# 取引システム特化型不変データ構造
@dataclass(frozen=True)
class ImmutableTrade(FrozenRecord):
    """不変取引記録"""
    trade_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal('0')
    execution_id: Optional[str] = None
    
    def with_execution(self, execution_id: str, executed_price: Decimal) -> 'ImmutableTrade':
        """約定情報付き取引作成"""
        return self.updated(
            execution_id=execution_id,
            price=executed_price
        )
    
    def total_value(self) -> Decimal:
        """取引総額"""
        return self.price * Decimal(self.quantity)
    
    def net_value(self) -> Decimal:
        """手数料差引後金額"""
        gross = self.total_value()
        return gross - self.commission if self.side == 'SELL' else gross + self.commission


@dataclass(frozen=True)
class ImmutablePosition(FrozenRecord):
    """不変ポジション"""
    symbol: str
    quantity: int
    average_price: Decimal
    current_price: Decimal
    last_updated: datetime
    realized_pnl: Decimal = Decimal('0')
    
    def market_value(self) -> Decimal:
        """時価総額"""
        return self.current_price * Decimal(abs(self.quantity))
    
    def unrealized_pnl(self) -> Decimal:
        """含み損益"""
        if self.quantity == 0:
            return Decimal('0')
        
        cost_basis = self.average_price * Decimal(abs(self.quantity))
        market_value = self.market_value()
        
        pnl = market_value - cost_basis
        return pnl if self.quantity > 0 else -pnl
    
    def with_price_update(self, new_price: Decimal) -> 'ImmutablePosition':
        """価格更新"""
        return self.updated(
            current_price=new_price,
            last_updated=datetime.utcnow()
        )
    
    def with_trade(self, trade: ImmutableTrade) -> 'ImmutablePosition':
        """取引反映"""
        if trade.symbol != self.symbol:
            raise ValueError("Symbol mismatch")
        
        trade_qty = trade.quantity if trade.side == 'BUY' else -trade.quantity
        new_quantity = self.quantity + trade_qty
        
        if new_quantity == 0:
            # ポジションクローズ
            realized = self.unrealized_pnl()
            return self.updated(
                quantity=0,
                average_price=Decimal('0'),
                realized_pnl=self.realized_pnl + realized,
                last_updated=trade.timestamp
            )
        else:
            # ポジション更新
            if (self.quantity >= 0 and trade_qty > 0) or (self.quantity < 0 and trade_qty < 0):
                # 同方向取引 - 平均価格更新
                total_cost = (self.average_price * Decimal(abs(self.quantity)) + 
                             trade.price * Decimal(abs(trade_qty)))
                total_qty = Decimal(abs(self.quantity) + abs(trade_qty))
                new_avg_price = total_cost / total_qty
            else:
                # 反対方向取引 - 部分決済
                close_qty = min(abs(self.quantity), abs(trade_qty))
                realized = (trade.price - self.average_price) * Decimal(close_qty)
                if self.quantity < 0:
                    realized = -realized
                
                new_avg_price = self.average_price
                return self.updated(
                    quantity=new_quantity,
                    average_price=new_avg_price,
                    realized_pnl=self.realized_pnl + realized,
                    last_updated=trade.timestamp
                )
            
            return self.updated(
                quantity=new_quantity,
                average_price=new_avg_price,
                last_updated=trade.timestamp
            )


@dataclass(frozen=True)
class ImmutablePortfolio(FrozenRecord):
    """不変ポートフォリオ"""
    portfolio_id: str
    cash_balance: Decimal
    positions: ImmutableDict[str, ImmutablePosition]
    trade_history: ImmutableList[ImmutableTrade]
    created_at: datetime
    last_updated: datetime
    
    @classmethod
    def create(cls, portfolio_id: str, initial_cash: Decimal) -> 'ImmutablePortfolio':
        """ポートフォリオ作成"""
        now = datetime.utcnow()
        return cls(
            portfolio_id=portfolio_id,
            cash_balance=initial_cash,
            positions=ImmutableDict.empty(),
            trade_history=ImmutableList.empty(),
            created_at=now,
            last_updated=now
        )
    
    def total_value(self) -> Decimal:
        """総資産価値"""
        positions_value = self.positions.values().fold_left(
            Decimal('0'),
            lambda acc, pos: acc + pos.market_value()
        )
        return self.cash_balance + positions_value
    
    def total_pnl(self) -> Decimal:
        """総損益"""
        return self.positions.values().fold_left(
            Decimal('0'),
            lambda acc, pos: acc + pos.unrealized_pnl() + pos.realized_pnl
        )
    
    def with_trade(self, trade: ImmutableTrade) -> 'ImmutablePortfolio':
        """取引実行"""
        # 資金チェック
        if trade.side == 'BUY':
            required_cash = trade.total_value() + trade.commission
            if required_cash > self.cash_balance:
                raise ValueError("Insufficient funds")
        
        # 現金更新
        cash_change = trade.commission  # 手数料は常に支払い
        if trade.side == 'BUY':
            cash_change += trade.total_value()  # 購入代金支払い
        else:
            cash_change -= trade.total_value()  # 売却代金受取り
        
        new_cash = self.cash_balance - cash_change if trade.side == 'BUY' else self.cash_balance + cash_change - trade.commission
        
        # ポジション更新
        current_position = self.positions.get(trade.symbol)
        if current_position:
            new_position = current_position.with_trade(trade)
        else:
            # 新規ポジション
            quantity = trade.quantity if trade.side == 'BUY' else -trade.quantity
            new_position = ImmutablePosition(
                symbol=trade.symbol,
                quantity=quantity,
                average_price=trade.price,
                current_price=trade.price,
                last_updated=trade.timestamp,
                realized_pnl=Decimal('0')
            )
        
        new_positions = self.positions.updated(trade.symbol, new_position)
        
        return self.updated(
            cash_balance=new_cash,
            positions=new_positions,
            trade_history=self.trade_history.append(trade),
            last_updated=trade.timestamp
        )
    
    def with_price_updates(self, prices: Dict[str, Decimal]) -> 'ImmutablePortfolio':
        """価格一括更新"""
        new_positions = self.positions
        
        for symbol, price in prices.items():
            if self.positions.contains_key(symbol):
                current_pos = self.positions.get(symbol)
                updated_pos = current_pos.with_price_update(price)
                new_positions = new_positions.updated(symbol, updated_pos)
        
        return self.updated(
            positions=new_positions,
            last_updated=datetime.utcnow()
        )