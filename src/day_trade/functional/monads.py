#!/usr/bin/env python3
"""
Monadic Programming for Trading Systems
取引システム向けモナドプログラミング
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar, Union, Optional, Tuple, Dict
from dataclasses import dataclass
from functools import wraps
import traceback
from decimal import Decimal
from datetime import datetime

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
E = TypeVar('E')  # Error type


class Functor(Generic[A], ABC):
    """ファンクター抽象基底クラス"""
    
    @abstractmethod
    def map(self, func: Callable[[A], B]) -> 'Functor[B]':
        """map: (A -> B) -> F[A] -> F[B]"""
        pass
    
    def __rshift__(self, func: Callable[[A], B]) -> 'Functor[B]':
        """演算子オーバーロード: >> for map"""
        return self.map(func)


class Monad(Functor[A], ABC):
    """モナド抽象基底クラス"""
    
    @classmethod
    @abstractmethod
    def pure(cls, value: A) -> 'Monad[A]':
        """pure: A -> M[A]"""
        pass
    
    @abstractmethod
    def bind(self, func: Callable[[A], 'Monad[B]']) -> 'Monad[B]':
        """bind: M[A] -> (A -> M[B]) -> M[B]"""
        pass
    
    def __rrshift__(self, func: Callable[[A], 'Monad[B]']) -> 'Monad[B]':
        """演算子オーバーロード: >>= for bind"""
        return self.bind(func)


class Maybe(Monad[A]):
    """Maybeモナド - Null安全性"""
    
    def __init__(self, value: Optional[A] = None):
        self._value = value
    
    @classmethod
    def some(cls, value: A) -> 'Maybe[A]':
        """値ありMaybe作成"""
        if value is None:
            raise ValueError("Cannot create Some with None value")
        return cls(value)
    
    @classmethod
    def none(cls) -> 'Maybe[A]':
        """値なしMaybe作成"""
        return cls(None)
    
    @classmethod
    def pure(cls, value: A) -> 'Maybe[A]':
        """pure実装"""
        return cls.some(value) if value is not None else cls.none()
    
    def is_some(self) -> bool:
        """値の存在確認"""
        return self._value is not None
    
    def is_none(self) -> bool:
        """値の非存在確認"""
        return self._value is None
    
    def get(self) -> A:
        """値取得（unsafe）"""
        if self.is_none():
            raise ValueError("Cannot get value from None")
        return self._value
    
    def get_or_else(self, default: A) -> A:
        """値取得またはデフォルト値"""
        return self._value if self.is_some() else default
    
    def map(self, func: Callable[[A], B]) -> 'Maybe[B]':
        """map実装"""
        if self.is_none():
            return Maybe.none()
        try:
            return Maybe.some(func(self._value))
        except Exception:
            return Maybe.none()
    
    def bind(self, func: Callable[[A], 'Maybe[B]']) -> 'Maybe[B]':
        """bind実装"""
        if self.is_none():
            return Maybe.none()
        try:
            return func(self._value)
        except Exception:
            return Maybe.none()
    
    def filter(self, predicate: Callable[[A], bool]) -> 'Maybe[A]':
        """フィルター"""
        if self.is_none():
            return self
        try:
            return self if predicate(self._value) else Maybe.none()
        except Exception:
            return Maybe.none()
    
    def __str__(self) -> str:
        return f"Some({self._value})" if self.is_some() else "None"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class TradingError:
    """取引エラー型"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.utcnow())


class Either(Monad[A]):
    """Eitherモナド - エラーハンドリング"""
    
    def __init__(self, value: Union[E, A], is_right: bool):
        self._value = value
        self._is_right = is_right
    
    @classmethod
    def left(cls, error: E) -> 'Either[E, A]':
        """Left（エラー）作成"""
        return cls(error, False)
    
    @classmethod
    def right(cls, value: A) -> 'Either[E, A]':
        """Right（成功値）作成"""
        return cls(value, True)
    
    @classmethod
    def pure(cls, value: A) -> 'Either[E, A]':
        """pure実装"""
        return cls.right(value)
    
    def is_left(self) -> bool:
        """エラー判定"""
        return not self._is_right
    
    def is_right(self) -> bool:
        """成功判定"""
        return self._is_right
    
    def get_left(self) -> E:
        """エラー値取得"""
        if self.is_right():
            raise ValueError("Cannot get left value from Right")
        return self._value
    
    def get_right(self) -> A:
        """成功値取得"""
        if self.is_left():
            raise ValueError("Cannot get right value from Left")
        return self._value
    
    def get_or_else(self, default: A) -> A:
        """値取得またはデフォルト"""
        return self._value if self.is_right() else default
    
    def map(self, func: Callable[[A], B]) -> 'Either[E, B]':
        """map実装"""
        if self.is_left():
            return Either.left(self._value)
        try:
            return Either.right(func(self._value))
        except Exception as e:
            return Either.left(e)
    
    def bind(self, func: Callable[[A], 'Either[E, B]']) -> 'Either[E, B]':
        """bind実装"""
        if self.is_left():
            return Either.left(self._value)
        try:
            return func(self._value)
        except Exception as e:
            return Either.left(e)
    
    def map_left(self, func: Callable[[E], C]) -> 'Either[C, A]':
        """エラー側のマップ"""
        if self.is_right():
            return Either.right(self._value)
        try:
            return Either.left(func(self._value))
        except Exception as e:
            return Either.left(e)
    
    def fold(self, on_left: Callable[[E], C], on_right: Callable[[A], C]) -> C:
        """畳み込み"""
        if self.is_left():
            return on_left(self._value)
        else:
            return on_right(self._value)
    
    def __str__(self) -> str:
        if self.is_left():
            return f"Left({self._value})"
        else:
            return f"Right({self._value})"
    
    def __repr__(self) -> str:
        return self.__str__()


class IO(Monad[A]):
    """IOモナド - 副作用の制御"""
    
    def __init__(self, computation: Callable[[], A]):
        self._computation = computation
    
    @classmethod
    def pure(cls, value: A) -> 'IO[A]':
        """pure実装"""
        return cls(lambda: value)
    
    def map(self, func: Callable[[A], B]) -> 'IO[B]':
        """map実装"""
        def computation():
            result = self._computation()
            return func(result)
        return IO(computation)
    
    def bind(self, func: Callable[[A], 'IO[B]']) -> 'IO[B]':
        """bind実装"""
        def computation():
            result = self._computation()
            return func(result).run()
        return IO(computation)
    
    def run(self) -> A:
        """計算実行"""
        return self._computation()
    
    def __str__(self) -> str:
        return f"IO(<computation>)"


class Reader(Monad[A]):
    """Readerモナド - 環境依存計算"""
    
    def __init__(self, computation: Callable[[Any], A]):
        self._computation = computation
    
    @classmethod
    def pure(cls, value: A) -> 'Reader[A]':
        """pure実装"""
        return cls(lambda _: value)
    
    @classmethod
    def ask(cls) -> 'Reader[Any]':
        """環境取得"""
        return cls(lambda env: env)
    
    def map(self, func: Callable[[A], B]) -> 'Reader[B]':
        """map実装"""
        def computation(env):
            result = self._computation(env)
            return func(result)
        return Reader(computation)
    
    def bind(self, func: Callable[[A], 'Reader[B]']) -> 'Reader[B]':
        """bind実装"""
        def computation(env):
            result = self._computation(env)
            return func(result).run(env)
        return Reader(computation)
    
    def run(self, environment: Any) -> A:
        """環境で実行"""
        return self._computation(environment)
    
    def local(self, func: Callable[[Any], Any]) -> 'Reader[A]':
        """ローカル環境変更"""
        def computation(env):
            new_env = func(env)
            return self._computation(new_env)
        return Reader(computation)


class State(Monad[A]):
    """Stateモナド - 状態遷移"""
    
    def __init__(self, state_func: Callable[[Any], Tuple[A, Any]]):
        self._state_func = state_func
    
    @classmethod
    def pure(cls, value: A) -> 'State[A]':
        """pure実装"""
        return cls(lambda state: (value, state))
    
    @classmethod
    def get(cls) -> 'State[Any]':
        """状態取得"""
        return cls(lambda state: (state, state))
    
    @classmethod
    def put(cls, new_state: Any) -> 'State[None]':
        """状態設定"""
        return cls(lambda _: (None, new_state))
    
    @classmethod
    def modify(cls, func: Callable[[Any], Any]) -> 'State[None]':
        """状態変更"""
        return cls(lambda state: (None, func(state)))
    
    def map(self, func: Callable[[A], B]) -> 'State[B]':
        """map実装"""
        def state_func(state):
            value, new_state = self._state_func(state)
            return func(value), new_state
        return State(state_func)
    
    def bind(self, func: Callable[[A], 'State[B]']) -> 'State[B]':
        """bind実装"""
        def state_func(state):
            value, new_state = self._state_func(state)
            return func(value).run(new_state)
        return State(state_func)
    
    def run(self, initial_state: Any) -> Tuple[A, Any]:
        """状態計算実行"""
        return self._state_func(initial_state)
    
    def eval(self, initial_state: Any) -> A:
        """値のみ取得"""
        value, _ = self.run(initial_state)
        return value
    
    def exec(self, initial_state: Any) -> Any:
        """状態のみ取得"""
        _, state = self.run(initial_state)
        return state


# 取引システム特化型モナド
class TradingResult(Either[TradingError, A]):
    """取引結果モナド"""
    
    @classmethod
    def success(cls, value: A) -> 'TradingResult[A]':
        """成功結果"""
        return cls.right(value)
    
    @classmethod
    def failure(cls, code: str, message: str, details: Dict[str, Any] = None) -> 'TradingResult[A]':
        """失敗結果"""
        error = TradingError(code, message, details)
        return cls.left(error)
    
    @classmethod
    def validation_error(cls, message: str, field: str = None) -> 'TradingResult[A]':
        """バリデーションエラー"""
        details = {'field': field} if field else None
        return cls.failure('VALIDATION_ERROR', message, details)
    
    @classmethod
    def not_found(cls, resource: str) -> 'TradingResult[A]':
        """リソース未発見エラー"""
        return cls.failure('NOT_FOUND', f'{resource} not found')
    
    @classmethod
    def insufficient_funds(cls, required: Decimal, available: Decimal) -> 'TradingResult[A]':
        """資金不足エラー"""
        details = {
            'required': str(required),
            'available': str(available),
            'shortfall': str(required - available)
        }
        return cls.failure('INSUFFICIENT_FUNDS', 'Insufficient funds for trade', details)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        if self.is_right():
            return {
                'success': True,
                'data': self.get_right(),
                'error': None
            }
        else:
            error = self.get_left()
            return {
                'success': False,
                'data': None,
                'error': {
                    'code': error.code,
                    'message': error.message,
                    'details': error.details,
                    'timestamp': error.timestamp.isoformat()
                }
            }


# モナド変換子
class MaybeT(Monad[A]):
    """MaybeT変換子"""
    
    def __init__(self, wrapped: Monad[Maybe[A]]):
        self._wrapped = wrapped
    
    @classmethod
    def pure(cls, value: A) -> 'MaybeT[A]':
        """pure実装"""
        return cls(IO.pure(Maybe.some(value)))
    
    @classmethod
    def lift(cls, monad: Monad[A]) -> 'MaybeT[A]':
        """モナドリフト"""
        return cls(monad.map(Maybe.some))
    
    def map(self, func: Callable[[A], B]) -> 'MaybeT[B]':
        """map実装"""
        return MaybeT(self._wrapped.map(lambda maybe: maybe.map(func)))
    
    def bind(self, func: Callable[[A], 'MaybeT[B]']) -> 'MaybeT[B]':
        """bind実装"""
        def inner(maybe):
            if maybe.is_none():
                return IO.pure(Maybe.none())
            return func(maybe.get())._wrapped
        
        return MaybeT(self._wrapped.bind(inner))
    
    def run(self) -> Monad[Maybe[A]]:
        """内部モナド取得"""
        return self._wrapped


# ユーティリティ関数
def safe_divide(a: float, b: float) -> Maybe[float]:
    """安全な除算"""
    if b == 0:
        return Maybe.none()
    return Maybe.some(a / b)


def safe_parse_decimal(value: str) -> Maybe[Decimal]:
    """安全な小数解析"""
    try:
        return Maybe.some(Decimal(value))
    except Exception:
        return Maybe.none()


def validate_trade_quantity(quantity: int) -> TradingResult[int]:
    """取引数量バリデーション"""
    if quantity <= 0:
        return TradingResult.validation_error("Quantity must be positive", "quantity")
    if quantity > 1000000:
        return TradingResult.validation_error("Quantity too large", "quantity")
    return TradingResult.success(quantity)


def validate_trade_price(price: Decimal) -> TradingResult[Decimal]:
    """取引価格バリデーション"""
    if price <= 0:
        return TradingResult.validation_error("Price must be positive", "price")
    return TradingResult.success(price)


# 関数型装飾子
def maybe_result(func: Callable[..., A]) -> Callable[..., Maybe[A]]:
    """Maybe結果装飾子"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Maybe[A]:
        try:
            result = func(*args, **kwargs)
            return Maybe.some(result) if result is not None else Maybe.none()
        except Exception:
            return Maybe.none()
    return wrapper


def trading_result(func: Callable[..., A]) -> Callable[..., TradingResult[A]]:
    """TradingResult装飾子"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> TradingResult[A]:
        try:
            result = func(*args, **kwargs)
            return TradingResult.success(result)
        except ValueError as e:
            return TradingResult.validation_error(str(e))
        except Exception as e:
            return TradingResult.failure('EXECUTION_ERROR', str(e))
    return wrapper


def io_operation(func: Callable[..., A]) -> Callable[..., IO[A]]:
    """IO操作装飾子"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> IO[A]:
        return IO(lambda: func(*args, **kwargs))
    return wrapper