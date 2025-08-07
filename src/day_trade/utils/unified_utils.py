"""
統合ユーティリティパッケージ

分散していたユーティリティ機能を統合し、
共通の便利関数とヘルパーを提供します。

統合される機能:
- 各モジュールで重複していた共通ユーティリティ
- エラーハンドリングとロギング支援
- パフォーマンス最適化ヘルパー
- データ変換とバリデーション機能
"""

import functools
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from ..config.unified_config import get_unified_config_manager
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="unified_utils")

T = TypeVar("T")


def safe_execute(
    func: Callable[..., T],
    *args,
    default_value: T = None,
    log_errors: bool = True,
    **kwargs,
) -> T:
    """
    関数を安全に実行し、エラー時にデフォルト値を返す

    Args:
        func: 実行する関数
        *args: 関数の位置引数
        default_value: エラー時のデフォルト値
        log_errors: エラーをログに記録するか
        **kwargs: 関数のキーワード引数

    Returns:
        関数の戻り値またはデフォルト値
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(
                f"Safe execution failed for {func.__name__}",
                error=str(e),
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )
        return default_value


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    失敗時にリトライするデコレータ

    Args:
        max_retries: 最大リトライ回数
        delay: 初期遅延時間（秒）
        backoff_factor: バックオフ係数
        exceptions: リトライ対象の例外タプル
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__}",
                            error=str(e),
                            delay=current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}",
                            error=str(e),
                            total_attempts=max_retries + 1,
                        )

            raise last_exception

        return wrapper

    return decorator


def timing_decorator(log_threshold: float = 1.0):
    """
    実行時間を測定するデコレータ

    Args:
        log_threshold: ログを出力する閾値時間（秒）
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                if execution_time >= log_threshold:
                    logger.info(
                        f"Function {func.__name__} took {execution_time:.3f}s",
                        execution_time=execution_time,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys()),
                    )

        return wrapper

    return decorator


def batch_process(
    items: List[Any],
    process_func: Callable[[Any], T],
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
) -> List[T]:
    """
    アイテムを並列バッチ処理する

    Args:
        items: 処理するアイテムのリスト
        process_func: 各アイテムに適用する処理関数
        batch_size: バッチサイズ（Noneの場合は設定から取得）
        max_workers: 最大ワーカー数（Noneの場合は設定から取得）
        timeout: タイムアウト時間（秒）

    Returns:
        処理結果のリスト
    """
    if not items:
        return []

    # 設定から値を取得
    config_manager = get_unified_config_manager()
    perf_config = config_manager.get_performance_config()

    if batch_size is None:
        batch_size = perf_config.batch_size

    if max_workers is None:
        max_workers = perf_config.max_workers

    # シングルスレッド処理の場合
    if max_workers <= 1 or len(items) <= batch_size:
        return [safe_execute(process_func, item) for item in items]

    # 並列処理
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # バッチに分割して並列実行
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        future_to_batch = {
            executor.submit(_process_batch, batch, process_func): batch
            for batch in batches
        }

        for future in as_completed(future_to_batch, timeout=timeout):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # 失敗したバッチは空の結果で埋める
                batch = future_to_batch[future]
                results.extend([None] * len(batch))

    return results


def _process_batch(batch: List[Any], process_func: Callable[[Any], T]) -> List[T]:
    """バッチを処理する内部関数"""
    return [safe_execute(process_func, item) for item in batch]


def generate_unique_id(prefix: str = "", length: int = 8) -> str:
    """
    ユニークIDを生成する

    Args:
        prefix: ID のプレフィックス
        length: ランダム部分の長さ

    Returns:
        ユニークID文字列
    """
    timestamp = int(time.time() * 1000000)  # マイクロ秒
    random_part = hashlib.sha256(str(timestamp).encode()).hexdigest()[:length]

    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    else:
        return f"{timestamp}_{random_part}"


def validate_data_structure(
    data: Any,
    required_fields: Optional[List[str]] = None,
    field_types: Optional[Dict[str, type]] = None,
) -> tuple[bool, List[str]]:
    """
    データ構造をバリデーションする

    Args:
        data: バリデーション対象のデータ
        required_fields: 必須フィールドのリスト
        field_types: フィールドタイプの辞書

    Returns:
        (バリデーション成功フラグ, エラーメッセージリスト)
    """
    errors = []

    if not isinstance(data, dict):
        errors.append(f"Data must be a dictionary, got {type(data).__name__}")
        return False, errors

    # 必須フィールドチェック
    if required_fields:
        for field in required_fields:
            if field not in data:
                errors.append(f"Required field '{field}' is missing")

    # フィールドタイプチェック
    if field_types:
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                errors.append(
                    f"Field '{field}' should be {expected_type.__name__}, "
                    f"got {type(data[field]).__name__}"
                )

    return len(errors) == 0, errors


def convert_to_safe_dict(obj: Any, max_depth: int = 5) -> Dict[str, Any]:
    """
    オブジェクトを安全な辞書形式に変換する

    Args:
        obj: 変換するオブジェクト
        max_depth: 最大再帰深度

    Returns:
        辞書形式のデータ
    """
    if max_depth <= 0:
        return {"__truncated__": str(type(obj).__name__)}

    if obj is None:
        return {"value": None, "type": "NoneType"}

    if isinstance(obj, (str, int, float, bool)):
        return {"value": obj, "type": type(obj).__name__}

    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "length": len(obj),
            "items": [
                convert_to_safe_dict(item, max_depth - 1) for item in obj[:10]
            ],  # 最初の10個のみ
        }

    if isinstance(obj, dict):
        safe_dict = {"type": "dict", "keys": list(obj.keys())[:20]}  # 最初の20キーのみ
        for key, value in list(obj.items())[:10]:  # 最初の10項目のみ
            safe_key = str(key)
            safe_dict[safe_key] = convert_to_safe_dict(value, max_depth - 1)
        return safe_dict

    if hasattr(obj, "__dict__"):
        return {
            "type": type(obj).__name__,
            "module": getattr(type(obj), "__module__", "unknown"),
            "attributes": convert_to_safe_dict(obj.__dict__, max_depth - 1),
        }

    return {
        "type": type(obj).__name__,
        "value": str(obj),
        "repr": repr(obj)[:200],  # 200文字まで
    }


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    ディレクトリの存在を保証する

    Args:
        path: ディレクトリパス

    Returns:
        Path オブジェクト
    """
    path_obj = Path(path)
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def clean_string(
    text: str, max_length: int = 1000, remove_special: bool = False
) -> str:
    """
    文字列をクリーンアップする

    Args:
        text: クリーンアップする文字列
        max_length: 最大文字数
        remove_special: 特殊文字を削除するか

    Returns:
        クリーンアップされた文字列
    """
    if not isinstance(text, str):
        text = str(text)

    # 制御文字を削除
    cleaned = "".join(char for char in text if ord(char) >= 32 or char in "\t\n\r")

    # 特殊文字削除
    if remove_special:
        import re

        cleaned = re.sub(r'[^\w\s\-\.,!?()[\]{}"]', "", cleaned)

    # 長さ制限
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 3] + "..."

    return cleaned.strip()


def merge_dictionaries(*dicts: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    複数の辞書をマージする

    Args:
        *dicts: マージする辞書
        deep: 深いマージを行うか

    Returns:
        マージされた辞書
    """
    if not dicts:
        return {}

    result = {}

    for d in dicts:
        if not isinstance(d, dict):
            continue

        for key, value in d.items():
            if (
                deep
                and key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_dictionaries(result[key], value, deep=True)
            else:
                result[key] = value

    return result


def calculate_time_difference(
    start: Union[datetime, str], end: Optional[Union[datetime, str]] = None
) -> Dict[str, Union[int, float]]:
    """
    時間差を計算する

    Args:
        start: 開始時刻
        end: 終了時刻（Noneの場合は現在時刻）

    Returns:
        時間差の詳細辞書
    """
    if isinstance(start, str):
        start = datetime.fromisoformat(start)

    if end is None:
        end = datetime.now()
    elif isinstance(end, str):
        end = datetime.fromisoformat(end)

    diff = end - start
    total_seconds = diff.total_seconds()

    return {
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "total_hours": total_seconds / 3600,
        "total_days": diff.days,
        "days": diff.days,
        "seconds": diff.seconds,
        "microseconds": diff.microseconds,
        "human_readable": _format_time_difference(total_seconds),
    }


def _format_time_difference(total_seconds: float) -> str:
    """時間差を人間が読みやすい形式にフォーマット"""
    if total_seconds < 60:
        return f"{total_seconds:.1f}秒"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        return f"{minutes:.1f}分"
    elif total_seconds < 86400:
        hours = total_seconds / 3600
        return f"{hours:.1f}時間"
    else:
        days = total_seconds / 86400
        return f"{days:.1f}日"


class ThreadSafeCounter:
    """スレッドセーフなカウンター"""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """カウンターを増加させる"""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """カウンターを減少させる"""
        with self._lock:
            self._value -= amount
            return self._value

    def get(self) -> int:
        """現在の値を取得する"""
        with self._lock:
            return self._value

    def reset(self, value: int = 0) -> int:
        """カウンターをリセットする"""
        with self._lock:
            old_value = self._value
            self._value = value
            return old_value


class CircuitBreaker:
    """簡易サーキットブレーカー実装"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """サーキットブレーカー経由で関数を呼び出す"""
        with self._lock:
            if self._state == "OPEN":
                if self._should_attempt_reset():
                    self._state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """リセットを試みるべきかどうか"""
        return (
            self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        """成功時の処理"""
        with self._lock:
            self._failure_count = 0
            if self._state == "HALF_OPEN":
                self._state = "CLOSED"

    def _on_failure(self):
        """失敗時の処理"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"

    @property
    def state(self) -> str:
        """現在の状態を取得"""
        return self._state

    @property
    def failure_count(self) -> int:
        """失敗回数を取得"""
        return self._failure_count
