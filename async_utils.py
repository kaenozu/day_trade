import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# アプリケーション全体で共有するThreadPoolExecutor
# コンテキストマネージャーを使わないため、シャットダウンは手動管理が必要だが、
# このユーティリティのライフサイクルがアプリケーションとほぼ同じであると仮定する。
_executor = ThreadPoolExecutor()

class AsyncUtils:
    """非同期処理ユーティリティ"""

    @staticmethod
    async def run_in_thread(func, *args, **kwargs):
        """
        別スレッドで同期関数を実行します。
        Python 3.9以上の場合はasyncio.to_threadを使用し、
        それ以前のバージョンでは共有のThreadPoolExecutorにフォールバックします。
        """
        try:
            # Python 3.9+ではネイティブサポートされている `to_thread` を使用
            return await asyncio.to_thread(func, *args, **kwargs)
        except AttributeError:
            # Python 3.9未満のフォールバック
            logger.warning("asyncio.to_thread is not available. Falling back to a shared ThreadPoolExecutor.")
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_executor, func, *args, **kwargs)

    @staticmethod
    def safe_create_task(coro):
        """
        安全にバックグラウンドタスクを作成します。

        作成されたタスクは呼び出し元で参照を保持し、適切に管理する必要があります。
        (例: タスクのリストに追加して後でawaitする、など)
        実行中のイベントループがない場合、この関数はRuntimeErrorを送出します。
        """
        return asyncio.create_task(coro)

# --- 使用例 ---
#
# import time
#
# def blocking_io_operation(duration):
#     """ブロッキングI/O処理をシミュレートする同期関数"""
#     print(f"Blocking operation for {duration} seconds started...")
#     time.sleep(duration)
#     print("Blocking operation finished.")
#     return f"Completed after {duration} seconds"
#
# async def main():
#     print("Running a blocking function in a separate thread...")
#     result = await AsyncUtils.run_in_thread(blocking_io_operation, 2)
#     print(f"Result from thread: {result}")
#
#     print("\nCreating a background task...")
#     task = AsyncUtils.safe_create_task(some_async_function())
#     # taskの完了を待つ場合
#     await task
#
# async def some_async_function():
#     """バックグラウンドで実行される非同期関数"""
#     print("Background task started...")
#     await asyncio.sleep(1)
#     print("Background task finished.")
#
# if __name__ == '__main__':
#     # このスクリプトを直接実行して使用例をテストする場合
#     logging.basicConfig(level=logging.INFO)
#     asyncio.run(main())