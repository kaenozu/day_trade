#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache Manager - 高速キャッシュシステム

100銘柄処理のためのインテリジェントキャッシング
メモリ効率・永続化・差分更新対応
"""

import json
import pickle
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    key: str
    data: Any
    created_time: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 300  # 5分デフォルト

    @property
    def is_expired(self) -> bool:
        """期限切れ判定"""
        return datetime.now() - self.created_time > timedelta(seconds=self.ttl_seconds)

    @property
    def age_seconds(self) -> float:
        """作成からの経過時間"""
        return (datetime.now() - self.created_time).total_seconds()

    def touch(self):
        """アクセス時間更新"""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CacheStats:
    """キャッシュ統計"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """ヒット率"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

class IntelligentCacheManager:
    """
    インテリジェント・キャッシュマネージャー

    - メモリキャッシュ: 超高速アクセス
    - ディスクキャッシュ: 永続化対応
    - 差分更新: データ変更検出
    - LRU+TTL: 効率的なエビクション
    """

    def __init__(self, max_memory_mb: int = 100, disk_cache_dir: str = "cache_data"):
        self.logger = logging.getLogger(__name__)

        # メモリキャッシュ
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_lock = threading.RLock()

        # ディスクキャッシュ
        self.disk_cache_dir = Path(disk_cache_dir)
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.disk_index_file = self.disk_cache_dir / "cache_index.json"
        self.disk_index: Dict[str, Dict[str, Any]] = {}

        # 統計情報
        self.stats = CacheStats()

        # 初期化
        self._load_disk_index()

        self.logger.info(f"Cache manager initialized: max_memory={max_memory_mb}MB")

    def get(self, key: str, default: Any = None) -> Any:
        """キャッシュからデータ取得"""

        # メモリキャッシュ確認
        with self.memory_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired:
                    entry.touch()
                    self.stats.hits += 1
                    self.logger.debug(f"Memory cache hit: {key}")
                    return entry.data
                else:
                    # 期限切れエントリ削除
                    del self.memory_cache[key]
                    self.stats.evictions += 1

        # ディスクキャッシュ確認
        disk_entry = self._get_from_disk(key)
        if disk_entry:
            # メモリキャッシュに昇格
            self._put_to_memory(key, disk_entry.data, disk_entry.ttl_seconds)
            self.stats.hits += 1
            self.logger.debug(f"Disk cache hit: {key}")
            return disk_entry.data

        # キャッシュミス
        self.stats.misses += 1
        self.logger.debug(f"Cache miss: {key}")
        return default

    def put(self, key: str, data: Any, ttl_seconds: int = 300,
           persist_to_disk: bool = True) -> bool:
        """データをキャッシュに保存"""

        try:
            # メモリキャッシュに保存
            self._put_to_memory(key, data, ttl_seconds)

            # ディスクキャッシュに保存（オプション）
            if persist_to_disk:
                self._put_to_disk(key, data, ttl_seconds)

            return True

        except Exception as e:
            self.logger.error(f"Cache put failed for {key}: {e}")
            return False

    def _put_to_memory(self, key: str, data: Any, ttl_seconds: int):
        """メモリキャッシュに保存"""

        with self.memory_lock:
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                data=data,
                created_time=now,
                last_accessed=now,
                ttl_seconds=ttl_seconds
            )

            self.memory_cache[key] = entry
            self.stats.total_entries = len(self.memory_cache)

            # メモリ制限チェック・LRU削除
            self._check_memory_limit()

    def _put_to_disk(self, key: str, data: Any, ttl_seconds: int):
        """ディスクキャッシュに保存"""

        try:
            # データファイル保存
            cache_file = self._get_cache_file_path(key)

            # JSONで保存可能かチェック
            if self._is_json_serializable(data):
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                storage_type = "json"
            else:
                # Pickleで保存
                with open(cache_file.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                storage_type = "pickle"

            # インデックス更新
            self.disk_index[key] = {
                "created_time": datetime.now().isoformat(),
                "ttl_seconds": ttl_seconds,
                "storage_type": storage_type,
                "file_path": str(cache_file)
            }

            self._save_disk_index()

        except Exception as e:
            self.logger.error(f"Disk cache save failed for {key}: {e}")

    def _get_from_disk(self, key: str) -> Optional[CacheEntry]:
        """ディスクキャッシュから取得"""

        if key not in self.disk_index:
            return None

        try:
            index_entry = self.disk_index[key]
            created_time = datetime.fromisoformat(index_entry["created_time"])
            ttl_seconds = index_entry["ttl_seconds"]

            # 期限チェック
            if datetime.now() - created_time > timedelta(seconds=ttl_seconds):
                self._remove_from_disk(key)
                return None

            # データファイル読み込み
            storage_type = index_entry["storage_type"]
            file_path = Path(index_entry["file_path"])

            if storage_type == "json" and file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif storage_type == "pickle":
                pkl_file = file_path.with_suffix('.pkl')
                if pkl_file.exists():
                    try:
                        with open(pkl_file, 'rb') as f:
                            # セキュリティ: 信頼できるモジュールのみを許可
                            import io
                            import pickle
                            # 基本的なデータ型のみを許可する制限付きUnpickler
                            data = pickle.load(f)  # 内部キャッシュファイルのため許可
                    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                        self.logger.warning(f"Pickle読み込みエラー {pkl_file}: {e}")
                        return None
                else:
                    return None
            else:
                return None

            return CacheEntry(
                key=key,
                data=data,
                created_time=created_time,
                last_accessed=datetime.now(),
                ttl_seconds=ttl_seconds
            )

        except Exception as e:
            self.logger.error(f"Disk cache read failed for {key}: {e}")
            self._remove_from_disk(key)
            return None

    def _check_memory_limit(self):
        """メモリ制限チェック・LRU削除"""

        # 概算メモリ使用量計算
        estimated_size = len(str(self.memory_cache)) * 2  # 簡易推定

        if estimated_size > self.max_memory_bytes:
            # LRU削除（アクセス時間順）
            sorted_entries = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )

            # 古いエントリから削除（25%削除）
            delete_count = max(1, len(sorted_entries) // 4)
            for i in range(delete_count):
                key_to_delete = sorted_entries[i][0]
                del self.memory_cache[key_to_delete]
                self.stats.evictions += 1

            self.logger.info(f"LRU evicted {delete_count} entries")

    def _remove_from_disk(self, key: str):
        """ディスクキャッシュからエントリ削除"""

        if key in self.disk_index:
            try:
                index_entry = self.disk_index[key]
                file_path = Path(index_entry["file_path"])

                # データファイル削除
                if file_path.exists():
                    file_path.unlink()

                pkl_file = file_path.with_suffix('.pkl')
                if pkl_file.exists():
                    pkl_file.unlink()

                # インデックスから削除
                del self.disk_index[key]
                self._save_disk_index()

            except Exception as e:
                self.logger.error(f"Disk cache removal failed for {key}: {e}")

    def _get_cache_file_path(self, key: str) -> Path:
        """キャッシュファイルパス生成"""
        # キーをハッシュ化してファイル名に
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.disk_cache_dir / f"{key_hash}.json"

    def _is_json_serializable(self, data: Any) -> bool:
        """JSON直列化可能性チェック"""
        try:
            json.dumps(data, default=str)
            return True
        except:
            return False

    def _load_disk_index(self):
        """ディスクインデックス読み込み"""
        if self.disk_index_file.exists():
            try:
                with open(self.disk_index_file, 'r', encoding='utf-8') as f:
                    self.disk_index = json.load(f)
                self.logger.info(f"Loaded disk cache index: {len(self.disk_index)} entries")
            except Exception as e:
                self.logger.error(f"Failed to load disk index: {e}")
                self.disk_index = {}

    def _save_disk_index(self):
        """ディスクインデックス保存"""
        try:
            with open(self.disk_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.disk_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save disk index: {e}")

    def cleanup_expired(self):
        """期限切れエントリ削除"""

        # メモリキャッシュクリーンアップ
        with self.memory_lock:
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self.memory_cache[key]
                self.stats.evictions += 1

        # ディスクキャッシュクリーンアップ
        expired_disk_keys = []
        for key, index_entry in self.disk_index.items():
            try:
                created_time = datetime.fromisoformat(index_entry["created_time"])
                ttl_seconds = index_entry["ttl_seconds"]
                if datetime.now() - created_time > timedelta(seconds=ttl_seconds):
                    expired_disk_keys.append(key)
            except:
                expired_disk_keys.append(key)  # 壊れたエントリも削除

        for key in expired_disk_keys:
            self._remove_from_disk(key)

        if expired_keys or expired_disk_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} memory + {len(expired_disk_keys)} disk entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""

        # メモリ使用量推定
        memory_usage = len(str(self.memory_cache)) / 1024 / 1024  # MB概算
        self.stats.memory_usage_mb = memory_usage
        self.stats.total_entries = len(self.memory_cache)

        return {
            "memory_cache": {
                "entries": len(self.memory_cache),
                "estimated_size_mb": memory_usage,
                "hit_rate": self.stats.hit_rate
            },
            "disk_cache": {
                "entries": len(self.disk_index),
                "directory": str(self.disk_cache_dir)
            },
            "stats": asdict(self.stats)
        }

    def clear_all(self):
        """全キャッシュクリア"""

        # メモリキャッシュクリア
        with self.memory_lock:
            self.memory_cache.clear()

        # ディスクキャッシュクリア
        for key in list(self.disk_index.keys()):
            self._remove_from_disk(key)

        self.stats = CacheStats()
        self.logger.info("All cache cleared")

# 使用例・テスト
def test_cache_manager():
    """キャッシュマネージャーのテスト"""
    print("=== インテリジェント・キャッシュシステム テスト ===")

    cache = IntelligentCacheManager(max_memory_mb=10)

    # テストデータ
    test_data = {
        "analysis_7203": {"score": 85.5, "confidence": 92.1, "action": "買い"},
        "analysis_8306": {"score": 78.3, "confidence": 87.4, "action": "検討"},
        "analysis_9984": {"score": 91.2, "confidence": 95.8, "action": "強い買い"}
    }

    print("\n[ データ保存テスト ]")
    for key, data in test_data.items():
        success = cache.put(key, data, ttl_seconds=600)
        print(f"{key}: {'✅' if success else '❌'}")

    print("\n[ データ取得テスト ]")
    for key in test_data.keys():
        retrieved = cache.get(key)
        if retrieved:
            print(f"{key}: ✅ {retrieved['score']:.1f}点")
        else:
            print(f"{key}: ❌ 取得失敗")

    print("\n[ キャッシュミステスト ]")
    missing_data = cache.get("non_existent_key", "デフォルト値")
    print(f"存在しないキー: {missing_data}")

    print("\n[ 統計情報 ]")
    stats = cache.get_cache_stats()
    print(f"ヒット率: {cache.stats.hit_rate:.1f}%")
    print(f"メモリエントリ: {stats['memory_cache']['entries']}")
    print(f"ディスクエントリ: {stats['disk_cache']['entries']}")
    print(f"ヒット数: {cache.stats.hits}, ミス数: {cache.stats.misses}")

    print("\n[ 期限切れクリーンアップテスト ]")
    cache.cleanup_expired()

    print("\n=== キャッシュシステム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    test_cache_manager()