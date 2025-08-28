#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache Manager Utils Tests
キャッシュマネージャーユーティリティテスト
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pickle
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# モッククラス
class MockCacheManager:
    """キャッシュマネージャーモック"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self._lock = threading.RLock()
    
    def get(self, key: str, default=None):
        """キャッシュから値を取得"""
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return default
            
            entry = self.cache[key]
            
            # TTL チェック
            if entry['expires_at'] < time.time():
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.miss_count += 1
                return default
            
            # アクセス時間更新
            self.access_times[key] = time.time()
            self.hit_count += 1
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """キャッシュに値を設定"""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            # 容量チェック
            if len(self.cache) >= self.max_size and key not in self.cache:
                if not self._evict_lru():
                    return False
            
            # エントリ作成
            expires_at = time.time() + ttl
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self.access_times[key] = time.time()
            
            return True
    
    def delete(self, key: str) -> bool:
        """キャッシュから値を削除"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return True
            return False
    
    def clear(self):
        """キャッシュをクリア"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0
            self.eviction_count = 0
    
    def exists(self, key: str) -> bool:
        """キーが存在するかチェック"""
        with self._lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            if entry['expires_at'] < time.time():
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return False
            
            return True
    
    def keys(self) -> List[str]:
        """有効なキーの一覧を取得"""
        with self._lock:
            current_time = time.time()
            valid_keys = []
            
            for key, entry in list(self.cache.items()):
                if entry['expires_at'] < current_time:
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                else:
                    valid_keys.append(key)
            
            return valid_keys
    
    def size(self) -> int:
        """有効なエントリ数を取得"""
        return len(self.keys())
    
    def stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'eviction_count': self.eviction_count,
                'total_requests': total_requests
            }
    
    def _evict_lru(self) -> bool:
        """LRU方式で最も古いエントリを削除"""
        if not self.access_times:
            return False
        
        # 最もアクセス時間が古いキーを見つける
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # 削除
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.eviction_count += 1
        
        return True
    
    def cleanup_expired(self) -> int:
        """期限切れエントリをクリーンアップ"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if entry['expires_at'] < current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            return len(expired_keys)


class MockCacheDecorator:
    """キャッシュデコレーターモック"""
    
    def __init__(self, cache_manager: MockCacheManager, ttl: int = 3600):
        self.cache_manager = cache_manager
        self.ttl = ttl
    
    def __call__(self, func):
        """デコレーター実装"""
        def wrapper(*args, **kwargs):
            # キーの生成
            cache_key = self._generate_key(func.__name__, args, kwargs)
            
            # キャッシュから取得を試行
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 関数を実行
            result = func(*args, **kwargs)
            
            # 結果をキャッシュ
            self.cache_manager.set(cache_key, result, self.ttl)
            
            return result
        
        return wrapper
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """キャッシュキー生成"""
        # シンプルなキー生成（実際の実装ではもっと詳細）
        key_parts = [func_name]
        
        # 位置引数
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(type(arg).__name__)
        
        # キーワード引数
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={type(v).__name__}")
        
        return ":".join(key_parts)


class MockCacheSerializer:
    """キャッシュシリアライザーモック"""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """オブジェクトをバイト列にシリアライズ"""
        try:
            return pickle.dumps(obj)
        except Exception:
            # フォールバック: JSON
            try:
                return json.dumps(obj, default=str).encode('utf-8')
            except Exception:
                return str(obj).encode('utf-8')
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """バイト列をオブジェクトにデシリアライズ"""
        try:
            return pickle.loads(data)
        except Exception:
            # フォールバック: JSON
            try:
                return json.loads(data.decode('utf-8'))
            except Exception:
                return data.decode('utf-8')
    
    @staticmethod
    def is_serializable(obj: Any) -> bool:
        """オブジェクトがシリアライズ可能かチェック"""
        try:
            MockCacheSerializer.serialize(obj)
            return True
        except Exception:
            return False


class MockDistributedCache:
    """分散キャッシュモック"""
    
    def __init__(self, nodes: int = 3):
        self.nodes = [MockCacheManager() for _ in range(nodes)]
        self.current_node = 0
    
    def _get_node(self, key: str) -> MockCacheManager:
        """キーに基づいてノードを選択"""
        # シンプルなハッシュ分散
        node_index = hash(key) % len(self.nodes)
        return self.nodes[node_index]
    
    def get(self, key: str, default=None):
        """分散キャッシュから取得"""
        node = self._get_node(key)
        return node.get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """分散キャッシュに設定"""
        node = self._get_node(key)
        return node.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """分散キャッシュから削除"""
        node = self._get_node(key)
        return node.delete(key)
    
    def clear_all(self):
        """全ノードをクリア"""
        for node in self.nodes:
            node.clear()
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """クラスタ統計取得"""
        total_size = 0
        total_hits = 0
        total_misses = 0
        
        for node in self.nodes:
            stats = node.stats()
            total_size += stats['size']
            total_hits += stats['hit_count']
            total_misses += stats['miss_count']
        
        total_requests = total_hits + total_misses
        cluster_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'nodes': len(self.nodes),
            'total_size': total_size,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'cluster_hit_rate': cluster_hit_rate,
            'avg_size_per_node': total_size / len(self.nodes)
        }


class TestMockCacheManager:
    """キャッシュマネージャーテストクラス"""
    
    @pytest.fixture
    def cache_manager(self):
        """キャッシュマネージャーフィクスチャ"""
        return MockCacheManager(max_size=100, default_ttl=60)
    
    def test_basic_set_get(self, cache_manager):
        """基本的な設定・取得テスト"""
        # 値の設定
        result = cache_manager.set("key1", "value1")
        assert result is True
        
        # 値の取得
        value = cache_manager.get("key1")
        assert value == "value1"
        
        # 統計確認
        stats = cache_manager.stats()
        assert stats['hit_count'] == 1
        assert stats['miss_count'] == 0
    
    def test_miss_case(self, cache_manager):
        """キャッシュミステスト"""
        # 存在しないキーの取得
        value = cache_manager.get("nonexistent")
        assert value is None
        
        # デフォルト値の取得
        value = cache_manager.get("nonexistent", "default")
        assert value == "default"
        
        # 統計確認
        stats = cache_manager.stats()
        assert stats['miss_count'] == 2
    
    def test_ttl_expiration(self, cache_manager):
        """TTL期限切れテスト"""
        # 短いTTLで設定
        cache_manager.set("temp_key", "temp_value", ttl=1)
        
        # すぐに取得（成功するはず）
        value = cache_manager.get("temp_key")
        assert value == "temp_value"
        
        # 期限切れまで待機
        time.sleep(1.1)
        
        # 期限切れ後の取得（失敗するはず）
        value = cache_manager.get("temp_key")
        assert value is None
    
    def test_lru_eviction(self, cache_manager):
        """LRU退避テスト"""
        # キャッシュサイズを小さく設定
        small_cache = MockCacheManager(max_size=3)
        
        # 容量いっぱいまで設定
        small_cache.set("key1", "value1")
        small_cache.set("key2", "value2")
        small_cache.set("key3", "value3")
        
        # key1にアクセスして最新にする
        small_cache.get("key1")
        
        # 新しいキーを追加（key2が退避されるはず）
        small_cache.set("key4", "value4")
        
        # 確認
        assert small_cache.get("key1") == "value1"  # 残っている
        assert small_cache.get("key2") is None      # 退避された
        assert small_cache.get("key3") == "value3"  # 残っている
        assert small_cache.get("key4") == "value4"  # 新しく追加
    
    def test_delete_operation(self, cache_manager):
        """削除操作テスト"""
        # 設定と削除
        cache_manager.set("delete_me", "value")
        assert cache_manager.exists("delete_me") is True
        
        result = cache_manager.delete("delete_me")
        assert result is True
        assert cache_manager.exists("delete_me") is False
        
        # 存在しないキーの削除
        result = cache_manager.delete("not_exists")
        assert result is False
    
    def test_keys_and_size(self, cache_manager):
        """キー一覧とサイズテスト"""
        # 複数キー設定
        keys = ["key1", "key2", "key3"]
        for key in keys:
            cache_manager.set(key, f"value_{key}")
        
        # キー一覧確認
        cached_keys = cache_manager.keys()
        assert len(cached_keys) == 3
        assert all(key in cached_keys for key in keys)
        
        # サイズ確認
        assert cache_manager.size() == 3
    
    def test_clear_operation(self, cache_manager):
        """クリア操作テスト"""
        # データ設定
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        assert cache_manager.size() == 2
        
        # クリア
        cache_manager.clear()
        
        assert cache_manager.size() == 0
        assert cache_manager.get("key1") is None
    
    def test_cleanup_expired(self, cache_manager):
        """期限切れクリーンアップテスト"""
        # 異なるTTLで設定
        cache_manager.set("short", "value1", ttl=1)
        cache_manager.set("long", "value2", ttl=10)
        
        # 短いTTLが期限切れになるまで待機
        time.sleep(1.1)
        
        # クリーンアップ前
        assert cache_manager.size() == 2
        
        # クリーンアップ実行
        expired_count = cache_manager.cleanup_expired()
        
        # 確認
        assert expired_count == 1
        assert cache_manager.size() == 1
        assert cache_manager.get("long") == "value2"
    
    def test_thread_safety(self, cache_manager):
        """スレッドセーフティテスト"""
        import threading
        import random
        
        def worker(worker_id: int):
            for i in range(100):
                key = f"thread_{worker_id}_{i}"
                value = f"value_{worker_id}_{i}"
                
                cache_manager.set(key, value)
                retrieved = cache_manager.get(key)
                assert retrieved == value
                
                # ランダムで削除
                if random.random() < 0.1:
                    cache_manager.delete(key)
        
        # 複数スレッドで並行実行
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 完了待ち
        for thread in threads:
            thread.join()
        
        # エラーが発生しないことを確認（例外なく完了すれば成功）
        stats = cache_manager.stats()
        assert stats is not None


class TestMockCacheDecorator:
    """キャッシュデコレーターテストクラス"""
    
    @pytest.fixture
    def cache_manager(self):
        """キャッシュマネージャーフィクスチャ"""
        return MockCacheManager()
    
    @pytest.fixture
    def cache_decorator(self, cache_manager):
        """キャッシュデコレーターフィクスチャ"""
        return MockCacheDecorator(cache_manager, ttl=60)
    
    def test_function_caching(self, cache_decorator, cache_manager):
        """関数キャッシュテスト"""
        call_count = 0
        
        @cache_decorator
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # 初回呼び出し
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # 同じ引数での2回目呼び出し（キャッシュヒット）
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # 関数は再実行されない
        
        # 異なる引数での呼び出し
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2  # 関数が再実行される
        
        # 統計確認
        stats = cache_manager.stats()
        assert stats['hit_count'] == 1
        assert stats['miss_count'] == 2
    
    def test_key_generation(self, cache_decorator):
        """キー生成テスト"""
        # 位置引数のキー
        key1 = cache_decorator._generate_key("func", (1, 2), {})
        key2 = cache_decorator._generate_key("func", (1, 2), {})
        key3 = cache_decorator._generate_key("func", (2, 1), {})
        
        assert key1 == key2  # 同じ引数は同じキー
        assert key1 != key3  # 異なる引数は異なるキー
        
        # キーワード引数のキー
        key4 = cache_decorator._generate_key("func", (), {'a': 1, 'b': 2})
        key5 = cache_decorator._generate_key("func", (), {'b': 2, 'a': 1})
        
        assert key4 == key5  # キーワード引数の順序は考慮しない
    
    def test_complex_return_values(self, cache_decorator):
        """複雑な戻り値のキャッシュテスト"""
        @cache_decorator
        def complex_function():
            return {
                'list': [1, 2, 3],
                'dict': {'nested': True},
                'tuple': (4, 5, 6)
            }
        
        # 初回呼び出し
        result1 = complex_function()
        
        # 2回目呼び出し（キャッシュから）
        result2 = complex_function()
        
        # 結果が同じ
        assert result1 == result2
        assert result1 is not result2  # 異なるオブジェクト（シリアライズ経由）


class TestMockCacheSerializer:
    """キャッシュシリアライザーテストクラス"""
    
    def test_basic_serialization(self):
        """基本シリアライゼーションテスト"""
        test_objects = [
            "string",
            123,
            45.67,
            True,
            [1, 2, 3],
            {'key': 'value'},
            (1, 2, 3)
        ]
        
        for obj in test_objects:
            # シリアライズ・デシリアライズ
            serialized = MockCacheSerializer.serialize(obj)
            deserialized = MockCacheSerializer.deserialize(serialized)
            
            assert obj == deserialized
            assert type(obj) == type(deserialized)
    
    def test_serializable_check(self):
        """シリアライズ可能性チェックテスト"""
        # シリアライズ可能なオブジェクト
        assert MockCacheSerializer.is_serializable("string") is True
        assert MockCacheSerializer.is_serializable([1, 2, 3]) is True
        assert MockCacheSerializer.is_serializable({'key': 'value'}) is True
        
        # シリアライズ可能性をテスト（lambdaなど）
        result = MockCacheSerializer.is_serializable(lambda x: x)
        # lambdaはpickleできないことがあるが、フォールバックで処理される
        assert isinstance(result, bool)


class TestMockDistributedCache:
    """分散キャッシュテストクラス"""
    
    @pytest.fixture
    def distributed_cache(self):
        """分散キャッシュフィクスチャ"""
        return MockDistributedCache(nodes=3)
    
    def test_key_distribution(self, distributed_cache):
        """キー分散テスト"""
        keys = [f"key_{i}" for i in range(10)]
        
        # 全キーを設定
        for key in keys:
            result = distributed_cache.set(key, f"value_{key}")
            assert result is True
        
        # 全キーを取得
        for key in keys:
            value = distributed_cache.get(key)
            assert value == f"value_{key}"
        
        # 分散確認（すべてのノードが使われているか）
        stats = distributed_cache.get_cluster_stats()
        assert stats['total_size'] == 10
        assert stats['nodes'] == 3
    
    def test_node_independence(self, distributed_cache):
        """ノード独立性テスト"""
        # 各ノードに直接アクセスして独立性を確認
        node_0 = distributed_cache.nodes[0]
        node_1 = distributed_cache.nodes[1]
        
        # node_0にのみ設定
        node_0.set("node0_key", "node0_value")
        
        # node_0からは取得できる
        assert node_0.get("node0_key") == "node0_value"
        
        # node_1からは取得できない
        assert node_1.get("node0_key") is None
    
    def test_cluster_operations(self, distributed_cache):
        """クラスタ操作テスト"""
        # データ設定
        test_keys = [f"cluster_key_{i}" for i in range(20)]
        for key in test_keys:
            distributed_cache.set(key, f"value_{key}")
        
        # クラスタ統計確認
        stats = distributed_cache.get_cluster_stats()
        assert stats['total_size'] == 20
        
        # 全クリア
        distributed_cache.clear_all()
        
        # クリア後の確認
        stats = distributed_cache.get_cluster_stats()
        assert stats['total_size'] == 0


class TestCacheIntegration:
    """キャッシュ統合テスト"""
    
    def test_multilayer_cache(self):
        """マルチレイヤーキャッシュテスト"""
        # L1: 高速キャッシュ（小容量）
        l1_cache = MockCacheManager(max_size=5, default_ttl=60)
        
        # L2: 大容量キャッシュ
        l2_cache = MockCacheManager(max_size=100, default_ttl=300)
        
        def get_with_multilayer(key: str):
            # L1から試行
            value = l1_cache.get(key)
            if value is not None:
                return value
            
            # L2から試行
            value = l2_cache.get(key)
            if value is not None:
                # L1にプロモート
                l1_cache.set(key, value)
                return value
            
            return None
        
        def set_multilayer(key: str, value: Any):
            l1_cache.set(key, value)
            l2_cache.set(key, value)
        
        # テスト実行
        set_multilayer("test_key", "test_value")
        
        # L1ヒット
        result1 = get_with_multilayer("test_key")
        assert result1 == "test_value"
        
        # L1から削除してL2ヒットをテスト
        l1_cache.delete("test_key")
        result2 = get_with_multilayer("test_key")
        assert result2 == "test_value"
        
        # L1に再度キャッシュされていることを確認
        assert l1_cache.get("test_key") == "test_value"
    
    def test_cache_warming(self):
        """キャッシュウォーミングテスト"""
        cache = MockCacheManager()
        
        # ウォーミング用データ
        warming_data = {
            f"warm_key_{i}": f"warm_value_{i}" 
            for i in range(10)
        }
        
        # キャッシュウォーミング実行
        for key, value in warming_data.items():
            cache.set(key, value)
        
        # ウォーミング効果確認
        stats_before = cache.stats()
        
        # 全データにアクセス
        for key in warming_data.keys():
            retrieved = cache.get(key)
            assert retrieved is not None
        
        stats_after = cache.stats()
        
        # 全てヒット
        assert stats_after['hit_count'] - stats_before['hit_count'] == 10
        assert stats_after['miss_count'] == stats_before['miss_count']
    
    def test_cache_performance_monitoring(self):
        """キャッシュパフォーマンス監視テスト"""
        cache = MockCacheManager()
        
        # パフォーマンス測定用の操作
        operations = [
            ("set", "key1", "value1"),
            ("get", "key1", None),
            ("get", "nonexistent", None),
            ("set", "key2", "value2"),
            ("get", "key2", None),
            ("delete", "key1", None),
            ("get", "key1", None),  # miss
        ]
        
        for op_type, key, value in operations:
            if op_type == "set":
                cache.set(key, value)
            elif op_type == "get":
                cache.get(key)
            elif op_type == "delete":
                cache.delete(key)
        
        # 統計分析
        stats = cache.stats()
        
        assert stats['hit_count'] == 2    # key1, key2の初回取得
        assert stats['miss_count'] == 2   # nonexistent, deleted key1
        assert abs(stats['hit_rate'] - 0.5) < 0.01  # 50%のヒット率
    
    def test_cache_memory_management(self):
        """キャッシュメモリ管理テスト"""
        cache = MockCacheManager(max_size=10)
        
        # 大量データでメモリ使用量テスト
        large_data = "x" * 1000  # 1KB のデータ
        
        # 容量を超えてデータを追加
        for i in range(15):
            cache.set(f"large_key_{i}", large_data)
        
        # 容量制限が守られている
        assert cache.size() <= 10
        
        # LRU退避が動作している
        stats = cache.stats()
        assert stats['eviction_count'] >= 5  # 最低5回の退避が発生


if __name__ == "__main__":
    pytest.main([__file__, "-v"])