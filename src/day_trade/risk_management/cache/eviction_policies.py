#!/usr/bin/env python3
"""
Cache Eviction Policies
キャッシュ立ち退きポリシー

メモリ制限時のキャッシュアイテム立ち退き戦略
"""

import time
import random
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from abc import ABC, abstractmethod

from ..interfaces.cache_interfaces import ICacheEvictionPolicy

class LRUEvictionPolicy(ICacheEvictionPolicy):
    """LRU（Least Recently Used）立ち退きポリシー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evict(self, cache_items: Dict[str, Any], count: int) -> List[str]:
        """最も最近使用されていないアイテムを立ち退き"""
        if not cache_items:
            return []

        # OrderedDict の場合は先頭から取得
        if isinstance(cache_items, OrderedDict):
            keys_to_evict = list(cache_items.keys())[:count]
        else:
            # 最終アクセス時刻でソート
            items_with_access_time = []
            for key, item in cache_items.items():
                if hasattr(item, 'last_accessed'):
                    items_with_access_time.append((key, item.last_accessed))
                else:
                    # アクセス時刻がない場合は作成時刻を使用
                    access_time = getattr(item, 'created_at', time.time())
                    items_with_access_time.append((key, access_time))

            # アクセス時刻の古い順にソート
            items_with_access_time.sort(key=lambda x: x[1])
            keys_to_evict = [key for key, _ in items_with_access_time[:count]]

        return keys_to_evict

    def should_evict(self, cache_items: Dict[str, Any], new_item_key: str, new_item_value: Any) -> bool:
        """立ち退きが必要かどうか判定"""
        max_size = self.config.get('max_size', 1000)
        return len(cache_items) >= max_size

    def get_metadata(self) -> Dict[str, Any]:
        """ポリシーメタデータ取得"""
        return {
            'name': 'LRU',
            'description': 'Least Recently Used eviction policy',
            'config': self.config
        }

class LFUEvictionPolicy(ICacheEvictionPolicy):
    """LFU（Least Frequently Used）立ち退きポリシー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evict(self, cache_items: Dict[str, Any], count: int) -> List[str]:
        """最も使用頻度の低いアイテムを立ち退き"""
        if not cache_items:
            return []

        # アクセス回数でソート
        items_with_frequency = []
        for key, item in cache_items.items():
            if hasattr(item, 'access_count'):
                frequency = item.access_count
            else:
                frequency = 1  # デフォルトは1回

            # 同じ頻度の場合は最終アクセス時刻で決定
            last_accessed = getattr(item, 'last_accessed', time.time())
            items_with_frequency.append((key, frequency, last_accessed))

        # 使用頻度の少ない順、同じ頻度なら古いアクセス順
        items_with_frequency.sort(key=lambda x: (x[1], x[2]))
        keys_to_evict = [key for key, _, _ in items_with_frequency[:count]]

        return keys_to_evict

    def should_evict(self, cache_items: Dict[str, Any], new_item_key: str, new_item_value: Any) -> bool:
        """立ち退きが必要かどうか判定"""
        max_size = self.config.get('max_size', 1000)
        return len(cache_items) >= max_size

    def get_metadata(self) -> Dict[str, Any]:
        """ポリシーメタデータ取得"""
        return {
            'name': 'LFU',
            'description': 'Least Frequently Used eviction policy',
            'config': self.config
        }

class FIFOEvictionPolicy(ICacheEvictionPolicy):
    """FIFO（First In First Out）立ち退きポリシー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evict(self, cache_items: Dict[str, Any], count: int) -> List[str]:
        """最も古く作成されたアイテムを立ち退き"""
        if not cache_items:
            return []

        # 作成時刻でソート
        items_with_creation_time = []
        for key, item in cache_items.items():
            if hasattr(item, 'created_at'):
                created_at = item.created_at
            else:
                created_at = time.time()  # 現在時刻をデフォルト

            items_with_creation_time.append((key, created_at))

        # 作成時刻の古い順にソート
        items_with_creation_time.sort(key=lambda x: x[1])
        keys_to_evict = [key for key, _ in items_with_creation_time[:count]]

        return keys_to_evict

    def should_evict(self, cache_items: Dict[str, Any], new_item_key: str, new_item_value: Any) -> bool:
        """立ち退きが必要かどうか判定"""
        max_size = self.config.get('max_size', 1000)
        return len(cache_items) >= max_size

    def get_metadata(self) -> Dict[str, Any]:
        """ポリシーメタデータ取得"""
        return {
            'name': 'FIFO',
            'description': 'First In First Out eviction policy',
            'config': self.config
        }

class TTLEvictionPolicy(ICacheEvictionPolicy):
    """TTL（Time To Live）ベース立ち退きポリシー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_ttl = config.get('default_ttl_seconds', 3600)
        self.cleanup_threshold = config.get('cleanup_threshold', 0.8)  # 80%でクリーンアップ開始

    def evict(self, cache_items: Dict[str, Any], count: int) -> List[str]:
        """期限切れまたは期限切れ間近のアイテムを立ち退き"""
        if not cache_items:
            return []

        current_time = time.time()
        items_with_expiry = []

        for key, item in cache_items.items():
            if hasattr(item, 'expires_at') and item.expires_at:
                expires_at = item.expires_at
            elif hasattr(item, 'created_at') and hasattr(item, 'ttl_seconds'):
                expires_at = item.created_at + (item.ttl_seconds or self.default_ttl)
            else:
                # TTL情報がない場合は現在時刻＋デフォルトTTL
                expires_at = current_time + self.default_ttl

            # 期限までの残り時間を計算
            time_to_expire = expires_at - current_time
            items_with_expiry.append((key, time_to_expire, expires_at))

        # 期限切れまたは期限切れ間近の順にソート
        items_with_expiry.sort(key=lambda x: x[1])
        keys_to_evict = [key for key, _, _ in items_with_expiry[:count]]

        return keys_to_evict

    def should_evict(self, cache_items: Dict[str, Any], new_item_key: str, new_item_value: Any) -> bool:
        """立ち退きが必要かどうか判定"""
        max_size = self.config.get('max_size', 1000)
        current_size = len(cache_items)

        # サイズ制限チェック
        if current_size >= max_size:
            return True

        # 期限切れアイテムの割合チェック
        if current_size > 0:
            expired_count = self._count_expired_items(cache_items)
            expired_ratio = expired_count / current_size
            return expired_ratio >= self.cleanup_threshold

        return False

    def get_metadata(self) -> Dict[str, Any]:
        """ポリシーメタデータ取得"""
        return {
            'name': 'TTL',
            'description': 'Time To Live based eviction policy',
            'config': self.config
        }

    def _count_expired_items(self, cache_items: Dict[str, Any]) -> int:
        """期限切れアイテム数カウント"""
        current_time = time.time()
        expired_count = 0

        for item in cache_items.values():
            if hasattr(item, 'is_expired') and item.is_expired():
                expired_count += 1
            elif hasattr(item, 'expires_at') and item.expires_at:
                if current_time > item.expires_at:
                    expired_count += 1

        return expired_count

class RandomEvictionPolicy(ICacheEvictionPolicy):
    """ランダム立ち退きポリシー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.seed = config.get('random_seed')
        if self.seed is not None:
            random.seed(self.seed)

    def evict(self, cache_items: Dict[str, Any], count: int) -> List[str]:
        """ランダムにアイテムを立ち退き"""
        if not cache_items:
            return []

        all_keys = list(cache_items.keys())
        evict_count = min(count, len(all_keys))

        return random.sample(all_keys, evict_count)

    def should_evict(self, cache_items: Dict[str, Any], new_item_key: str, new_item_value: Any) -> bool:
        """立ち退きが必要かどうか判定"""
        max_size = self.config.get('max_size', 1000)
        return len(cache_items) >= max_size

    def get_metadata(self) -> Dict[str, Any]:
        """ポリシーメタデータ取得"""
        return {
            'name': 'Random',
            'description': 'Random eviction policy',
            'config': self.config
        }

class SizeBasedEvictionPolicy(ICacheEvictionPolicy):
    """サイズベース立ち退きポリシー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_memory_bytes = config.get('max_memory_bytes', 100 * 1024 * 1024)  # 100MB
        self.size_estimation_method = config.get('size_estimation_method', 'sys.getsizeof')

    def evict(self, cache_items: Dict[str, Any], count: int) -> List[str]:
        """サイズの大きいアイテムを優先的に立ち退き"""
        if not cache_items:
            return []

        import sys

        items_with_size = []
        for key, item in cache_items.items():
            if self.size_estimation_method == 'sys.getsizeof':
                size = sys.getsizeof(item)
            else:
                # カスタムサイズ計算
                size = self._estimate_size(item)

            # アクセス頻度も考慮（大きくて使用頻度の低いものを優先）
            access_count = getattr(item, 'access_count', 1)
            priority_score = size / access_count  # サイズを使用頻度で割る

            items_with_size.append((key, size, priority_score))

        # 優先スコアの高い順（立ち退き優先度が高い順）にソート
        items_with_size.sort(key=lambda x: x[2], reverse=True)
        keys_to_evict = [key for key, _, _ in items_with_size[:count]]

        return keys_to_evict

    def should_evict(self, cache_items: Dict[str, Any], new_item_key: str, new_item_value: Any) -> bool:
        """立ち退きが必要かどうか判定"""
        current_memory = self._calculate_total_memory(cache_items)
        new_item_size = self._estimate_size(new_item_value)

        return (current_memory + new_item_size) > self.max_memory_bytes

    def get_metadata(self) -> Dict[str, Any]:
        """ポリシーメタデータ取得"""
        return {
            'name': 'SizeBased',
            'description': 'Size based eviction policy',
            'config': self.config
        }

    def _estimate_size(self, obj: Any) -> int:
        """オブジェクトサイズ推定"""
        import sys

        if self.size_estimation_method == 'sys.getsizeof':
            return sys.getsizeof(obj)
        elif hasattr(obj, '__sizeof__'):
            return obj.__sizeof__()
        else:
            # 推定値
            return len(str(obj)) * 2  # 文字列長の2倍を目安

    def _calculate_total_memory(self, cache_items: Dict[str, Any]) -> int:
        """総メモリ使用量計算"""
        total_memory = 0
        for item in cache_items.values():
            total_memory += self._estimate_size(item)
        return total_memory

class CompositeEvictionPolicy(ICacheEvictionPolicy):
    """複合立ち退きポリシー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policies = []
        self.weights = []

        # 子ポリシー作成
        policy_configs = config.get('policies', [])
        for policy_config in policy_configs:
            policy_type = policy_config['type']
            policy_weight = policy_config.get('weight', 1.0)
            policy_params = policy_config.get('config', {})

            policy = self._create_policy(policy_type, policy_params)
            self.policies.append(policy)
            self.weights.append(policy_weight)

        if not self.policies:
            # デフォルトポリシー（LRU + TTL）
            self.policies = [
                LRUEvictionPolicy({}),
                TTLEvictionPolicy({})
            ]
            self.weights = [0.7, 0.3]

    def evict(self, cache_items: Dict[str, Any], count: int) -> List[str]:
        """複数ポリシーを組み合わせて立ち退き"""
        if not cache_items:
            return []

        # 各ポリシーから候補を取得
        candidate_scores = {}

        for policy, weight in zip(self.policies, self.weights):
            policy_candidates = policy.evict(cache_items, len(cache_items))

            # 候補にスコア付け（順位に基づく）
            for rank, key in enumerate(policy_candidates):
                if key not in candidate_scores:
                    candidate_scores[key] = 0

                # 順位に基づくスコア（低い順位ほど高スコア）
                score = weight * (len(policy_candidates) - rank) / len(policy_candidates)
                candidate_scores[key] += score

        # スコアの高い順にソートして指定数を返す
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [key for key, _ in sorted_candidates[:count]]

    def should_evict(self, cache_items: Dict[str, Any], new_item_key: str, new_item_value: Any) -> bool:
        """いずれかのポリシーが立ち退きを推奨する場合は立ち退き"""
        return any(
            policy.should_evict(cache_items, new_item_key, new_item_value)
            for policy in self.policies
        )

    def get_metadata(self) -> Dict[str, Any]:
        """ポリシーメタデータ取得"""
        sub_policies = [policy.get_metadata() for policy in self.policies]
        return {
            'name': 'Composite',
            'description': 'Composite eviction policy',
            'sub_policies': sub_policies,
            'weights': self.weights,
            'config': self.config
        }

    def _create_policy(self, policy_type: str, config: Dict[str, Any]) -> ICacheEvictionPolicy:
        """ポリシーインスタンス作成"""
        policy_classes = {
            'lru': LRUEvictionPolicy,
            'lfu': LFUEvictionPolicy,
            'fifo': FIFOEvictionPolicy,
            'ttl': TTLEvictionPolicy,
            'random': RandomEvictionPolicy,
            'size': SizeBasedEvictionPolicy
        }

        policy_class = policy_classes.get(policy_type.lower())
        if not policy_class:
            raise ValueError(f"Unknown eviction policy type: {policy_type}")

        return policy_class(config)

# ポリシーファクトリー関数

def create_eviction_policy(policy_type: str, config: Dict[str, Any]) -> ICacheEvictionPolicy:
    """立ち退きポリシー作成"""

    policy_classes = {
        'lru': LRUEvictionPolicy,
        'lfu': LFUEvictionPolicy,
        'fifo': FIFOEvictionPolicy,
        'ttl': TTLEvictionPolicy,
        'random': RandomEvictionPolicy,
        'size': SizeBasedEvictionPolicy,
        'composite': CompositeEvictionPolicy
    }

    policy_class = policy_classes.get(policy_type.lower())
    if not policy_class:
        available_policies = ', '.join(policy_classes.keys())
        raise ValueError(
            f"Unknown eviction policy type: {policy_type}. "
            f"Available policies: {available_policies}"
        )

    return policy_class(config)
