#!/usr/bin/env python3
"""
モデルプルーニングエンジンモジュール
Issue #379: ML Model Inference Performance Optimization

様々なプルーニング手法の実装
- 重み大きさベースプルーニング（非構造化）
- 構造化チャネルプルーニング
- ブロック構造化プルーニング
- Issue #723対応: ベクトル化最適化版
"""

from typing import Dict

import numpy as np

from ..utils.logging_config import get_context_logger
from .data_structures import CompressionConfig, PruningType

logger = get_context_logger(__name__)


class ModelPruningEngine:
    """モデルプルーニングエンジン"""

    def __init__(self, config: CompressionConfig):
        """プルーニングエンジンの初期化
        
        Args:
            config: 圧縮設定（プルーニングタイプ、削減率等）
        """
        self.config = config

    def apply_magnitude_based_pruning(
        self,
        model_weights: Dict[str, np.ndarray],
        pruning_ratio: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """重み大きさベースプルーニング - Issue #723対応: ベクトル化最適化版
        
        Args:
            model_weights: レイヤー別重み辞書
            pruning_ratio: プルーニング率（削減する重みの割合）
            
        Returns:
            プルーニング済み重み辞書
        """
        pruned_weights = {}

        for layer_name, weights in model_weights.items():
            if len(weights.shape) < 2:  # バイアス等はスキップ
                pruned_weights[layer_name] = weights
                continue

            # Issue #723対応: ベクトル化マグニチュードプルーニング
            pruned_weight = self._vectorized_magnitude_pruning(
                weights, pruning_ratio
            )
            pruned_weights[layer_name] = pruned_weight

        logger.info(
            f"重み大きさベースプルーニング完了（ベクトル化版）: "
            f"{pruning_ratio:.1%}削減"
        )
        return pruned_weights

    def _vectorized_magnitude_pruning(
        self, weights: np.ndarray, pruning_ratio: float
    ) -> np.ndarray:
        """Issue #723対応: ベクトル化マグニチュードプルーニング"""
        try:
            # ベクトル化された絶対値計算と閾値決定
            abs_weights = np.abs(weights)

            # パーセンタイルを使用した高速閾値計算
            threshold = np.percentile(abs_weights, pruning_ratio * 100)

            # ブールマスクの直接生成（メモリ効率良い）
            mask = abs_weights >= threshold

            # インプレース演算でメモリ効率化
            pruned_weight = weights * mask

            # 統計計算（ベクトル化）
            sparsity = np.mean(~mask)  # False の割合

            logger.debug(
                f"ベクトル化マグニチュードプルーニング: "
                f"スパース率 {sparsity:.2%}, 閾値 {threshold:.6f}"
            )

            return pruned_weight

        except Exception as e:
            logger.warning(
                f"ベクトル化マグニチュードプルーニング失敗: {e} - フォールバック"
            )
            return self._fallback_magnitude_pruning(weights, pruning_ratio)

    def _fallback_magnitude_pruning(
        self, weights: np.ndarray, pruning_ratio: float
    ) -> np.ndarray:
        """フォールバック: 従来のマグニチュードプルーニング"""
        # 重みの絶対値でソート（従来実装）
        flat_weights = weights.flatten()
        abs_weights = np.abs(flat_weights)
        threshold_idx = int(len(abs_weights) * pruning_ratio)
        threshold = np.partition(abs_weights, threshold_idx)[threshold_idx]

        # 閾値以下の重みを0に
        mask = np.abs(weights) >= threshold
        pruned_weight = weights * mask

        # 統計記録
        sparsity = np.sum(mask == 0) / mask.size
        logger.debug(
            f"フォールバックマグニチュードプルーニング: "
            f"スパース率 {sparsity:.2%}"
        )

        return pruned_weight

    def apply_structured_channel_pruning(
        self,
        model_weights: Dict[str, np.ndarray],
        pruning_ratio: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """構造化チャネルプルーニング
        
        Args:
            model_weights: レイヤー別重み辞書
            pruning_ratio: プルーニング率（削減するチャネルの割合）
            
        Returns:
            プルーニング済み重み辞書
        """
        pruned_weights = {}

        for layer_name, weights in model_weights.items():
            if len(weights.shape) != 4:  # 畳み込み層以外はスキップ
                pruned_weights[layer_name] = weights
                continue

            # チャネル重要度計算（L2ノルム）
            channel_importance = np.linalg.norm(
                weights.reshape(weights.shape[0], -1), axis=1
            )

            # 重要度下位をプルーニング
            num_channels_to_prune = int(
                len(channel_importance) * pruning_ratio
            )
            pruning_indices = np.argpartition(
                channel_importance, num_channels_to_prune
            )[:num_channels_to_prune]

            # チャネル削除
            mask = np.ones(weights.shape[0], dtype=bool)
            mask[pruning_indices] = False
            pruned_weights[layer_name] = weights[mask]

            logger.debug(
                f"レイヤー {layer_name}: {len(pruning_indices)}チャネル削除"
            )

        logger.info(
            f"構造化チャネルプルーニング完了: {pruning_ratio:.1%}削減"
        )
        return pruned_weights

    def apply_block_structured_pruning(
        self,
        model_weights: Dict[str, np.ndarray],
        block_size: int = 4,
        pruning_ratio: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """ブロック構造化プルーニング - Issue #723対応: ベクトル化最適化版
        
        Args:
            model_weights: レイヤー別重み辞書
            block_size: ブロックサイズ（N×Nブロック）
            pruning_ratio: プルーニング率（削減するブロックの割合）
            
        Returns:
            プルーニング済み重み辞書
        """
        pruned_weights = {}

        for layer_name, weights in model_weights.items():
            if len(weights.shape) < 2:
                pruned_weights[layer_name] = weights
                continue

            # Issue #723対応: ベクトル化されたブロック構造化プルーニング
            pruned_weight = self._vectorized_block_pruning(
                weights, block_size, pruning_ratio
            )
            pruned_weights[layer_name] = pruned_weight

        logger.info(
            f"ブロック構造化プルーニング完了（ベクトル化版）: "
            f"{block_size}x{block_size}, {pruning_ratio:.1%}削減"
        )
        return pruned_weights

    def _vectorized_block_pruning(
        self, weights: np.ndarray, block_size: int, pruning_ratio: float
    ) -> np.ndarray:
        """Issue #723対応: ベクトル化ブロックプルーニング実装"""
        h, w = weights.shape[:2]
        blocks_h = h // block_size
        blocks_w = w // block_size

        # 実際に使用可能なサイズに調整
        effective_h = blocks_h * block_size
        effective_w = blocks_w * block_size
        effective_weights = weights[:effective_h, :effective_w]

        # NumPyのstride_tricksを使用してブロックビューを作成
        try:
            # 4Dブロックビューを作成: (blocks_h, blocks_w, block_size, block_size)
            if len(effective_weights.shape) == 2:
                # 2D重み行列の場合
                block_view = effective_weights.reshape(
                    blocks_h, block_size, blocks_w, block_size
                ).transpose(0, 2, 1, 3)

                # ベクトル化されたL2ノルム計算
                block_norms = np.linalg.norm(
                    block_view.reshape(
                        blocks_h * blocks_w, block_size * block_size
                    ),
                    axis=1,
                )

            elif len(effective_weights.shape) >= 3:
                # 3D以上（例: Conv層）の場合
                rest_dims = effective_weights.shape[2:]
                total_rest = np.prod(rest_dims)

                # reshapeして2D + 残り次元で処理
                temp_weights = effective_weights.reshape(
                    effective_h, effective_w, total_rest
                )

                block_norms_list = []
                for k in range(total_rest):
                    layer_slice = temp_weights[:, :, k]
                    block_view = layer_slice.reshape(
                        blocks_h, block_size, blocks_w, block_size
                    ).transpose(0, 2, 1, 3)

                    layer_block_norms = np.linalg.norm(
                        block_view.reshape(
                            blocks_h * blocks_w, block_size * block_size
                        ),
                        axis=1,
                    )
                    block_norms_list.append(layer_block_norms)

                # 全チャネルの平均ノルムを計算
                block_norms = np.mean(block_norms_list, axis=0)

        except Exception:
            # フォールバック: 従来のループ実装
            logger.warning("ベクトル化プルーニング失敗 - フォールバック実行")
            return self._fallback_block_pruning(
                weights, block_size, pruning_ratio
            )

        # ブロックインデックス生成（ベクトル化）
        block_indices = np.unravel_index(
            np.arange(blocks_h * blocks_w), (blocks_h, blocks_w)
        )
        block_coords = list(zip(block_indices[0], block_indices[1]))

        # 重要度でソートしてプルーニング対象を選択
        importance_with_coords = list(zip(block_norms, block_coords))
        importance_with_coords.sort(key=lambda x: x[0])

        num_blocks_to_prune = int(len(importance_with_coords) * pruning_ratio)
        blocks_to_prune = importance_with_coords[:num_blocks_to_prune]

        # ベクトル化されたマスク作成
        pruning_mask = np.ones_like(weights, dtype=bool)

        # ブロック座標をまとめて処理
        for _, (i, j) in blocks_to_prune:
            pruning_mask[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ] = False

        # マスクを適用してプルーニング実行
        pruned_weight = weights * pruning_mask

        logger.debug(
            f"ベクトル化プルーニング: {len(blocks_to_prune)}ブロック削除, "
            f"計算効率: {blocks_h * blocks_w}ブロック一括処理"
        )

        return pruned_weight

    def _fallback_block_pruning(
        self, weights: np.ndarray, block_size: int, pruning_ratio: float
    ) -> np.ndarray:
        """フォールバック: 従来のループベースブロックプルーニング"""
        h, w = weights.shape[:2]
        blocks_h = h // block_size
        blocks_w = w // block_size

        # ブロックごとのL2ノルム計算（従来実装）
        block_importance = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = weights[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ]
                importance = np.linalg.norm(block)
                block_importance.append((importance, i, j))

        # 重要度でソート
        block_importance.sort(key=lambda x: x[0])

        # 下位ブロックをゼロ化
        num_blocks_to_prune = int(len(block_importance) * pruning_ratio)
        pruned_blocks = block_importance[:num_blocks_to_prune]

        pruned_weight = weights.copy()
        for _, i, j in pruned_blocks:
            pruned_weight[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ] = 0

        logger.debug(f"フォールバックプルーニング: {len(pruned_blocks)}ブロック削除")
        return pruned_weight

    def apply_pruning_by_type(
        self,
        model_weights: Dict[str, np.ndarray],
        pruning_type: PruningType = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """プルーニングタイプに応じた自動プルーニング実行
        
        Args:
            model_weights: レイヤー別重み辞書
            pruning_type: プルーニングタイプ（未指定時は設定から取得）
            **kwargs: プルーニング固有のパラメータ
            
        Returns:
            プルーニング済み重み辞書
        """
        pruning_type = pruning_type or self.config.pruning_type
        pruning_ratio = kwargs.get(
            "pruning_ratio", self.config.pruning_ratio
        )

        if pruning_type == PruningType.MAGNITUDE_BASED:
            return self.apply_magnitude_based_pruning(
                model_weights, pruning_ratio
            )
        elif pruning_type == PruningType.STRUCTURED_CHANNEL:
            return self.apply_structured_channel_pruning(
                model_weights, pruning_ratio
            )
        elif pruning_type == PruningType.STRUCTURED_BLOCK:
            block_size = kwargs.get(
                "block_size", self.config.structured_pruning_granularity
            )
            return self.apply_block_structured_pruning(
                model_weights, block_size, pruning_ratio
            )
        elif pruning_type == PruningType.UNSTRUCTURED:
            # 非構造化 = マグニチュードベースと同じ
            return self.apply_magnitude_based_pruning(
                model_weights, pruning_ratio
            )
        elif pruning_type == PruningType.NONE:
            logger.info("プルーニングなし - 元の重みを返却")
            return model_weights
        else:
            logger.warning(f"未対応プルーニングタイプ: {pruning_type}")
            return model_weights

    def calculate_sparsity_stats(
        self, weights: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """重みのスパース性統計計算
        
        Args:
            weights: レイヤー別重み辞書
            
        Returns:
            スパース性統計辞書
        """
        stats = {}
        total_parameters = 0
        total_zero_parameters = 0

        for layer_name, weight_array in weights.items():
            # ゼロ要素の数
            zero_count = np.sum(weight_array == 0)
            total_count = weight_array.size

            layer_sparsity = zero_count / total_count if total_count > 0 else 0
            stats[layer_name] = layer_sparsity

            total_parameters += total_count
            total_zero_parameters += zero_count

        # 全体スパース率
        stats["overall_sparsity"] = (
            total_zero_parameters / total_parameters
            if total_parameters > 0
            else 0
        )

        logger.debug(
            f"スパース性統計: 全体 {stats['overall_sparsity']:.2%}, "
            f"レイヤー数 {len(weights)}"
        )

        return stats