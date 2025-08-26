#!/usr/bin/env python3
"""
PyTorch時系列データセット - Issue #697対応: データ読み込み最適化
"""

from typing import Any, Dict, Optional, Tuple, Union

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# Issue #696対応: PyTorch可用性チェックと明確化
PYTORCH_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset

    PYTORCH_AVAILABLE = True
    logger.info("PyTorch利用可能 - 完全なTimeSeriesDatasetを使用")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch未インストール - TimeSeriesDatasetは利用できません")

    # フォールバック用のダミークラス
    class Dataset:
        pass

    class torch:
        class Tensor:
            pass

        @staticmethod
        def from_numpy(array):
            return array

        @staticmethod
        def device(device_name):
            return device_name


if PYTORCH_AVAILABLE:

    class TimeSeriesDataset(Dataset):
        """PyTorch時系列データセット - Issue #697対応: データ読み込み最適化"""

        def __init__(
            self, X: Union[torch.Tensor, Any], y: Optional[Union[torch.Tensor, Any]] = None, device: Optional[Any] = None
        ):
            """
            Issue #697対応: Tensor/NumPyを直接受け取る効率的データセット

            Args:
                X: 入力データ（torch.Tensor or np.ndarray）
                y: ターゲットデータ（torch.Tensor or np.ndarray, optional）
                device: デバイス指定（GPU転送用）
            """
            # Issue #697対応: 入力タイプに応じた効率的処理
            if isinstance(X, torch.Tensor):
                self.X = X.clone().detach()  # 安全なコピー
            else:
                self.X = torch.from_numpy(X).float()  # NumPy -> Tensor

            if y is not None:
                if isinstance(y, torch.Tensor):
                    self.y = y.clone().detach()
                else:
                    self.y = torch.from_numpy(y).float()
            else:
                self.y = None

            # Issue #697対応: デバイス指定時の事前転送（pin_memory用）
            self.device = device
            if device is not None and device.type == 'cpu':
                # CPU使用時はpin_memoryのため、CPUテンソルのまま保持
                self.X = self.X.pin_memory() if not self.X.is_pinned() else self.X
                if self.y is not None:
                    self.y = self.y.pin_memory() if not self.y.is_pinned() else self.y

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            if self.y is not None:
                return self.X[idx], self.y[idx]
            return self.X[idx]

        @staticmethod
        def create_efficient_dataset(
            X: Any, y: Optional[Any] = None, device: Optional[Any] = None, use_pinned_memory: bool = True
        ) -> "TimeSeriesDataset":
            """
            Issue #697対応: 効率的データセット作成ヘルパー

            Args:
                X: 入力データ
                y: ターゲットデータ
                device: デバイス
                use_pinned_memory: ピンメモリ使用フラグ
            """
            if use_pinned_memory and device is not None and device.type == 'cuda':
                # GPU使用時はCPUでpin_memoryを活用
                dataset_device = torch.device('cpu')
            else:
                dataset_device = device

            return TimeSeriesDataset(X, y, dataset_device)

else:

    class TimeSeriesDataset:
        """PyTorch未利用時のダミークラス"""

        def __init__(self, X, y=None, device=None):
            logger.warning("PyTorchが利用できません - TimeSeriesDatasetは無効化されています")
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X) if self.X is not None else 0

        def __getitem__(self, idx):
            return self.X[idx] if self.X is not None else None

        @staticmethod
        def create_efficient_dataset(X, y=None, device=None, use_pinned_memory=True):
            return TimeSeriesDataset(X, y, device)