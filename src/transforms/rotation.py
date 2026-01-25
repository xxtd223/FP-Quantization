# custom_transforms.py

from typing import Optional
import torch
import numpy as np

# 假设 BaseTransform 来自您的 transforms.py
from .transforms import BaseTransform


class RotationTransform(BaseTransform):
    """
    旋转变换类，用于生成特定算法的旋转矩阵。
    """

    def __init__(
            self,
            size: int,
            device: torch.device,
            group_size: Optional[int] = None,
            # 额外参数




            custom_param: float = 0.5
    ):
        super().__init__(size, device, group_size)
        self.custom_param = custom_param
        self.transform_matrix = self._generate_custom_matrix()

    def _generate_custom_matrix(self, R_k=None) -> torch.Tensor:
        """
        旋转矩阵生成算法

        Args:
            self.group_size (k): 旋转块的维度。
            self.size (N): 整个特征维度。
            self.device: 生成矩阵的设备。

        Returns:
            torch.Tensor: (size, size) 形状的块对角旋转矩阵。
        """
        k = self.group_size
        N = self.size

        # 还没实现






        num_blocks = N // k
        blocks = [R_k] * num_blocks

        if N % k != 0:
            raise NotImplementedError("rotation requires size to be divisible by group_size")

        final_H = torch.block_diag(*blocks)
        return final_H.to(self.device).to(torch.float32)


def kronecker_product(A, B):
    """
    计算两个矩阵 A 和 B 的克罗内克积 (Kronecker Product)。
    """
    C = np.kron(A, B)
    return C
