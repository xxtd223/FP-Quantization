from typing import Tuple, Optional

import torch

'''
定义 quantizer 类，封装
'''


from .quant_args import QuantizationFormat, QuantizationGranularity, QuantizationObserver, ScalePrecision
from .quant_ops import FP8_E4M3_MAX, FP4_E2M1_MAX, FP4_SCALE, get_quantization_fns, get_quantization_range, cast_to_eBm0
from ..helpers import split_dim


def get_e6m2_rec_torch(sf_e6m2: torch.Tensor) -> torch.Tensor:
    """
    E6M2 格式的 4选1 查表  M7 = (4 - M2)/(4 + M2) * 128
    """
    # 提取阶码 E6
    e6 = torch.floor(torch.log2(sf_e6m2.clamp(min=1e-20)))

    # 提取 2-bit 尾数索引 M2 (0, 1, 2, 3)
    # 计算方法：sf_norm = sf_e6m2 / 2^e6，然后映射到 0-3
    m2 = torch.round(sf_e6m2 * torch.pow(2.0, -e6 + 2)) - 4
    m2 = m2.clamp(0, 3)  # 确保索引合法

    # M2: 0 (1.00) -> 0.0
    # M2: 1 (1.25) -> 77.0
    # M2: 2 (1.50) -> 43.0
    # M2: 3 (1.75) -> 18.0
    m7 = torch.where(m2 == 0, 0.0,
                     torch.where(m2 == 1, 77.0,
                                 torch.where(m2 == 2, 43.0, 18.0)))

    # 如果 M2 为 0 (数值为1.0), 倒数阶码为 -e6
    # 否则阶码多减 1
    e8 = torch.where(m2 == 0, -e6, -e6 - 1)

    # 最终组合公式: 2^E8 * (1 + M7 * 2^-7)
    res = torch.pow(2.0, e8) * (1.0 + m7 * torch.pow(2.0, -7.0))
    return res

# Utility function for inversion.
def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


class Quantizer:

    def __init__(
        self, 
        bits: int, 
        symmetric: bool = True,
        format: str = "int",
        granularity: str = "channel",
        observer: str = "minmax",
        dim: int = -1,
        group_size: Optional[int] = None,
        scale_precision: str = "fp16",
        scale_min_clip: Optional[float] = None
    ):
        # hif 格式的特殊处理
        if format == "hif":
            group_size = 64
            symmetric = True
            scale_precision = "e6m2"
        # Sanity checks
        if format in ["fp", "nvfp", "mxfp", "hif"]:
            assert symmetric, "Only symmetric quantization is supported for floating point formats."

        if granularity == "group":
            assert group_size is not None, "Group size must be specified when granularity is 'group'."
        else:
            assert group_size is None, "Group size must be None when granularity is not 'group'."

        self.bits = bits
        self.symmetric = symmetric
        self.format = QuantizationFormat(format)
        self.granularity = QuantizationGranularity(granularity)
        self.observer = QuantizationObserver(observer)
        self.scale_precision = ScalePrecision(scale_precision)
        self.dim = dim
        self.group_size = group_size
        self.scale_min_clip = scale_min_clip

        self.quant_fn, self.dequant_fn, self.quant_dequant_fn = get_quantization_fns(
            format=self.format,
            bits=self.bits,
        )

        self.q_min, self.q_max = get_quantization_range(
            format=self.format,
            bits=self.bits,
            symmetric=self.symmetric,
        )
        
        # Global scale is 3 for MXFP quantization
        if self.format == QuantizationFormat.MXFP:
            self.global_scale = torch.tensor([3.0], dtype=torch.float32)
        else:
            self.global_scale = torch.tensor([float("inf")], dtype=torch.float32)
        # Scale tracking is needed only for E4M3 scale quantization
        self._track_global_scale = (self.scale_precision == ScalePrecision.E4M3)

    def _reshape_before_quantization(
        self, 
        x: torch.Tensor, 
        scales: Optional[torch.Tensor] = None,
        zeros: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.group_size:
            dim = x.ndim - 1 if self.dim == -1 else self.dim
            num_groups = x.shape[dim] // self.group_size
            x = split_dim(x, num_groups, dim)
            if scales is not None:
                scales = scales.unsqueeze(dim + 1)
            if zeros is not None:
                zeros = zeros.unsqueeze(dim + 1)
        return x, scales, zeros

    def get_quantization_params(
        self, 
        x: torch.Tensor,
        # MSE observer quantization params
        scale_search_iters: int = 100,
        max_scale_shrink_factor: float = 0.80,
        error_norm: float = 2.4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get scale and zero point for an input tensor.
        """
        dim = x.ndim - 1 if self.dim == -1 else self.dim
        # --- HiF4 三级缩放逻辑 ---
        if self.format == QuantizationFormat.HiF:
            x_reshaped, _, _ = self._reshape_before_quantization(x)
            abs_x = x_reshaped.abs()

            # 1. Level-1: E6M2 (每 64 个元素一个)
            vmax = abs_x.max(dim=-1, keepdim=True).values
            sf_bf16 = vmax / 7.05
            sf_log2 = torch.log2(sf_bf16.clamp(min=1e-20))
            e6m2_exp = torch.floor(sf_log2).clamp(-48.0, 15.0)
            sf_norm = sf_bf16 / torch.pow(2, e6m2_exp)
            m2_val = torch.round(sf_norm * 4.0) / 4.0
            sf_e6m2 = m2_val * torch.pow(2, e6m2_exp)
            # 使用 4选1 查表法计算倒数
            e6m2_rec = get_e6m2_rec_torch(sf_e6m2)

            # 2. Level-2: E1_8 (每 8 个元素共享)
            v8 = abs_x.view(*abs_x.shape[:-1], 8, 8).max(dim=-1).values
            e1_8 = torch.where(v8 * e6m2_rec >= 4.0, 1.0, 0.0)

            # 3. Level-3: E1_16 (每 4 个元素共享)
            v16 = abs_x.view(*abs_x.shape[:-1], 16, 4).max(dim=-1).values
            e1_8_for_v16 = e1_8.unsqueeze(-1).repeat(1, 1, 1, 2).flatten(-2) if x.ndim > 1 else e1_8.repeat_interleave(
                2, dim=-1)
            e1_16 = torch.where(v16 * e6m2_rec * torch.pow(2, -e1_8_for_v16) >= 2.0, 1.0, 0.0)

            # 4. 合成 Total Scale (形状为 [..., N/64, 64])
            e1_8_full = e1_8.unsqueeze(-1).expand(*e1_8.shape, 8).flatten(-2)
            e1_16_full = e1_16.unsqueeze(-1).expand(*e1_16.shape, 4).flatten(-2)
            total_scale = sf_e6m2 * torch.pow(2, e1_8_full + e1_16_full)

            # 将 scales 压平回原始维度对应的形状，以便 quantize 函数处理
            # 结果形状应为 (..., num_groups, 64)
            return total_scale, torch.zeros_like(total_scale)

        if self.granularity == QuantizationGranularity.GROUP:
            reduce_dim = dim + 1
        elif self.granularity == QuantizationGranularity.CHANNEL:
            reduce_dim = dim
        else:
            reduce_dim = None
        x, _, _ = self._reshape_before_quantization(x)

        x_min = x.amin(dim=reduce_dim, keepdim=True)
        x_max = x.amax(dim=reduce_dim, keepdim=True)

        if self.symmetric:
            scales = 2 * torch.maximum(-x_min, x_max) / (self.q_max - self.q_min)
            zeros =  torch.zeros_like(x_min)
        else:
            scales = (x_max - x_min) / (self.q_max - self.q_min)
            zeros = -(x_min / scales).round()

        if self.observer == QuantizationObserver.MSE:
            init_scales = scales.clone() 
            best_quantization_error = torch.full(x.shape[:-1], float("inf"), device=x.device, dtype=x.dtype)

            for i in range(scale_search_iters):
                scale_shrink_factor = 1 - i * max_scale_shrink_factor / scale_search_iters
                candidate_scales = scale_shrink_factor * init_scales
                candidate_zeros = torch.zeros_like(x_min) if self.symmetric else -(x_min / candidate_scales).round() 
                q = self.quant_fn(x, candidate_scales, candidate_zeros, self.q_min, self.q_max)
                x_reconstructed = self.dequant_fn(q, candidate_scales, candidate_zeros)
                quantization_error = (x - x_reconstructed).abs_().pow_(error_norm).sum(dim=-1)

                if (quantization_error < best_quantization_error).any():
                    improved_ids = torch.where(quantization_error < best_quantization_error)
                    best_quantization_error[improved_ids] = quantization_error[improved_ids]
                    scales[improved_ids] = candidate_scales[improved_ids]
                    if not self.symmetric:
                        zeros[improved_ids] = candidate_zeros[improved_ids]

        # Reshape back
        if self.group_size:
            x = x.flatten(dim, dim + 1)
            scales = scales.squeeze(dim + 1)
            if zeros is not None:
                zeros = zeros.squeeze(dim + 1)

        if self.scale_precision == ScalePrecision.E4M3:
            with torch.no_grad():
                if self._track_global_scale:
                    current_global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX * get_reciprocal(x.abs().max().to(torch.float32).view(1))
                    if not current_global_scale:
                        raise ValueError(f"Current global scale is not finite: {current_global_scale}\n")
                    # Update global scale using min of current and computed scale
                    self.global_scale = torch.minimum(self.global_scale.to(x.device), current_global_scale)
                    
                    if not self.global_scale.isfinite():
                        raise ValueError(f"Global scale is not finite: {self.global_scale}\n")
                    
                # Clamp, convert to fp8, convert back, and rescale in one chain
                scales = (scales * self.global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX) \
                    .to(torch.float8_e4m3fn) \
                    .to(torch.float32) \
                    .mul(get_reciprocal(self.global_scale)) \
                    .to(x.dtype)
        
        elif self.scale_precision == ScalePrecision.E8M0:
            # Inspired by quantize_tseng (see https://github.com/IST-DASLab/Quartet/blob/main/notebooks/benchmark_mxfp4.ipynb)
            # NOTE (in quartet x.abs().max() is defined as a scale insteaf of x.abs().max() / q_max )
            scales = cast_to_eBm0(FP4_E2M1_MAX * scales, ebits=8, emax=2) / FP4_SCALE

        # Set scales to 1 if zero
        scales[scales == 0] = 1

        if scales.isnan().any():
            raise ValueError(f"Scales are not finite.")
      
        return scales, zeros
        
    def quantize(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        q = self.quant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q

    def dequantize(self, q: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = q.shape
        return self.dequant_fn(
            *self._reshape_before_quantization(q, scales, zeros), 
        ).reshape(original_shape)
    
    def __call__(self, x: torch.Tensor, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None) -> torch.Tensor:
        original_shape = x.shape
        q = self.quant_dequant_fn(
            *self._reshape_before_quantization(x, scales, zeros), 
            self.q_min, 
            self.q_max
        ).reshape(original_shape)
        return q
