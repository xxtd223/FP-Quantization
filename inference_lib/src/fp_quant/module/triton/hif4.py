import torch
import triton
import triton.language as tl


@tl.jit
def get_e6m2_rec(sf_e6m2):
    """
    E6M2 格式的 4选1 查表法计算倒数
    """
    # 提取阶码 E6 和 2-bit 尾数索引 M2
    e6 = tl.math.floor(tl.math.log2(tl.maximum(sf_e6m2, 1e-20)))
    m2 = tl.math.round(sf_e6m2 * tl.math.exp2(-e6 + 2)) - 4

    # 4选1 查表确定 M7 值 (M2: 0 (1.00), 1 (1.25), 2 (1.50), 3 (1.75))
    # M7 = (4 - M2)/(4 + M2) * 128
    m7 = tl.where(m2 == 0, 0.0,
                  tl.where(m2 == 1, 77.0,
                           tl.where(m2 == 2, 43.0, 18.0)))

    # 如果 M2 为 0，E8 = -E6；否则 E8 = -E6 - 1
    e8 = tl.where(m2 == 0, -e6, -e6 - 1)

    # 2^E8 * (1 + M7 * 2^-7)
    res = tl.math.exp2(e8) * (1.0 + m7 * tl.math.exp2(-7.0))
    return res


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 8 * 64}),
        triton.Config({"BLOCK_SIZE": 16 * 64}),
        triton.Config({"BLOCK_SIZE": 32 * 64}),
        triton.Config({"BLOCK_SIZE": 64 * 64}),
        triton.Config({"BLOCK_SIZE": 128 * 64}),
        triton.Config({"BLOCK_SIZE": 256 * 64}),
    ],
    key=[],
)
@triton.jit
def hif4_forward_kernel(
        x_ptr,
        hadamard_matrix_ptr,
        output_ptr,
        clip_mask_ptr,
        n_elements: tl.constexpr,
        hadamard_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    # 加载 Hadamard 矩阵用于旋转数据分布
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(
        hadamard_dim, hadamard_dim
    )

    # 加载输入数据 x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)

    # Hadamard 变换
    # 将数据排列为 (行, hadamard_dim) 进行矩阵乘法
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)

    # 分组 (G=64)
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // 64, 64))
    abs_x = tl.abs(x_had_grouped)

    # Level-1: Scale (E6M2) 计算
    # Vmax: 每 64 个元素的最大值
    vmax = tl.max(abs_x, axis=1, keep_dims=True)
    # SF_BF16 = Vmax * (1/7.05)
    sf_bf16 = vmax * (1.0 / 7.05)

    # BF16 to E6M2
    sf_log2 = tl.math.log2(tl.maximum(sf_bf16, 1e-20))  # 防止 log(0)
    e6m2_exp = tl.math.floor(sf_log2)
    # 截断到 E6M2 范围 [-48, 15]
    e6m2_exp = tl.maximum(tl.minimum(e6m2_exp, 15.0), -48.0)
    # 计算 E6M2 的 2-bit 尾数 (1.M2 格式)
    sf_norm = sf_bf16 / tl.math.exp2(e6m2_exp)
    # 四舍五入到最近的 0.25
    m2_val = tl.math.round(sf_norm * 4.0) / 4.0
    sf_e6m2 = m2_val * tl.math.exp2(e6m2_exp)

    e6m2_rec = get_e6m2_rec(sf_e6m2)

    # Level-2: 每 8 个元素共享一个 E1_8 (64/8=8个块)
    v8 = tl.max(tl.reshape(abs_x, (BLOCK_SIZE // 64, 8, 8)), axis=2)
    e1_8 = tl.where(v8 * e6m2_rec >= 4.0, 1.0, 0.0)  #

    # Level-3: 每 4 个元素共享一个 E1_16 (64/4=16个块)
    v16 = tl.max(tl.reshape(abs_x, (BLOCK_SIZE // 64, 16, 4)), axis=2)
    # 将 e1_8 广播到 16 个块的维度与 v16 匹配
    e1_8_expanded = tl.reshape(tl.broadcast_to(tl.reshape(e1_8, (BLOCK_SIZE // 64, 8, 1)), (BLOCK_SIZE // 64, 8, 2)),
                               (BLOCK_SIZE // 64, 16))
    e1_16 = tl.where(v16 * e6m2_rec * tl.math.exp2(-e1_8_expanded) >= 2.0, 1.0, 0.0)  #

    # 计算每个元素的总指数偏移 DE64 = E1_8 + E1_16
    e1_16_full = tl.reshape(tl.broadcast_to(tl.reshape(e1_16, (BLOCK_SIZE // 64, 16, 1)), (BLOCK_SIZE // 64, 16, 4)),
                            (BLOCK_SIZE // 64, 64))
    e1_8_full = tl.reshape(tl.broadcast_to(tl.reshape(e1_8, (BLOCK_SIZE // 64, 8, 1)), (BLOCK_SIZE // 64, 8, 8)),
                           (BLOCK_SIZE // 64, 64))
    total_exp = e1_8_full + e1_16_full

    # Vin = V64 * E6M2_REC * 2^(-E1_8 - E1_16)
    vin = x_had_grouped * e6m2_rec * tl.math.exp2(-total_exp)

    # S1P2 映射: 1.M2 格式 (0, 0.25, ..., 1.75 对应 3-bit 0-7) (步长 0.25)
    s1p2_quantized = tl.math.round(vin / 0.25) * 0.25
    s1p2_quantized = tl.maximum(tl.minimum(s1p2_quantized, 1.75), -1.75)

    if clip_mask_ptr is not None:
        tl.store(
            clip_mask_ptr + offsets,
            tl.reshape(tl.abs(vin) <= 1.75, (BLOCK_SIZE,)),
            mask=mask,
        )

    # 反量化
    # 公式: E6M2 * 2^(E1_8 + E1_16) * S1P2
    x_dequantized = s1p2_quantized * (sf_e6m2 * tl.math.exp2(total_exp))

    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def hif4_forward_kernel_wrapper(
        x,
        hadamard_matrix,
        return_clip_mask=False,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    clip_mask = torch.empty_like(x, dtype=torch.bool) if return_clip_mask else None

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch.device(x.device):
        hif4_forward_kernel[grid](
            x_ptr=x,
            hadamard_matrix_ptr=hadamard_matrix,
            output_ptr=output,
            clip_mask_ptr=clip_mask,
            n_elements=n_elements,
            hadamard_dim=hadamard_matrix.shape[-1],
        )

    return output, clip_mask