@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    rotary_dim,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # 检索给定轴的程序 ID。用于标识当前执行实例负责处理的数据块。
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2  # 旋转维度的一半，因为旋转编码在维度对（如实部和虚部）上操作。

    X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rm_cs = rm + SEQLEN_OFFSETS
    tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
    X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    cos = tl.load(
        COS,
        mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        SIN,
        mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
        other=0.0,
    ).to(tl.float32)
    x0 = tl.load(
        X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
    ).to(tl.float32)
    x1 = tl.load(
        X + rotary_dim_half * stride_x_headdim,
        mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        other=0.0,
    ).to(tl.float32)

    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    # write back result
    OUT = OUT + (
        rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
    )
    tl.store(
        OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half)
    )
    tl.store(
        OUT + rotary_dim_half * stride_out_headdim,
        o1,
        mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
    )
