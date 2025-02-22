import math
import torch
import triton
import triton.language as tl


__all__ = ["attention"]


def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

def rounded_multiple(a, b):
    return (a + b - 1) // b * b

# --------------------------- public API ---------------------------
def attention(q, k, v, tree_mask, sm_scale=None, return_log_normalizer=True):
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
    assert Dk in {16, 32, 64, 128}

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk, Hv = k.shape[1], v.shape[1]
    assert Hk == Hv, "num of heads in k and v should be equal"
    assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
    num_groups = H // Hk

    P_SEQ = N - M
    larger_m = M > N

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # contiguity
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

    device = torch.cuda.device_of(q)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    with torch.cuda.device(device):
        seed, offset = 0, 0

        config = get_fwd_config(B, H, M, N, D, causal=True)
        BLOCK_M, BLOCK_N, num_stages, num_warps = config

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        # consider using 3d grid to avoid div & rem
        grid = (triton.cdiv(M, BLOCK_M), H, B)
        o = torch.empty_like(q)
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        _fwd_kernel[grid](
            q, k, v, tree_mask, sm_scale,
            seed, offset,
            L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            tree_mask.stride(0), tree_mask.stride(1), tree_mask.stride(2),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, M, N, P_SEQ, num_groups,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D, LARGER_M=larger_m,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            num_warps=num_warps, num_stages=num_stages,
        )


    if return_log_normalizer:
        outs = (
            o,
            L 
        )
        return outs
    return o

# --------------------------- Forward ---------------------------
# NOTE: this function can be overwritten at runtime to use your custom config
def get_fwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
                else:
                    BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 3, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else: # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


@triton.jit
def _fwd_kernel(
    Q, K, V, MASK, sm_scale,
    seed,
    offset,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_maskz, stride_maskm, stride_maskn,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    MASK += off_z * stride_maskz
    L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    #Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I).to(input_dtype)

    hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
    if LARGER_M:
        hi = tl.maximum(0, hi)

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    mask_ptrs = MASK + (offs_m[:, None] * stride_maskm + offs_n_init[None, :] * stride_maskn) # BLOCK_M, BLOCK_N
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
            tree_mask = tl.load(mask_ptrs, cache_modifier=".cg")

        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
            tree_mask = tl.load(mask_ptrs, mask=(mask_m[:, None] | mask_n[None, :]), cache_modifier=".cg")
        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)
        tree_mask = tl.where(tree_mask, 0, float("-inf"))
        s += tree_mask
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
        s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)


        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        mask_ptrs += BLOCK_N * stride_maskn
    # write back l & o
    if LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i) # log(normalizer)


    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


if __name__ == "__main__":
    import torch
    import math

    B, M, N, K = 1, 69, 80, 128
    device = "cuda"
    dtype = torch.float16

    q = torch.randn([B, 32, M, K], device=device, dtype=dtype)
    k = torch.randn([B, 8, N, K], device=device, dtype=dtype)
    v = torch.randn([B, 8, N, K], device=device, dtype=dtype)

    mask = torch.zeros((B, M, N), device=device, dtype=dtype).normal_()
    mask_id = mask.clone()
    mask[mask_id > 0] = float('-inf')
    mask[mask_id < 0] = 0
    print(attention(q, k, v, mask))
