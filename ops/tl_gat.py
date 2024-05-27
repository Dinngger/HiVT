import triton
import triton.language as tl

@triton.jit
def load_matrix(ptr, row: tl.constexpr, col: tl.constexpr):
    return tl.load(ptr + tl.arange(0, row)[:, None] + tl.arange(0, col)[None, :])

@triton.jit
def load_vector(ptr, row: tl.constexpr):
    return tl.load(ptr + tl.arange(0, row))

@triton.jit
def fake_dot(x, w):
    # return tl.dot(x, w)
    tl.static_print(x, w)
    res = tl.sum(x[:, :, None] * w[None, :, :], axis=1)
    return res

@triton.jit
def linear2(x, W, B):
    w = load_matrix(W, 64, 2).trans()
    b = load_vector(B, 64)
    res = fake_dot(x, w)
    return res + b

@triton.jit
def linear(x, W, B):
    w = load_matrix(W, 64, 64).to(tl.float16).trans()
    b = load_vector(B, 64)
    res = tl.dot(x.to(tl.float16), w)
    return res + b

@triton.jit
def layer_norm(x, Gamma, Beta):
    gamma = load_vector(Gamma, 64)
    beta = load_vector(Beta, 64)
    mean = tl.sum(x, axis=0) / 64
    x -= mean
    var = tl.sum(x * x, axis=0) / 64
    rstd = 1 / tl.sqrt(var + 1e-5)
    x = x * rstd * gamma + beta
    return x

@triton.jit
def relu(x):
    return tl.maximum(x, 0)

@triton.jit
def gat_kernel(X, Y,
        B, N,
        W_ce_0, B_ce_0,
        W_ce_1, B_ce_1,
        W_ce_3, B_ce_3,
        W_ce_4, B_ce_4,
        W_ce_6, B_ce_6,
        W_ce_7, B_ce_7,
        Mask_bos, Token_bos,
        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    b = idx // N
    n = idx % N
    mask = idx < B * N
    x_ptr = X + (b * N * 2 + n * 2)[:, None] + tl.arange(0, 2)[None, :]
    x = tl.load(x_ptr, mask=mask[:, None])
    ce = linear2(x, W_ce_0, B_ce_0)
    ce = layer_norm(ce, W_ce_1, B_ce_1)
    ce = relu(ce)
    ce = linear(ce, W_ce_3, B_ce_3)
    ce = layer_norm(ce, W_ce_4, B_ce_4)
    ce = relu(ce)
    ce = linear(ce, W_ce_6, B_ce_6)
    ce = layer_norm(ce, W_ce_7, B_ce_7)
    bos_mask = tl.load(Mask_bos + n * B + b, mask=mask)

    b0 = pid * BLOCK_SIZE // N
    bm = ((pid + 1) * BLOCK_SIZE - 1) // N
    bos_token = load_matrix(Token_bos + b0 * 64, 1, 64)
    if bm == b0:
        ce = tl.where(bos_mask[:, None], bos_token, ce)
    else:
        bos_token2 = load_matrix(Token_bos + (b0 + 1) * 64, 1, 64)
        ce = tl.where((bos_mask & (b == b0))[:, None], bos_token, ce)
        ce = tl.where((bos_mask & (b == bm))[:, None], bos_token2, ce)
    y_ptr = Y + (b * N * 64 + n * 64)[:, None] + tl.arange(0, 64)[None, :]
    tl.store(y_ptr, ce, mask=mask[:, None])

def gat(X, Y,
        W_ce_0, B_ce_0,
        W_ce_1, B_ce_1,
        W_ce_3, B_ce_3,
        W_ce_4, B_ce_4,
        W_ce_6, B_ce_6,
        W_ce_7, B_ce_7,
        Mask_bos, Token_bos):
    B, N, _ = X.shape
    BLOCK_SIZE = 32
    num_warps = 4
    grid = (triton.cdiv(B * N, BLOCK_SIZE),)
    gat_kernel[grid](
        X, Y, B, N,
        W_ce_0, B_ce_0, W_ce_1, B_ce_1,
        W_ce_3, B_ce_3, W_ce_4, B_ce_4,
        W_ce_6, B_ce_6, W_ce_7, B_ce_7,
        Mask_bos, Token_bos,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE)

@triton.jit
def gat2_kernel(ces_ptr, x_ptr, out_ptr, N,
         edge_begins_ptr, edge_index_ptr, E,
         W_n1, B_n1,
         Rotate, edge_attr_ptr,
         W_nbr_0, B_nbr_0,
         W_nbr_1, B_nbr_1,
         W_nbr_3, B_nbr_3,
         W_ea_0, B_ea_0,
         W_ea_1, B_ea_1,
         W_ea_3, B_ea_3,
         W_aggr_0, B_aggr_0,
         W_aggr_2, B_aggr_2,
         W_aggr_3, B_aggr_3,
         W_q, B_q, W_k, B_k, W_v, B_v,
         W_ih, B_ih, W_hh, B_hh,
         W_self, B_self,
         W_out, B_out,
         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    idx_mask = idx < N
    ce = tl.load(ces_ptr + (idx * 64)[:, None] + tl.arange(0, 64)[None, :], mask=idx_mask[:, None])
    ce = layer_norm(ce, W_n1, B_n1)
    query = linear(ce, W_q, B_q)
    cr_i = idx % (N // 20)
    cr = tl.load(Rotate + cr_i[:, None] + tl.arange(0, 4)[None, :], mask=idx_mask[:, None])
    cr = cr.reshape(BLOCK_SIZE, 2, 2)
    begin = tl.load(edge_begins_ptr + idx, mask=idx_mask)
    if (pid + 1) * BLOCK_SIZE <= N:
        end = tl.load(edge_begins_ptr + BLOCK_SIZE + pid * BLOCK_SIZE)
    else:
        end = tl.load(edge_begins_ptr + N)
    alpha_sum = tl.zeros((8,), dtype=tl.float32)
    alpha_max = tl.zeros((8,), dtype=tl.float32)
    out_sum = tl.zeros((BLOCK_SIZE, 64), dtype=tl.float32)
    node_i = 0
    begin0 = tl.load(edge_begins_ptr + pid * BLOCK_SIZE)
    for start in tl.range(begin0, end, BLOCK_SIZE):
        ei_ptr = start + tl.arange(0, BLOCK_SIZE)
        ei_mask = ei_ptr < end
        ei = tl.load(edge_index_ptr + ei_ptr, mask=ei_mask)
        x_j_ptr = x_ptr + (ei * 2)[:, None] + tl.arange(0, 2)[None, :]
        x_j = tl.load(x_j_ptr, mask=ei_mask[:, None])
        x_j = tl.sum(x_j[:, :, None] * cr, axis=1)
        x_j = linear2(x_j, W_nbr_0, B_nbr_0)
        x_j = layer_norm(x_j, W_nbr_1, B_nbr_1)
        x_j = relu(x_j)
        x_j = linear(x_j, W_nbr_3, B_nbr_3)

        ea_ptr = (ei_ptr * 2)[:, None] + tl.arange(0, 2)[None, :]
        ea = tl.load(edge_attr_ptr + ea_ptr, mask=ei_mask[:, None])
        ea = tl.sum(ea[:, :, None] * cr, axis=1)
        ea = linear2(ea, W_ea_0, B_ea_0)
        ea = layer_norm(ea, W_ea_1, B_ea_1)
        ea = relu(ea)
        ea = linear(ea, W_ea_3, B_ea_3)

        aggr = x_j + ea
        aggr = layer_norm(aggr, W_aggr_0, B_aggr_0)
        aggr = relu(aggr)
        aggr = linear(aggr, W_aggr_2, B_aggr_2)
        nbr = layer_norm(aggr, W_aggr_3, B_aggr_3)

        key = linear(nbr, W_k, B_k)
        value = linear(nbr, W_v, B_v)
        alpha = tl.sum((query * key).reshape(BLOCK_SIZE, 8, 8), axis=-1) / tl.sqrt(8)
        while node_i < BLOCK_SIZE:
            begin_i = begin[node_i]
            end_i = begin[node_i + 1]
            alpha_i = alpha[begin_i-start:end_i-start]
            alpha_max_i = tl.max(alpha_i, axis=0)
            if begin_i >= start: # first
                alpha_i -= alpha_max_i
                alpha_i = tl.exp(alpha_i)
                alpha_sum_i = tl.sum(alpha_i, axis=0)
                value_i = value[begin_i-start:end_i-start]
                alpha_i /= alpha_sum_i
                value_i = value_i.reshape(-1, 8, 8) * alpha_i
                out_sum[node_i] = value_i.reshape(-1, 64).sum(axis=0)
                alpha_max = alpha_max_i
            else: # not first
                if alpha_max_i > alpha_max:
                    delta = tl.exp(alpha_max - alpha_max_i)
                    alpha_max = alpha_max_i
                    alpha_sum *= delta
                    out_sum[node_i] *= delta
                alpha_i -= alpha_max
                alpha_i = tl.exp(alpha_i)
                alpha_sum_i = tl.sum(alpha_i, axis=0) + alpha_sum
                value_i = value[begin_i-start:end_i-start]
                alpha_i /= alpha_sum_i
                value_i = value_i.reshape(-1, 8, 8) * alpha_i
                out_sum[node_i] += value_i.reshape(-1, 64).sum(axis=0)
            alpha_sum = alpha_sum_i
            if end_i > start + BLOCK_SIZE:
                break
            node_i += 1
            if end_i == start + BLOCK_SIZE:
                break
    gate = linear(out_sum, W_ih, B_ih) + linear(ce, W_hh, B_hh)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    self_diff = linear(ce, W_self, B_self) - out_sum
    out_sum += gate * self_diff
    out_sum = linear(out_sum, W_out, B_out)
    tl.store(out_ptr + (idx * 64)[:, None] + tl.arange(0, 64)[None, :], out_sum, mask=idx_mask[:, None])

def gat2(ces, x, out,
         edge_begins, edge_index,
         W_n1, B_n1,
         Rotate, edge_attr,
         W_nbr_0, B_nbr_0,
         W_nbr_1, B_nbr_1,
         W_nbr_3, B_nbr_3,
         W_ea_0, B_ea_0,
         W_ea_1, B_ea_1,
         W_ea_3, B_ea_3,
         W_aggr_0, B_aggr_0,
         W_aggr_2, B_aggr_2,
         W_aggr_3, B_aggr_3,
         W_q, B_q, W_k, B_k, W_v, B_v,
         W_ih, B_ih, W_hh, B_hh,
         W_self, B_self,
         W_out, B_out):
    N = ces.shape[0]
    E = edge_index.shape[0]
    BLOCK_SIZE = 32
    num_warps = 4
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gat2_kernel[grid](
        ces, x, out, N,
        edge_begins, edge_index, E,
        W_n1, B_n1,
        Rotate, edge_attr,
        W_nbr_0, B_nbr_0,
        W_nbr_1, B_nbr_1,
        W_nbr_3, B_nbr_3,
        W_ea_0, B_ea_0,
        W_ea_1, B_ea_1,
        W_ea_3, B_ea_3,
        W_aggr_0, B_aggr_0,
        W_aggr_2, B_aggr_2,
        W_aggr_3, B_aggr_3,
        W_q, B_q, W_k, B_k, W_v, B_v,
        W_ih, B_ih, W_hh, B_hh,
        W_self, B_self,
        W_out, B_out,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE)
