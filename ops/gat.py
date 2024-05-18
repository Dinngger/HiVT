import taichi as ti

ti.init(arch=ti.cuda, unrolling_limit=128)
t3 = ti.types.ndarray(ti.f32, ndim=3)
t2 = ti.types.ndarray(ti.f32, ndim=2)
t2u = ti.types.ndarray(ti.u1, ndim=2)
t1 = ti.types.ndarray(ti.f32, ndim=1)
t1i = ti.types.ndarray(ti.i64, ndim=1)

@ti.func
def linear2(x, w: t2, b: t1):
    res = ti.Vector.zero(ti.f32, 64)
    for i in ti.static(range(64)):
        res[i] = b[i]
        for j in ti.static(range(2)):
            res[i] += w[i, j] * x[j]
    return res

@ti.func
def linear64(x, w: t2, b: t1):
    res = ti.Vector.zero(ti.f32, 64)
    for i in range(64):
        res[i] = b[i]
        for j in ti.static(range(64)):
            res[i] += w[i, j] * x[j]
    return res

@ti.func
def layer_norm(x, gamma: t1, beta: t1):
    mean: ti.f32 = 0.0
    for i in ti.static(range(64)):
        mean += x[i]
    mean /= 64
    for i in ti.static(range(64)):
        x[i] -= mean
    std: ti.f32 = 0.0
    for i in ti.static(range(64)):
        std += x[i] ** 2
    std = ti.sqrt(std / 64 + 1e-5)
    for i in ti.static(range(64)):
        x[i] = x[i] * gamma[i] / std + beta[i]
    return x

@ti.func
def relu(x):
    for i in ti.static(range(64)):
        x[i] = ti.max(x[i], 0)
    return x

@ti.kernel
def gat(x: t3,
        out: t3,
        ce_0_w: t2, ce_0_b: t1,
        ce_1_w: t1, ce_1_b: t1,
        ce_3_w: t2, ce_3_b: t1,
        ce_4_w: t1, ce_4_b: t1,
        ce_6_w: t2, ce_6_b: t1,
        ce_7_w: t1, ce_7_b: t1,
        bos_mask: t2u, bos_token: t2):
    for b, n in ti.ndrange(x.shape[0], x.shape[1]):
        ce = linear2(ti.Vector([x[b, n, 0], x[b, n, 1]]), ce_0_w, ce_0_b)
        ce = layer_norm(ce, ce_1_w, ce_1_b)
        ce = relu(ce)
        ce = linear64(ce, ce_3_w, ce_3_b)
        ce = layer_norm(ce, ce_4_w, ce_4_b)
        ce = relu(ce)
        ce = linear64(ce, ce_6_w, ce_6_b)
        ce = layer_norm(ce, ce_7_w, ce_7_b)

        if bos_mask[n, b]:
            for i in ti.static(range(64)):
                ce[i] = bos_token[b, i]
        for i in ti.static(range(64)):
            out[b, n, i] = ce[i]

@ti.kernel
def gat2(ces: t2, x: t2, out: t2, ce_out: t2, gates: t2,
         edge_begins: t1i, edge_index: t1i,
         n1_w: t1, n1_b: t1,
         rotate: t3, edge_attr: t2,
         nbr_0_w: t2, nbr_0_b: t1,
         nbr_1_w: t1, nbr_1_b: t1,
         nbr_3_w: t2, nbr_3_b: t1,
         ea_0_w: t2, ea_0_b: t1,
         ea_1_w: t1, ea_1_b: t1,
         ea_3_w: t2, ea_3_b: t1,
         aggr_0_w: t1, aggr_0_b: t1,
         aggr_2_w: t2, aggr_2_b: t1,
         aggr_3_w: t1, aggr_3_b: t1,
         q_w: t2, q_b: t1,
         k_w: t2, k_b: t1,
         v_w: t2, v_b: t1,
         ih_w: t2, ih_b: t1,
         hh_w: t2, hh_b: t1,
         self_w: t2, self_b: t1,
         out_w: t2, out_b: t1):
    for idx in range(ces.shape[0]):
        ce = ti.Vector.zero(ti.f32, 64)
        for i in ti.static(range(64)):
            ce[i] = ces[idx, i]
        ce = layer_norm(ce, n1_w, n1_b)
        for i in ti.static(range(64)):
            ce_out[idx, i] = ce[i]
        query = linear64(ce, q_w, q_b)
        cr_i = idx % (ces.shape[0] // 20)
        cr = ti.Matrix([[rotate[cr_i, 0, 0], rotate[cr_i, 0, 1]],
                        [rotate[cr_i, 1, 0], rotate[cr_i, 1, 1]]]).transpose()
        end = edge_index.shape[0] if idx == ces.shape[0] - 1 else edge_begins[idx + 1]
        first: ti.u1 = True
        alpha_sum = ti.Vector.one(ti.f32, 8)
        alpha_max = ti.Vector.one(ti.f32, 8)
        out_sum = ti.Vector.zero(ti.f32, 64)
        for e in range(edge_begins[idx], end):
            ei = edge_index[e]

            x_j2 = cr @ ti.Vector([x[ei, 0], x[ei, 1]])
            x_j = linear2(x_j2, nbr_0_w, nbr_0_b)
            x_j = layer_norm(x_j, nbr_1_w, nbr_1_b)
            x_j = relu(x_j)
            x_j = linear64(x_j, nbr_3_w, nbr_3_b)

            ea2 = cr @ ti.Vector([edge_attr[e, 0], edge_attr[e, 1]])
            ea = linear2(ea2, ea_0_w, ea_0_b)
            ea = layer_norm(ea, ea_1_w, ea_1_b)
            ea = relu(ea)
            ea = linear64(ea, ea_3_w, ea_3_b)

            aggr = x_j + ea
            aggr = layer_norm(aggr, aggr_0_w, aggr_0_b)
            aggr = relu(aggr)
            aggr = linear64(aggr, aggr_2_w, aggr_2_b)
            nbr = layer_norm(aggr, aggr_3_w, aggr_3_b)

            key = linear64(nbr, k_w, k_b)
            value = linear64(nbr, v_w, v_b)
            alpha = ti.Vector.zero(ti.f32, 8)
            for i in ti.static(range(8)):
                alpha[i] = query[i*8:(i+1)*8].dot(key[i*8:(i+1)*8]) / ti.sqrt(64 / 8)

            if first:
                first = False
                alpha_max = alpha
                out_sum = value
            else:
                for i in ti.static(range(8)):
                    if alpha[i] <= alpha_max[i]:
                        alpha[i] = ti.exp(alpha[i] - alpha_max[i])
                        alpha_sum[i] += alpha[i]
                        out_sum[i*8:(i+1)*8] += value[i*8:(i+1)*8] * alpha[i]
                    else:
                        delta = ti.exp(alpha_max[i] - alpha[i])
                        alpha_max[i] = alpha[i]
                        alpha_sum[i] *= delta
                        out_sum[i*8:(i+1)*8] *= delta
                        alpha_sum[i] += 1.0
                        out_sum[i*8:(i+1)*8] += value[i*8:(i+1)*8]
        for i in ti.static(range(8)):
            out_sum[i*8:(i+1)*8] /= alpha_sum[i]
        gate = linear64(out_sum, ih_w, ih_b) + linear64(ce, hh_w, hh_b)
        gate = 1.0 / (1.0 + ti.exp(-gate))
        self_diff = linear64(ce, self_w, self_b) - out_sum
        for i in ti.static(range(64)):
            out_sum[i] += gate[i] * self_diff[i]
        # out_sum = linear64(out_sum, out_w, out_b)
        for i in ti.static(range(64)):
            out[idx, i] = out_sum[i]
            gates[idx, i] = gate[i]


@ti.kernel
def count_index(len: ti.i64, edge: t1i, counts: t1i, out: t1i):
    j: ti.i64 = -1
    ti.loop_config(serialize=True)
    for i in range(len):
        if j < 0:
            out[i] = 0
            if i == edge[0]:
                j += 1
            else:
                continue
        if i == edge[j]:
            out[i] = counts[j]
            j += 1
        else:
            out[i] = counts[j - 1]
