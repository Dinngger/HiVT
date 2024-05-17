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
def gat2(ces: t2, x: t2, out: t2, x_j: t2,
         edge_begins: t1i, edge_index: t1i,
         n1_w: t1, n1_b: t1):
    for n in range(ces.shape[0]):
        ce = ti.Vector.zero(ti.f32, 64)
        for i in ti.static(range(64)):
            ce[i] = ces[n, i]
        ce = layer_norm(ce, n1_w, n1_b)
        for i in ti.static(range(64)):
            out[n, i] = ce[i]
        begin = 0 if n == 0 else edge_begins[n - 1]
        for e in range(begin, edge_begins[n]):
            ei = edge_index[e]
            for j in ti.static(range(2)):
                x_j[e, j] = x[ei, j]

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
