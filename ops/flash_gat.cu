#include <torch/extension.h>

inline bool CheckCudaError(cudaError_t cuda_result) {
    if (cuda_result != cudaSuccess) {
        const std::string msg = cudaGetErrorString(cuda_result);
        std::cerr << "CUDA ERROR: " << msg << std::endl;
        throw std::runtime_error(msg);
        return false;
    }
    return true;
}

__device__ inline float cr_dot(float cr, float xi, int head_idx) {
    float x0 = __shfl_sync(0xffffffff, xi, 0, 8);
    float x1 = __shfl_sync(0xffffffff, xi, 1, 8);
    float dot = cr * (head_idx < 2 ? x0 : x1);
    float yi = dot + __shfl_down_sync(0xffffffff, dot, 2, 8);
    return yi;
}

__device__ inline void load_64(float* dst, const float* ptr, int n) {
    for (int i = 0; i < 64 * n; i += 64) {
        dst[i + threadIdx.x] = ptr[i + threadIdx.x];
    }
}

__device__ inline void load_64_stride(float* dst, const float* ptr, int n, int s) {
    for (int i = 0; i < 64 * n; i += 64) {
        dst[i + threadIdx.x] = ptr[i * s + threadIdx.x];
    }
}

__device__ inline void linear2(float* res, float x,
    float* weight, float* bias,
    const float* weight_ptr, const float* bias_ptr,
    int head_idx) {
    
    load_64(weight, weight_ptr, 2);
    load_64(bias, bias_ptr, 1);
    __syncthreads();

    float x0 = __shfl_sync(0xffffffff, x, 0, 8);
    float x1 = __shfl_sync(0xffffffff, x, 1, 8);

    for (int i = 0; i < 8; i++) {
        int row = head_idx * 8 + i;
        // bank conflict here
        res[i] = x0 * weight[row * 2] + x1 * weight[row * 2 + 1] + bias[row];
    }
}

__device__ inline float node_sum(float x) {
    x = x + __shfl_down_sync(0xffffffff, x, 1, 2);
    x = x + __shfl_down_sync(0xffffffff, x, 2, 4);
    x = x + __shfl_down_sync(0xffffffff, x, 4, 8);
    x = __shfl_sync(0xffffffff, x, 0, 8);
    return x;
}

__device__ void layer_norm(float* inout,
    float* gamma, float* beta,
    const float* gamma_ptr, const float* beta_ptr, int head_idx) {

    load_64(gamma, gamma_ptr, 1);
    load_64(beta, beta_ptr, 1);
    __syncthreads();

    float mean = 0;
    for (int i = 0; i < 8; i++)
        mean += inout[i];
    mean = node_sum(mean) / 64;
    for (int i = 0; i < 8; i++)
        inout[i] -= mean;
    float var = 0;
    for (int i = 0; i < 8; i++)
        var += inout[i] * inout[i];
    var = rsqrtf(node_sum(var) / 64 + 1e-5);
    for (int i = 0; i < 8; i++) // bank conflict here
        inout[i] = inout[i] * var * gamma[head_idx * 8 + i] +
                   beta[head_idx * 8 + i];
}

__device__ void linear64(float* res, const float* x,
    float* weight, float* bias,
    const float* weight_ptr, const float* bias_ptr,
    int head_idx) {
    load_64(bias, bias_ptr, 1);
    for (int i = 0; i < 8; i++) {
        load_64(weight, weight_ptr + i * 64 * 8, 8);
        __syncthreads();
        for (int j = 0; j < 8; j++) {
            float tmp = 0;
            // bank conflict here
            for (int k = 0; k < 8; k++)
                tmp += x[k] * weight[j * 64 + head_idx * 8 + k];
            tmp = node_sum(tmp) + bias[i * 8 + j];
            if (head_idx == i)  // save to i head.
                res[j] = tmp;   // j value in i head.
        }
    }
}

__device__ inline void linear64_add(float* res, const float* x,
    float* weight, float* bias,
    const float* weight_ptr, const float* bias_ptr,
    int head_idx) {
    load_64(bias, bias_ptr, 1);
    for (int i = 0; i < 8; i++) {
        load_64(weight, weight_ptr + i * 64 * 8, 8);
        __syncthreads();
        for (int j = 0; j < 8; j++) {
            float tmp = 0;
            // bank conflict here
            for (int k = 0; k < 8; k++)
                tmp += x[k] * weight[j * 64 + head_idx * 8 + k];
            tmp = node_sum(tmp) + bias[i * 8 + j];
            if (head_idx == i)  // save to i head.
                res[j] += tmp;   // j value in i head.
        }
    }
}

__device__ inline void linear64_gate(float* res,
    const float* gate, const float* x,
    float* weight, float* bias,
    const float* weight_ptr, const float* bias_ptr,
    int head_idx) {
    load_64(bias, bias_ptr, 1);
    for (int i = 0; i < 8; i++) {
        load_64(weight, weight_ptr + i * 64 * 8, 8);
        __syncthreads();
        for (int j = 0; j < 8; j++) {
            float tmp = 0;
            // bank conflict here
            for (int k = 0; k < 8; k++)
                tmp += x[k] * weight[j * 64 + head_idx * 8 + k];
            tmp = node_sum(tmp) + bias[i * 8 + j];
            if (head_idx == i)  // save to i head.
                res[j] += gate[j] * (tmp - res[j]);
        }
    }
}

__device__ inline void mlp256(float* out,
    float* mid, const float* x,
    float* weight, float* bias,
    const float* mlp_0_w, const float* mlp_0_b,
    const float* mlp_3_w, const float* mlp_3_b,
    int head_idx) {
    for (int i = 0; i < 8; i++)
        out[i] = 0;
    for (int i = 0; i < 4; i++) {
        load_64(bias, mlp_0_b + i * 64, 1);
        for (int j = 0; j < 8; j++) {
            load_64(weight, mlp_0_w + (i * 64 + j * 8) * 64, 8);
            __syncthreads();
            for (int k = 0; k < 8; k++) {
                float tmp = 0;
                // bank conflict here
                for (int l = 0; l < 8; l++)
                    tmp += x[l] * weight[k * 64 + head_idx * 8 + l];
                tmp = node_sum(tmp) + bias[j * 8 + k];
                if (head_idx == j)  // save to j head.
                    mid[k] = max(tmp, 0.0f);   // k value in j head.
            }
        }

        for (int j = 0; j < 8; j++) {
            load_64_stride(weight, mlp_3_w + i * 64 + j * 4 * 8 * 64, 8, 4);
            __syncthreads();
            for (int k = 0; k < 8; k++) {
                float tmp = 0;
                // bank conflict here
                for (int l = 0; l < 8; l++)
                    tmp += mid[l] * weight[k * 64 + head_idx * 8 + l];
                tmp = node_sum(tmp);
                if (head_idx == j)  // save to j head.
                    out[k] += tmp;   // k value in j head.
            }
        }
    }
    load_64(bias, mlp_3_b, 1);
    __syncthreads();
    for (int i = 0; i < 8; i++)
        out[i] += bias[head_idx * 8 + i];
}

__device__ inline void relu(float* x) {
    for (int i = 0; i < 8; i++)
        x[i] = max(x[i], 0.0f);
}

__device__ inline void block_max(float x, bool mask, float* block_tmp) {
    x = mask ? x : -INFINITY;
    x = max(x, __shfl_down_sync(0xffffffff, x, 8));
    x = max(x, __shfl_down_sync(0xffffffff, x, 16));
    __syncthreads();    // make sure other warp don't use block_tmp
    if (threadIdx.x < 8)
        block_tmp[threadIdx.x] = x;
    __syncthreads();
    if (threadIdx.x >= 32 && threadIdx.x < 40)
        block_tmp[threadIdx.x - 32] = max(block_tmp[threadIdx.x - 32], x);
    __syncthreads();
}

__device__ inline void block_sum(float x, bool mask, float* block_tmp) {
    x = mask ? x : 0;
    x = x + __shfl_down_sync(0xffffffff, x, 8);
    x = x + __shfl_down_sync(0xffffffff, x, 16);
    if (threadIdx.x < 8)
        block_tmp[threadIdx.x] = x;
    __syncthreads();
    if (threadIdx.x >= 32 && threadIdx.x < 40)
        block_tmp[threadIdx.x - 32] += x;
    __syncthreads();
}

__global__ void flash_gat_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ rotate,
    const float* __restrict__ ce_0_w,
    const float* __restrict__ ce_0_b,
    const float* __restrict__ ce_1_w,
    const float* __restrict__ ce_1_b,
    const float* __restrict__ ce_3_w,
    const float* __restrict__ ce_3_b,
    const float* __restrict__ ce_4_w,
    const float* __restrict__ ce_4_b,
    const float* __restrict__ ce_6_w,
    const float* __restrict__ ce_6_b,
    const float* __restrict__ ce_7_w,
    const float* __restrict__ ce_7_b,
    const bool* __restrict__ bos_mask,
    const float* __restrict__ bos_token,
    const int64_t* __restrict__ edge_begins,
    const int64_t* __restrict__ edge_index,
    const float* __restrict__ n1_w,
    const float* __restrict__ n1_b,
    const float* __restrict__ edge_attr,
    const float* __restrict__ nbr_0_w,
    const float* __restrict__ nbr_0_b,
    const float* __restrict__ nbr_1_w,
    const float* __restrict__ nbr_1_b,
    const float* __restrict__ nbr_3_w,
    const float* __restrict__ nbr_3_b,
    const float* __restrict__ ea_0_w,
    const float* __restrict__ ea_0_b,
    const float* __restrict__ ea_1_w,
    const float* __restrict__ ea_1_b,
    const float* __restrict__ ea_3_w,
    const float* __restrict__ ea_3_b,
    const float* __restrict__ aggr_0_w,
    const float* __restrict__ aggr_0_b,
    const float* __restrict__ aggr_2_w,
    const float* __restrict__ aggr_2_b,
    const float* __restrict__ aggr_3_w,
    const float* __restrict__ aggr_3_b,
    const float* __restrict__ q_w,
    const float* __restrict__ q_b,
    const float* __restrict__ k_w,
    const float* __restrict__ k_b,
    const float* __restrict__ v_w,
    const float* __restrict__ v_b,
    const float* __restrict__ ih_w,
    const float* __restrict__ ih_b,
    const float* __restrict__ hh_w,
    const float* __restrict__ hh_b,
    const float* __restrict__ self_w,
    const float* __restrict__ self_b,
    const float* __restrict__ out_w,
    const float* __restrict__ out_b,
    const float* __restrict__ n2_w,
    const float* __restrict__ n2_b,
    const float* __restrict__ mlp_0_w,
    const float* __restrict__ mlp_0_b,
    const float* __restrict__ mlp_3_w,
    const float* __restrict__ mlp_3_b,
    int x_size,
    int e_size) {
    int node_idx_0 = blockIdx.x * blockDim.x / 8;
    int node_idx = node_idx_0 + threadIdx.x / 8;
    int head_idx = threadIdx.x % 8;
    bool mask = node_idx < x_size;

    int n = node_idx % (x_size / 20);
    int b = node_idx / (x_size / 20);
    const float* cr_ptr = rotate + n * 4;
    float cr = (mask & head_idx < 4) ? cr_ptr[head_idx] : 0;
    float xi = (mask & head_idx < 2) ? x[node_idx * 2 + head_idx] : 0;
    xi = cr_dot(cr, xi, head_idx);      // only head_idx < 2 is valid.

    __shared__ float weight[64 * 8];
    __shared__ float bias[64];

    float ce1[8];
    float ce2[8];
    linear2(ce1, xi, weight, bias, ce_0_w, ce_0_b, head_idx);
    layer_norm(ce1, weight, bias, ce_1_w, ce_1_b, head_idx);
    relu(ce1);
    linear64(ce2, ce1, weight, bias, ce_3_w, ce_3_b, head_idx);
    layer_norm(ce2, weight, bias, ce_4_w, ce_4_b, head_idx);
    relu(ce2);
    linear64(ce1, ce2, weight, bias, ce_6_w, ce_6_b, head_idx);
    layer_norm(ce1, weight, bias, ce_7_w, ce_7_b, head_idx);

    if (mask & bos_mask[n * 20 + b]) {
        for (int i = 0; i < 8; i++)
            ce1[i] = bos_token[b * 64 + head_idx * 8 + i];
    }

    if (mask) {
        for (int i = 0; i < 8; i++)
            out[node_idx * 64 + head_idx * 8 + i] = ce1[i];   // save the old ce.
    }
    layer_norm(ce1, weight, bias, n1_w, n1_b, head_idx);
    float ce3[8];
    linear64(ce3, ce1, weight, bias, q_w, q_b, head_idx);

    __shared__ float alpha_sum[8];
    __shared__ float alpha_max[8];
    __shared__ float block_tmp[8];
    __shared__ float out_sum[8 * 64];
    __shared__ int node_i, begin0, end;
    if (threadIdx.x == 0) {
        node_i = 0;
        begin0 = edge_begins[node_idx_0];
        if (node_idx_0 + 8 <= x_size)
            end = edge_begins[node_idx_0 + 8];
        else
            end = edge_begins[x_size];
    }
    __syncthreads();

    for (int start = begin0; start < end; start += 8) {
        int ei_ptr = start + threadIdx.x / 8;
        bool ei_mask = ei_ptr < end;
        int ei0 = ei_mask ? edge_index[ei_ptr] : -1;
        int ei1 = ei_mask ? edge_index[ei_ptr + e_size] : -1;
        xi = (ei_mask & head_idx < 2) ? x[ei1 * 2 + head_idx] : 0;
        xi = cr_dot(cr, xi, head_idx);      // only head_idx < 2 is valid.
        linear2(ce1, xi, weight, bias, nbr_0_w, nbr_0_b, head_idx);
        layer_norm(ce1, weight, bias, nbr_1_w, nbr_1_b, head_idx);
        relu(ce1);
        linear64(ce2, ce1, weight, bias, nbr_3_w, nbr_3_b, head_idx);

        xi = (ei_mask & head_idx < 2) ? edge_attr[ei_ptr * 2 + head_idx] : 0;
        xi = cr_dot(cr, xi, head_idx);      // only head_idx < 2 is valid.
        linear2(ce1, xi, weight, bias, ea_0_w, ea_0_b, head_idx);
        layer_norm(ce1, weight, bias, ea_1_w, ea_1_b, head_idx);
        relu(ce1);
        linear64_add(ce2, ce1, weight, bias, ea_3_w, ea_3_b, head_idx);   // ce2 is aggr = x_j + ea

        layer_norm(ce2, weight, bias, aggr_0_w, aggr_0_b, head_idx);
        relu(ce2);
        linear64(ce1, ce2, weight, bias, aggr_2_w, aggr_2_b, head_idx);
        layer_norm(ce1, weight, bias, aggr_3_w, aggr_3_b, head_idx);  // ce1 is nbr

        linear64(ce2, ce1, weight, bias, k_w, k_b, head_idx);     // key

        float alpha = 0;
        for (int i = 0; i < 8; i++)
            alpha += ce2[i] * ce3[i];
        alpha *= 0.353553391;   // 1.0 / sqrtf(8.0);
        linear64(ce2, ce1, weight, bias, v_w, v_b, head_idx);     // value

        while (node_i < 8) {
            __syncthreads();
            if (node_idx_0 + node_i >= x_size)
                break;
            int begin_i = edge_begins[node_idx_0 + node_i];
            int end_i = edge_begins[node_idx_0 + node_i + 1];
            bool mask_i = ei0 == node_idx_0 + node_i;
            if (end_i == begin_i) {
                out_sum[node_i * 64 + threadIdx.x] = 0;
                if (threadIdx.x == 0)
                    node_i++;
                __syncthreads();
                continue;
            }
            block_max(alpha, mask_i, block_tmp);
            float alpha_max_i = block_tmp[head_idx];
            float alpha_sum_i;
            if (begin_i >= start) { // first
                if (mask_i)
                    alpha = expf(alpha - alpha_max_i);
                block_sum(alpha, mask_i, block_tmp);
                alpha_sum_i = block_tmp[head_idx];
                if (mask_i) {
                    alpha /= alpha_sum_i;
                    for (int i = 0; i < 8; i++)
                        ce2[i] *= alpha;
                }
                for (int i = 0; i < 8; i++) {
                    block_sum(ce2[i], mask_i, block_tmp);
                    if (threadIdx.x < 8)
                        out_sum[node_i * 64 + head_idx * 8 + i] = block_tmp[i];
                }
                if (threadIdx.x < 8)
                    alpha_max[head_idx] = alpha_max_i;
            } else {    // not first
                __syncthreads();    // wait alpha_max and alpha_sum.
                if (threadIdx.x < 8 && alpha_max_i > alpha_max[head_idx]) {
                    // only need [0-7] threads to do this global update.
                    float delta = expf(alpha_max[head_idx] - alpha_max_i);
                    alpha_max[head_idx] = alpha_max_i;
                    alpha_sum[head_idx] *= delta;
                    for (int i = 0; i < 8; i++) // maybe use all threads?
                        out_sum[node_i * 64 + head_idx * 8 + i] *= delta;
                }
                __syncthreads();
                if (mask_i)
                    alpha = expf(alpha - alpha_max_i);
                block_sum(alpha, mask_i, block_tmp);
                alpha_sum_i = block_tmp[head_idx] + alpha_sum[head_idx];
                if (threadIdx.x < 8) {
                    float delta = alpha_sum[head_idx] / alpha_sum_i;
                    for (int i = 0; i < 8; i++)
                        out_sum[node_i * 64 + head_idx * 8 + i] *= delta;
                }
                __syncthreads();
                if (mask_i) {
                    alpha /= alpha_sum_i;
                    for (int i = 0; i < 8; i++)
                        ce2[i] *= alpha;
                }
                for (int i = 0; i < 8; i++) {
                    block_sum(ce2[i], mask_i, block_tmp);
                    if (threadIdx.x < 8)
                        out_sum[node_i * 64 + head_idx * 8 + i] += block_tmp[i];
                }
            }
            if (threadIdx.x < 8)
                alpha_sum[head_idx] = alpha_sum_i;
            if (end_i > start + 8)
                break;
            if (threadIdx.x == 0)
                node_i++;
            __syncthreads();
            if (end_i == start + 8)
                break;
        }
    }
    __syncthreads();
    if (mask) {
        for (int i = 0; i < 8; i++) {
            ce1[i] = out_sum[(threadIdx.x / 8) * 64 + head_idx * 8 + i];
            ce2[i] = out[node_idx * 64 + head_idx * 8 + i];     // old ce
            ce3[i] = ce2[i];
        }
    }
    layer_norm(ce2, weight, bias, n1_w, n1_b, head_idx);
    float gate[8];
    linear64(gate, ce1, weight, bias, ih_w, ih_b, head_idx);
    linear64_add(gate, ce2, weight, bias, hh_w, hh_b, head_idx);
    for (int i = 0; i < 8; i++)
        gate[i] = 1.0 / (1.0 + expf(-gate[i]));
    linear64_gate(ce1, gate, ce2, weight, bias, self_w, self_b, head_idx);
    linear64(ce2, ce1, weight, bias, out_w, out_b, head_idx);
    for (int i = 0; i < 8; i++) {
        ce2[i] += ce3[i];
        ce3[i] = ce2[i];
    }
    layer_norm(ce2, weight, bias, n2_w, n2_b, head_idx);
    mlp256(ce1, gate, ce2, weight, bias, mlp_0_w, mlp_0_b, mlp_3_w, mlp_3_b, head_idx);
    if (mask) {
        for (int i = 0; i < 8; i++)
            out[node_idx * 64 + head_idx * 8 + i] = ce1[i] + ce3[i];
    }
}

at::Tensor flash_gat(
    at::Tensor x, at::Tensor out, at::Tensor rotate,
    at::Tensor ce_0_w, at::Tensor ce_0_b,
    at::Tensor ce_1_w, at::Tensor ce_1_b,
    at::Tensor ce_3_w, at::Tensor ce_3_b,
    at::Tensor ce_4_w, at::Tensor ce_4_b,
    at::Tensor ce_6_w, at::Tensor ce_6_b,
    at::Tensor ce_7_w, at::Tensor ce_7_b,
    at::Tensor bos_mask, at::Tensor bos_token,
    at::Tensor edge_begins, at::Tensor edge_index,
    at::Tensor n1_w, at::Tensor n1_b,
    at::Tensor edge_attr,
    at::Tensor nbr_0_w, at::Tensor nbr_0_b,
    at::Tensor nbr_1_w, at::Tensor nbr_1_b,
    at::Tensor nbr_3_w, at::Tensor nbr_3_b,
    at::Tensor ea_0_w, at::Tensor ea_0_b,
    at::Tensor ea_1_w, at::Tensor ea_1_b,
    at::Tensor ea_3_w, at::Tensor ea_3_b,
    at::Tensor aggr_0_w, at::Tensor aggr_0_b,
    at::Tensor aggr_2_w, at::Tensor aggr_2_b,
    at::Tensor aggr_3_w, at::Tensor aggr_3_b,
    at::Tensor q_w, at::Tensor q_b,
    at::Tensor k_w, at::Tensor k_b,
    at::Tensor v_w, at::Tensor v_b,
    at::Tensor ih_w, at::Tensor ih_b,
    at::Tensor hh_w, at::Tensor hh_b,
    at::Tensor self_w, at::Tensor self_b,
    at::Tensor out_w, at::Tensor out_b,
    at::Tensor n2_w, at::Tensor n2_b,
    at::Tensor mlp_0_w, at::Tensor mlp_0_b,
    at::Tensor mlp_3_w, at::Tensor mlp_3_b) {
    std::cout << "x.sizes " << x.sizes() << std::endl;
    std::cout << "out.sizes " << out.sizes() << std::endl;
    const int threads = 64;
    const int blocks = ceil(float(x.sizes()[0]) * 8 / threads);
    flash_gat_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rotate.data_ptr<float>(),
        ce_0_w.data_ptr<float>(),
        ce_0_b.data_ptr<float>(),
        ce_1_w.data_ptr<float>(),
        ce_1_b.data_ptr<float>(),
        ce_3_w.data_ptr<float>(),
        ce_3_b.data_ptr<float>(),
        ce_4_w.data_ptr<float>(),
        ce_4_b.data_ptr<float>(),
        ce_6_w.data_ptr<float>(),
        ce_6_b.data_ptr<float>(),
        ce_7_w.data_ptr<float>(),
        ce_7_b.data_ptr<float>(),
        bos_mask.data_ptr<bool>(),
        bos_token.data_ptr<float>(),
        edge_begins.data_ptr<int64_t>(),
        edge_index.data_ptr<int64_t>(),
        n1_w.data_ptr<float>(),
        n1_b.data_ptr<float>(),
        edge_attr.data_ptr<float>(),
        nbr_0_w.data_ptr<float>(),
        nbr_0_b.data_ptr<float>(),
        nbr_1_w.data_ptr<float>(),
        nbr_1_b.data_ptr<float>(),
        nbr_3_w.data_ptr<float>(),
        nbr_3_b.data_ptr<float>(),
        ea_0_w.data_ptr<float>(),
        ea_0_b.data_ptr<float>(),
        ea_1_w.data_ptr<float>(),
        ea_1_b.data_ptr<float>(),
        ea_3_w.data_ptr<float>(),
        ea_3_b.data_ptr<float>(),
        aggr_0_w.data_ptr<float>(),
        aggr_0_b.data_ptr<float>(),
        aggr_2_w.data_ptr<float>(),
        aggr_2_b.data_ptr<float>(),
        aggr_3_w.data_ptr<float>(),
        aggr_3_b.data_ptr<float>(),
        q_w.data_ptr<float>(),
        q_b.data_ptr<float>(),
        k_w.data_ptr<float>(),
        k_b.data_ptr<float>(),
        v_w.data_ptr<float>(),
        v_b.data_ptr<float>(),
        ih_w.data_ptr<float>(),
        ih_b.data_ptr<float>(),
        hh_w.data_ptr<float>(),
        hh_b.data_ptr<float>(),
        self_w.data_ptr<float>(),
        self_b.data_ptr<float>(),
        out_w.data_ptr<float>(),
        out_b.data_ptr<float>(),
        n2_w.data_ptr<float>(),
        n2_b.data_ptr<float>(),
        mlp_0_w.data_ptr<float>(),
        mlp_0_b.data_ptr<float>(),
        mlp_3_w.data_ptr<float>(),
        mlp_3_b.data_ptr<float>(),
        x.sizes()[0],
        edge_index.sizes()[1]);
    CheckCudaError(cudaGetLastError());
    CheckCudaError(cudaDeviceSynchronize());
    float out_data[64];
    CheckCudaError(cudaMemcpy(out_data, out.data_ptr<float>(), 64 * sizeof(float), cudaMemcpyDeviceToHost));
    CheckCudaError(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++)
            printf("%f ", out_data[i * 8 + j]);
        printf("\n");
    }
    return out;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("flash_gat", &flash_gat, "flash_gat");
// }
