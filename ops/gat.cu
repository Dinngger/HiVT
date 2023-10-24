#include <torch/extension.h>

__device__ static inline float atomicMax(float *address, const float val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, __float_as_uint(max(val, __uint_as_float(assumed))));
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);
}

template <typename scalar_t, uint32_t N_DIMS>
__global__ void scatter_max_kernel(
  const scalar_t* __restrict__ src,
  const int64_t* __restrict__ index,
  scalar_t* __restrict__ out,
  int src_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < src_size) {
    _Pragma("unroll")
    for (uint32_t i = 0; i < N_DIMS; ++i) {
      atomicMax(out + index[idx] * N_DIMS + i, src[idx * N_DIMS + i]);
    }
  }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor scatter_max(
  torch::Tensor src,
  torch::Tensor index,
  int dim,
  int dim_size) {
  CHECK_INPUT(src);
  CHECK_INPUT(index);
  dim = dim < 0 ? src.dim() + dim : dim;
  auto size = src.sizes().vec();
  size[dim] = dim_size;

  assert(src.sizes()[1] == 8);
  assert(src.type().scalarType() == at::ScalarType::Float);
  assert(index.type().scalarType() == at::ScalarType::Long);

  auto result = src.new_zeros(size);
  result.fill_(std::numeric_limits<float>::lowest());

  const int threads = 1024;
  const int blocks = src.sizes()[0] / threads + 1;
  scatter_max_kernel<float, 8><<<blocks, threads>>>(
    src.data<float>(),
    index.data<int64_t>(),
    result.data<float>(),
    src.sizes()[0]);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max", &scatter_max, "scatter_max");
}
