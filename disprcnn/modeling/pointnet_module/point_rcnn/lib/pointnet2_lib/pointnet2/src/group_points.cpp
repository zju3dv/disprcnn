#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "group_points_gpu.h"

//extern THCState *state;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int group_points_grad_wrapper_fast(int b, int c, int n, int npoints, int nsample, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    float *grad_points = grad_points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    const float *grad_out = grad_out_tensor.data<float>();

//    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    group_points_grad_kernel_launcher_fast(b, c, n, npoints, nsample, grad_out, idx, grad_points, c10::cuda::getCurrentCUDAStream());
    return 1;
}


int group_points_wrapper_fast(int b, int c, int n, int npoints, int nsample, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor) {
    CHECK_INPUT(points_tensor);
    CHECK_INPUT(idx_tensor);
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

//    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    group_points_kernel_launcher_fast(b, c, n, npoints, nsample, points, idx, out, c10::cuda::getCurrentCUDAStream());
    return 1;
}
