#include <torch/extension.h>
#include "cpu/croco_function_cpu.hpp"
#ifndef WITHOUT_CUDA
#include "cuda/croco_function_cuda.cuh"
#endif


void ray_casting_weight_calculation_forward(
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
)
{
    if (features_weight.is_cuda())
    {
#ifndef WITHOUT_CUDA
        ray_casting_weight_calculation_forward_function_cuda(
            features_weight,
            im_xy,
            voxel_conf
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        ray_casting_weight_calculation_forward_function_cpu(
            features_weight,
            im_xy,
            voxel_conf
        );
    }
}


void ray_casting_weight_calculation_backward(
    torch::Tensor &grad_features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &grad_voxel_conf // n_voxel
)
{
    if (grad_features_weight.is_cuda())
    {
#ifndef WITHOUT_CUDA
        ray_casting_weight_calculation_backward_function_cuda(
            grad_features_weight,
            im_xy,
            grad_voxel_conf
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        ray_casting_weight_calculation_backward_function_cpu(
            grad_features_weight,
            im_xy,
            grad_voxel_conf
        );
    }
}


void ray_casting_sample_forward(
    torch::Tensor &volume, // n_view * n_voxel * C
    torch::Tensor &features, // n_view * C *  H * W
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
)
{
    if (volume.is_cuda())
    {
#ifndef WITHOUT_CUDA
        ray_casting_sample_forward_function_cuda(
            volume,
            features,
            features_weight,
            im_xy,
            voxel_conf
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        ray_casting_sample_forward_function_cpu(
            volume,
            features,
            features_weight,
            im_xy,
            voxel_conf
        );
    }
}


void ray_casting_sample_backward(
    torch::Tensor &grad_volume, // n_view * n_voxel * C
    torch::Tensor &grad_features, // n_view * C *  H * W
    torch::Tensor &grad_features_weight, // n_view * H * W
    torch::Tensor &features, // n_view * C *  H * W
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
)
{
    if (grad_volume.is_cuda())
    {
#ifndef WITHOUT_CUDA
        ray_casting_sample_backward_function_cuda(
            grad_volume,
            grad_features,
            grad_features_weight,
            features,
            features_weight,
            im_xy,
            voxel_conf
        );
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        ray_casting_sample_backward_function_cpu(
            grad_volume,
            grad_features,
            grad_features_weight,
            features,
            features_weight,
            im_xy,
            voxel_conf
        );
    }
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ray_casting_weight_calculation_forward", &ray_casting_weight_calculation_forward, "Cross-view ray-casting project weight calculation forward function warpper");
    m.def("ray_casting_weight_calculation_backward", &ray_casting_weight_calculation_backward, "Cross-view ray-casting project weight calculation backward function warpper");
    m.def("ray_casting_sample_forward", &ray_casting_sample_forward, "Cross-view ray-casting project sample forward function warpper");
    m.def("ray_casting_sample_backward", &ray_casting_sample_backward, "Cross-view ray-casting project sample backward function warpper");
}