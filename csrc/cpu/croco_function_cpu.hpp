#pragma once
#include <torch/extension.h>

void ray_casting_weight_calculation_forward_function_cpu(
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
);


void ray_casting_weight_calculation_backward_function_cpu(
    torch::Tensor &grad_features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &grad_voxel_conf // n_voxel
);


void ray_casting_sample_forward_function_cpu(
    torch::Tensor &volume, // n_view * n_voxel * C
    torch::Tensor &features, // n_view * C *  H * W
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
);


void ray_casting_sample_backward_function_cpu(
    torch::Tensor &grad_volume, // n_view * n_voxel * C
    torch::Tensor &grad_features, // n_view * C *  H * W
    torch::Tensor &grad_features_weight, // n_view * H * W
    torch::Tensor &features, // n_view * C *  H * W
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
);
