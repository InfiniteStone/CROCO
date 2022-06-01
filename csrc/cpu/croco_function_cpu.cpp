#include "croco_function_cpu.hpp"

template <typename scalar_t>
void ray_casting_weight_calculation_forward_cpu_kernel(
          scalar_t* __restrict__ features_weight, // n_view * H * W
    const scalar_t* __restrict__ im_xy, // n_view * n_voxel * 2
    const scalar_t* __restrict__ voxel_conf, // n_voxel
    const size_t n_view,
    const int64_t H, const int64_t W,
    const size_t n)
{
    #pragma omp parallel for
    for (size_t n_v = 0; n_v < n_view; ++n_v)
    {
        for (size_t idx = 0; idx < n ; ++idx)
        {
            scalar_t conf_i = voxel_conf[idx];

            scalar_t w_i = im_xy[n_v * n * 2 + idx * 2 + 0];
            scalar_t h_i = im_xy[n_v * n * 2 + idx * 2 + 1];

            int64_t w_left = std::floor(w_i);
            int64_t w_right = std::ceil(w_i);
            int64_t h_top = std::floor(h_i);
            int64_t h_down = std::ceil(h_i);

            scalar_t w_left_dis = w_i - static_cast<scalar_t>(w_left);
            scalar_t w_right_dis = static_cast<scalar_t>(w_right) - w_i;
            scalar_t h_top_dis = h_i - static_cast<scalar_t>(h_top);
            scalar_t h_down_dis = static_cast<scalar_t>(h_down) - h_i;

            scalar_t weight_left_top = w_right_dis * h_down_dis;
            scalar_t weight_left_down = w_right_dis * h_top_dis;
            scalar_t weight_right_top = w_left_dis * h_down_dis;
            scalar_t weight_right_down = w_left_dis * h_top_dis;

            // from "left top" to "left down" to "right top" to "right down"
            // left top
            if (h_top >= 0 && h_top < H && w_left >= 0 && w_left < W)
            {
                features_weight[n_v * H * W + h_top * W + w_left] += conf_i * weight_left_top;
            }


            // left down
            if (h_down >= 0 && h_down < H && w_left >= 0 && w_left < W)
            {
                features_weight[n_v * H * W + h_down * W + w_left] += conf_i * weight_left_down;
            }


            // right top
            if (h_top >= 0 && h_top < H && w_right >= 0 && w_right < W)
            {
                features_weight[n_v * H * W + h_top * W + w_right] += conf_i * weight_right_top;
            }


            // right down
            if (h_down >= 0 && h_down < H && w_right >= 0 && w_right < W)
            {
                features_weight[n_v * H * W + h_down * W + w_right] += conf_i * weight_right_down;
            }
        }
    }
}


template <typename scalar_t>
void ray_casting_weight_calculation_backward_cpu_kernel(
    const scalar_t* __restrict__ grad_features_weight, // n_view * H * W
    const scalar_t* __restrict__ im_xy, // n_view * n_voxel * 2
          scalar_t* __restrict__ grad_voxel_conf, // n_voxel
    const size_t n_view,
    const int64_t H, const int64_t W,
    const size_t n)
{
    #pragma omp parallel for
    for (size_t idx = 0; idx < n ; ++idx)
    {
        scalar_t grad_conf_i = 0.0;

        for (size_t n_v = 0; n_v < n_view; ++n_v)
        {
            scalar_t w_i = im_xy[n_v * n * 2 + idx * 2 + 0];
            scalar_t h_i = im_xy[n_v * n * 2 + idx * 2 + 1];

            int64_t w_left = std::floor(w_i);
            int64_t w_right = std::ceil(w_i);
            int64_t h_top = std::floor(h_i);
            int64_t h_down = std::ceil(h_i);

            scalar_t w_left_dis = w_i - static_cast<scalar_t>(w_left);
            scalar_t w_right_dis = static_cast<scalar_t>(w_right) - w_i;
            scalar_t h_top_dis = h_i - static_cast<scalar_t>(h_top);
            scalar_t h_down_dis = static_cast<scalar_t>(h_down) - h_i;

            scalar_t weight_left_top = w_right_dis * h_down_dis;
            scalar_t weight_left_down = w_right_dis * h_top_dis;
            scalar_t weight_right_top = w_left_dis * h_down_dis;
            scalar_t weight_right_down = w_left_dis * h_top_dis;

            // from "left top" to "left down" to "right top" to "right down"
            // left top
            if (h_top >= 0 && h_top < H && w_left >= 0 && w_left < W)
            {
                grad_conf_i += grad_features_weight[n_v * H * W + h_top * W + w_left] * weight_left_top;
            }


            // left down
            if (h_down >= 0 && h_down < H && w_left >= 0 && w_left < W)
            {
                grad_conf_i += grad_features_weight[n_v * H * W + h_down * W + w_left] * weight_left_down;
            }


            // right top
            if (h_top >= 0 && h_top < H && w_right >= 0 && w_right < W)
            {
                grad_conf_i += grad_features_weight[n_v * H * W + h_top * W + w_right] * weight_right_top;
            }


            // right down
            if (h_down >= 0 && h_down < H && w_right >= 0 && w_right < W)
            {
                grad_conf_i += grad_features_weight[n_v * H * W + h_down * W + w_right] * weight_right_down;
            }
        }

        grad_voxel_conf[idx] = grad_conf_i;
    }
}


template <typename scalar_t>
void ray_casting_sample_forward_cpu_kernel(
          scalar_t* __restrict__ volume, // n_view * n_voxel * C
    const scalar_t* __restrict__ features, // n_view * C * H * W
    const scalar_t* __restrict__ features_weight, // n_view * H * W
    const scalar_t* __restrict__ im_xy, // n_view * n_voxel * 2
    const scalar_t* __restrict__ voxel_conf, // n_voxel
    const size_t C, const size_t n_view,
    const int64_t H, const int64_t W,
    const size_t n)
{
    #pragma omp parallel for
    for (size_t idx = 0; idx < n ; ++idx)
    {
        scalar_t conf_i = voxel_conf[idx];

        for (size_t n_v = 0; n_v < n_view; ++n_v)
        {
            scalar_t w_i = im_xy[n_v * n * 2 + idx * 2 + 0];
            scalar_t h_i = im_xy[n_v * n * 2 + idx * 2 + 1];

            int64_t w_left = std::floor(w_i);
            int64_t w_right = std::ceil(w_i);
            int64_t h_top = std::floor(h_i);
            int64_t h_down = std::ceil(h_i);

            scalar_t w_left_dis = w_i - static_cast<scalar_t>(w_left);
            scalar_t w_right_dis = static_cast<scalar_t>(w_right) - w_i;
            scalar_t h_top_dis = h_i - static_cast<scalar_t>(h_top);
            scalar_t h_down_dis = static_cast<scalar_t>(h_down) - h_i;

            scalar_t weight_left_top = w_right_dis * h_down_dis;
            scalar_t weight_left_down = w_right_dis * h_top_dis;
            scalar_t weight_right_top = w_left_dis * h_down_dis;
            scalar_t weight_right_down = w_left_dis * h_top_dis;

            // from "left top" to "left down" to "right top" to "right down"
            // left top
            if (h_top >= 0 && h_top < H && w_left >= 0 && w_left < W)
            {
                if (features_weight[n_v * H * W + h_top * W + w_left] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        volume[idx * C + i] += features[n_v * C * H * W + i * H * W + h_top * W + w_left]
                            * conf_i / features_weight[n_v * H * W + h_top * W + w_left]
                            * weight_left_top;
                    }
                }
            }


            // left down
            if (h_down >= 0 && h_down < H && w_left >= 0 && w_left < W)
            {
                if (features_weight[n_v * H * W + h_down * W + w_left] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        volume[idx * C + i] += features[n_v * C * H * W + i * H * W + h_down * W + w_left]
                            * conf_i / features_weight[n_v * H * W + h_down * W + w_left]
                            * weight_left_down;
                    }
                }
            }


            // right top
            if (h_top >= 0 && h_top < H && w_right >= 0 && w_right < W)
            {
                if (features_weight[n_v * H * W + h_top * W + w_right] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        volume[idx * C + i] += features[n_v * C * H * W + i * H * W + h_top * W + w_right]
                            * conf_i / features_weight[n_v * H * W + h_top * W + w_right]
                            * weight_right_down;
                    }
                }
            }


            // right down
            if (h_down >= 0 && h_down < H && w_right >= 0 && w_right < W)
            {
                if (features_weight[n_v * H * W + h_down * W + w_right] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        volume[idx * C + i] += features[n_v * C * H * W + i * H * W + h_down * W + w_right]
                            * conf_i / features_weight[n_v * H * W + h_down * W + w_right]
                            * weight_right_top;
                    }
                }
            }
        }
    }
}


template <typename scalar_t>
void ray_casting_sample_backward_cpu_kernel(
    const scalar_t* __restrict__ grad_volume, // n_view * n_voxel * C
          scalar_t* __restrict__ grad_features, // n_view * C * H * W
          scalar_t* __restrict__ grad_features_weight, // n_view * H * W
    const scalar_t* __restrict__ features, // n_view * C * H * W
    const scalar_t* __restrict__ features_weight, // n_view * H * W
    const scalar_t* __restrict__ im_xy, // n_view * n_voxel * 2
    const scalar_t* __restrict__ voxel_conf, // n_voxel
    const size_t C, const size_t n_view,
    const int64_t H, const int64_t W,
    const size_t n)
{
    #pragma omp parallel for
    for (size_t n_v = 0; n_v < n_view; ++n_v)
    {
        for (size_t idx = 0; idx < n ; ++idx)
        {
            scalar_t conf_i = voxel_conf[idx];

            scalar_t w_i = im_xy[n_v * n * 2 + idx * 2 + 0];
            scalar_t h_i = im_xy[n_v * n * 2 + idx * 2 + 1];

            int64_t w_left = std::floor(w_i);
            int64_t w_right = std::ceil(w_i);
            int64_t h_top = std::floor(h_i);
            int64_t h_down = std::ceil(h_i);

            scalar_t w_left_dis = w_i - static_cast<scalar_t>(w_left);
            scalar_t w_right_dis = static_cast<scalar_t>(w_right) - w_i;
            scalar_t h_top_dis = h_i - static_cast<scalar_t>(h_top);
            scalar_t h_down_dis = static_cast<scalar_t>(h_down) - h_i;

            scalar_t weight_left_top = w_right_dis * h_down_dis;
            scalar_t weight_left_down = w_right_dis * h_top_dis;
            scalar_t weight_right_top = w_left_dis * h_down_dis;
            scalar_t weight_right_down = w_left_dis * h_top_dis;

            // from "left top" to "left down" to "right top" to "right down"
            // left top
            if (h_top >= 0 && h_top < H && w_left >= 0 && w_left < W)
            {
                if (features_weight[n_v * H * W + h_top * W + w_left] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        grad_features[n_v * C * H * W + i * H * W + h_top * W + w_left] += grad_volume[idx * C + i]
                            * conf_i / features_weight[n_v * H * W + h_top * W + w_left] * weight_left_top;
                        grad_features_weight[n_v * H * W + h_top * W + w_left] += -grad_volume[idx * C + i]
                            * conf_i / (features_weight[n_v * H * W + h_top * W + w_left] * features_weight[n_v * H * W + h_top * W + w_left])
                            * weight_left_top;
                    }
                }
            }


            // left down
            if (h_down >= 0 && h_down < H && w_left >= 0 && w_left < W)
            {
                if (features_weight[n_v * H * W + h_down * W + w_left] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        grad_features[n_v * C * H * W + i * H * W + h_down * W + w_left] += grad_volume[idx * C + i]
                            * conf_i / features_weight[n_v * H * W + h_down * W + w_left]
                            * weight_left_down;
                        grad_features_weight[n_v * H * W + h_down * W + w_left] += -grad_volume[idx * C + i]
                            * conf_i / (features_weight[n_v * H * W + h_down * W + w_left] * features_weight[n_v * H * W + h_down * W + w_left])
                            * weight_left_down;
                    }
                }
            }


            // right top
            if (h_top >= 0 && h_top < H && w_right >= 0 && w_right < W)
            {
                if (features_weight[n_v * H * W + h_top * W + w_right] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        grad_features[n_v * C * H * W + i * H * W + h_top * W + w_right] += grad_volume[idx * C + i]
                            * conf_i / features_weight[n_v * H * W + h_top * W + w_right]
                            * weight_right_top;
                        grad_features_weight[n_v * H * W + h_top * W + w_right] += -grad_volume[idx * C + i]
                            * conf_i / (features_weight[n_v * H * W + h_top * W + w_right] * features_weight[n_v * H * W + h_top * W + w_right])
                            * weight_right_top;
                    }
                }
            }


            // right down
            if (h_down >= 0 && h_down < H && w_right >= 0 && w_right < W)
            {
                if (features_weight[n_v * H * W + h_down * W + w_right] > static_cast<scalar_t>(0.0))
                {
                    for (size_t i = 0; i < C; ++i)
                    {
                        grad_features[n_v * C * H * W + i * H * W + h_down * W + w_right] += grad_volume[idx * C + i]
                            * conf_i / features_weight[n_v * H * W + h_down * W + w_right]
                            * weight_right_down;
                        grad_features_weight[n_v * H * W + h_down * W + w_right] += -grad_volume[idx * C + i]
                            * conf_i / (features_weight[n_v * H * W + h_down * W + w_right] * features_weight[n_v * H * W + h_down * W + w_right])
                            * weight_right_down;
                    }
                }
            }
        }
    }
}


void ray_casting_weight_calculation_forward_function_cpu(
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
) 
{
    // all check and initilization has been done in python
    size_t n_view = features_weight.size(0);
    size_t img_h = features_weight.size(1);
    size_t img_w = features_weight.size(2);
    size_t n = voxel_conf.size(0);

    AT_DISPATCH_FLOATING_TYPES(features_weight.type(), "ray_casting_weight_calculation_forward_cpu_kernel", ([&] {
        ray_casting_weight_calculation_forward_cpu_kernel<scalar_t>(
            features_weight.data_ptr<scalar_t>(),
            im_xy.data_ptr<scalar_t>(),
            voxel_conf.data_ptr<scalar_t>(),
            n_view,
            img_h, img_w,
            n
        );
    }));

}


void ray_casting_weight_calculation_backward_function_cpu(
    torch::Tensor &grad_features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &grad_voxel_conf // n_voxel
) 
{
    // all check and initilization has been done in python
    size_t n_view = grad_features_weight.size(0);
    size_t img_h = grad_features_weight.size(1);
    size_t img_w = grad_features_weight.size(2);
    size_t n = grad_voxel_conf.size(0);

    AT_DISPATCH_FLOATING_TYPES(grad_features_weight.type(), "ray_casting_weight_calculation_backward_cpu_kernel", ([&] {
        ray_casting_weight_calculation_backward_cpu_kernel<scalar_t>(
            grad_features_weight.data_ptr<scalar_t>(),
            im_xy.data_ptr<scalar_t>(),
            grad_voxel_conf.data_ptr<scalar_t>(),
            n_view,
            img_h, img_w,
            n
        );
    }));

}


void ray_casting_sample_forward_function_cpu(
    torch::Tensor &volume, // n_voxel * C
    torch::Tensor &features, // n_view * C *  H * W
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
) 
{
    // all check and initilization has been done in python
    size_t n_view = features.size(0);
    size_t C = features.size(1);
    size_t img_h = features.size(2);
    size_t img_w = features.size(3);
    size_t n = voxel_conf.size(0);

    AT_DISPATCH_FLOATING_TYPES(volume.type(), "ray_casting_sample_forward_cpu_kernel", ([&] {
        ray_casting_sample_forward_cpu_kernel<scalar_t>(
            volume.data_ptr<scalar_t>(),
            features.data_ptr<scalar_t>(),
            features_weight.data_ptr<scalar_t>(),
            im_xy.data_ptr<scalar_t>(),
            voxel_conf.data_ptr<scalar_t>(),
            C, n_view,
            img_h, img_w,
            n
        );
    }));

}


void ray_casting_sample_backward_function_cpu(
    torch::Tensor &grad_volume, // n_voxel * C
    torch::Tensor &grad_features, // n_view * C *  H * W
    torch::Tensor &grad_features_weight, // n_view * H * W
    torch::Tensor &features, // n_view * C *  H * W
    torch::Tensor &features_weight, // n_view * H * W
    torch::Tensor &im_xy, // n_view * n_voxel * 2
    torch::Tensor &voxel_conf // n_voxel
) 
{
    // all check and initilization has been done in python
    size_t n_view = features.size(0);
    size_t C = features.size(1);
    size_t img_h = features.size(2);
    size_t img_w = features.size(3);
    size_t n = voxel_conf.size(0);

    AT_DISPATCH_FLOATING_TYPES(grad_volume.type(), "ray_casting_sample_backward_cpu_kernel", ([&] {
        ray_casting_sample_backward_cpu_kernel<scalar_t>(
            grad_volume.data_ptr<scalar_t>(),
            grad_features.data_ptr<scalar_t>(),
            grad_features_weight.data_ptr<scalar_t>(),
            features.data_ptr<scalar_t>(),
            features_weight.data_ptr<scalar_t>(),
            im_xy.data_ptr<scalar_t>(),
            voxel_conf.data_ptr<scalar_t>(),
            C, n_view,
            img_h, img_w,
            n
        );
    }));

}
