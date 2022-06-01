import math
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.functional import cosine_similarity

import cross_view_ray_casting._C as _C

class ray_casting_weight_calculation_function(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, im_xy, voxel_conf, H, W):
        assert torch.is_tensor(im_xy)
        im_xy = im_xy.contiguous()
        assert torch.is_tensor(voxel_conf)
        voxel_conf = voxel_conf.contiguous()
        assert isinstance(H, int) and isinstance(W, int)

        features_weight = torch.zeros([im_xy.shape[0], H, W], dtype=im_xy.dtype, device=im_xy.device)

        _C.ray_casting_weight_calculation_forward(features_weight, im_xy, voxel_conf)

        assert not torch.isnan(features_weight).any()

        ctx.save_for_backward(im_xy, voxel_conf)

        return features_weight

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_features_weight):
        grad_features_weight = grad_features_weight.contiguous()
        im_xy, voxel_conf = ctx.saved_tensors

        grad_voxel_conf = torch.zeros_like(voxel_conf)

        _C.ray_casting_weight_calculation_backward(grad_features_weight, im_xy, grad_voxel_conf)

        return None, grad_voxel_conf, None, None


class ray_casting_sample_function(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, features, features_weight, im_xy, voxel_conf):
        assert torch.is_tensor(features)
        features = features.contiguous()
        assert torch.is_tensor(features_weight)
        features_weight = features_weight.contiguous()
        assert torch.is_tensor(im_xy)
        im_xy = im_xy.contiguous()
        assert torch.is_tensor(voxel_conf)
        voxel_conf = voxel_conf.contiguous()

        volume = torch.zeros([features.shape[0], im_xy.shape[1], features.shape[1]], dtype=features.dtype, device=features.device)

        _C.ray_casting_project_sample_forward(volume, features, features_weight, im_xy, voxel_conf)

        assert not torch.isnan(volume).any()

        ctx.save_for_backward(features, features_weight, im_xy, voxel_conf)

        return volume

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_volume):
        grad_volume = grad_volume.contiguous()
        features, features_weight, im_xy, voxel_conf = ctx.saved_tensors

        grad_features = torch.zeros_like(features)
        grad_features_weight = torch.zeros_like(features_weight)

        _C.ray_casting_project_sample_backward(grad_volume, grad_features, grad_features_weight, features, features_weight, im_xy, voxel_conf)

        return grad_features, grad_features_weight, None, None


def cross_view_ray_cast(features_2d : torch.Tensor, im_xy : torch.Tensor, unaverage_features_3d : torch.Tensor, in_scope: torch.Tensor, reduce: bool=True) -> torch.Tensor:
    '''
    parameters:
        features_2d : torch.Tensor, Shape : (n_view * C * H * W)
        im_xy : torch.Tensor, Shape : (n_view * n_voxel * 2), valid number: 0 -> H-1 and 0 -> W-1
        unaverage_features_3d : torch.Tensor, Shape : (n_view * C * n_voxel)
        in_scope : torch.Tensor, Shape : (n_voxel)
        reduce : bool
    '''
    ### do pythonic check
    assert len(features_2d.shape) == 4
    assert len(im_xy.shape) == 3
    assert len(unaverage_features_3d.shape) == 3
    assert len(in_scope.shape) == 1
    assert features_2d.shape[0] == im_xy.shape[0] and features_2d.shape[0] == unaverage_features_3d.shape[0] ## check 'n_view'
    assert im_xy.shape[1] == unaverage_features_3d.shape[2] and im_xy.shape[1] == in_scope.shape[0] ## check 'n_voxel'
    assert features_2d.shape[1] == unaverage_features_3d.shape[1] ## check 'C'
    assert im_xy.shape[2] == 2

    n_views, C, H, W = features_2d.shape
    features_mean = unaverage_features_3d.mean(dim=0, keepdim=True) / in_scope # 1 * C * n_voxel
    features_dist = cosine_similarity(unaverage_features_3d, features_mean.expand(n_views, -1, -1), dim=1) # (n_view * n_voxel) [-1, 1]
    voxel_conf = features_dist.sum(0) / in_scope # n_voxel [-1, 1]
    # voxel_conf = voxel_conf * (torch.log(in_scope) + 1.0) / (math.log(n_views) + 1.0) # n_voxel [-1, 1]
    voxel_conf = voxel_conf.exp() # n_voxel [e^-1, e]
    features_weight = ray_casting_weight_calculation_function.apply(im_xy, voxel_conf, H, W) # n_view * H * W
    features_ray = ray_casting_sample_function.apply(features_2d, features_weight, im_xy, voxel_conf) # n_view * n_voxel * C
    features_ray = features_ray.mean(dim=0)
    if reduce:
        features_ray = features_ray / in_scope.unsqueeze(1) # n_voxel * C
    return features_ray