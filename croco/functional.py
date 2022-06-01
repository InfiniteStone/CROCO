import math
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.functional import cosine_similarity

import croco._C as _C

class ray_casting_weight_calculation_function(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, im_xy, voxel_conf, H, W):
        assert torch.is_tensor(im_xy)
        im_xy = im_xy.contiguous()
        assert torch.is_tensor(voxel_conf)
        voxel_conf = voxel_conf.contiguous()
        assert isinstance(H, int) and isinstance(W, int)

        if im_xy.device == torch.device('cpu'):
            assert im_xy.dtype == torch.float32 or im_xy.dtype == torch.float64
        if voxel_conf.device == torch.device('cpu'):
            assert voxel_conf.dtype == torch.float32 or voxel_conf.dtype == torch.float64

        features_weight = torch.zeros([im_xy.shape[0], H, W], dtype=im_xy.dtype, device=im_xy.device)

        _C.ray_casting_weight_calculation_forward(features_weight, im_xy, voxel_conf)

        # assert not torch.isnan(features_weight).any()

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

        if features.device == torch.device('cpu'):
            assert features.dtype == torch.float32 or features.dtype == torch.float64
        if features_weight.device == torch.device('cpu'):
            assert features_weight.dtype == torch.float32 or features_weight.dtype == torch.float64
        if im_xy.device == torch.device('cpu'):
            assert im_xy.dtype == torch.float32 or im_xy.dtype == torch.float64
        if voxel_conf.device == torch.device('cpu'):
            assert voxel_conf.dtype == torch.float32 or voxel_conf.dtype == torch.float64

        volume = torch.zeros([im_xy.shape[1], features.shape[1]], dtype=features.dtype, device=features.device)

        _C.ray_casting_sample_forward(volume, features, features_weight, im_xy, voxel_conf)

        # assert not torch.isnan(volume).any()

        ctx.save_for_backward(features, features_weight, im_xy, voxel_conf)

        return volume

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_volume):
        grad_volume = grad_volume.contiguous()
        features, features_weight, im_xy, voxel_conf = ctx.saved_tensors

        grad_features = torch.zeros_like(features)
        grad_features_weight = torch.zeros_like(features_weight)

        _C.ray_casting_sample_backward(grad_volume, grad_features, grad_features_weight, features, features_weight, im_xy, voxel_conf)

        return grad_features, grad_features_weight, None, None


def crossview_coorperative_reasoning(features_2d : torch.Tensor, im_xy : torch.Tensor, unaverage_features_3d : torch.Tensor, in_scope: torch.Tensor) -> torch.Tensor:
    '''
    parameters:
        features_2d : torch.Tensor, Shape : (n_view * C * H * W) / (batch_size * n_view * C * H * W)
        im_xy : torch.Tensor, Shape : (n_view * n_voxel * 2) / (batch_size * n_view * n_voxel * 2), valid number: 0 -> H-1 and 0 -> W-1
        unaverage_features_3d : torch.Tensor, Shape : (n_view * C * n_voxel) / (batch_size * n_view * C * n_voxel)
        in_scope : torch.Tensor, Shape : (n_voxel) /(batch_size * n_voxel)
    '''
    def croco_batch(feat_2d, xy, unaverage_feat_3d, in_sco):
        n_views, C, H, W = feat_2d.shape
        features_mean = unaverage_feat_3d.mean(dim=0) # C * n_voxel
        voxel_conf = 0.0
        for unaverage_feat_3d_view in unaverage_feat_3d: # C * n_voxel
            features_dist = cosine_similarity(unaverage_feat_3d_view, features_mean, dim=0) # (n_voxel) [-1, 1]
            voxel_conf += features_dist # n_voxel [-1, 1]
        del features_dist, features_mean
        # voxel_conf = (voxel_conf / in_sco).exp() # n_voxel [e^-1, e^1]
        voxel_conf = voxel_conf.exp() # n_voxel [e^-n, e^n]
        feat_weight = ray_casting_weight_calculation_function.apply(xy, voxel_conf, H, W) # n_view * H * W
        feat_ray = ray_casting_sample_function.apply(feat_2d, feat_weight, xy, voxel_conf) # n_voxel * C
        del feat_weight, voxel_conf
        feat_ray = feat_ray / in_sco.unsqueeze(1) # n_voxel * C
        return feat_ray

    ### do pythonic check
    if len(features_2d.shape) == 4: # single batch
        assert len(im_xy.shape) == 3
        assert len(unaverage_features_3d.shape) == 3
        assert len(in_scope.shape) == 1
        assert features_2d.shape[0] == im_xy.shape[0] and features_2d.shape[0] == unaverage_features_3d.shape[0] ## check 'n_view'
        assert im_xy.shape[1] == unaverage_features_3d.shape[2] and im_xy.shape[1] == in_scope.shape[0] ## check 'n_voxel'
        assert features_2d.shape[1] == unaverage_features_3d.shape[1] ## check 'C'
        assert im_xy.shape[2] == 2
        return croco_batch(features_2d, im_xy, unaverage_features_3d, in_scope)

    elif len(features_2d.shape) == 5: # multiple batch
        assert len(im_xy.shape) == 4
        assert len(unaverage_features_3d.shape) == 4
        assert len(in_scope.shape) == 2
        assert features_2d.shape[1] == im_xy.shape[1] and features_2d.shape[1] == unaverage_features_3d.shape[1] ## check 'n_view'
        assert im_xy.shape[2] == unaverage_features_3d.shape[3] and im_xy.shape[2] == in_scope.shape[1] ## check 'n_voxel'
        assert features_2d.shape[2] == unaverage_features_3d.shape[2] ## check 'C'
        assert im_xy.shape[3] == 2
        features_ray = []
        for (feat_2d, xy, unaverage_feat_3d, in_sco) in zip(features_2d, im_xy, unaverage_features_3d, in_scope):
            features_ray.append(croco_batch(feat_2d, xy, unaverage_feat_3d, in_sco))
        return torch.stack(features_ray)
