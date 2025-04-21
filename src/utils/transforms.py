import math
import random

import torch
from torch_geometric.transforms import BaseTransform

def random_rotation_matrix(device="cpu"):
    """
    쿼터니언 --> 3×3 회전행렬
    """
    u1, u2, u3 = random.random(), random.random(), random.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1)       * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1)       * math.cos(2 * math.pi * u3)
    R = torch.tensor([
        [1-2*(q3**2+q4**2),   2*(q2*q3-q1*q4),     2*(q2*q4+q1*q3)],
        [2*(q2*q3+q1*q4),     1-2*(q2**2+q4**2),   2*(q3*q4-q1*q2)],
        [2*(q2*q4-q1*q3),     2*(q3*q4+q1*q2),     1-2*(q2**2+q3**2)]
    ], dtype=torch.float, device=device)
    return R


class SO3RotateAndJitter(BaseTransform):
    """
    cart_dir:  Nx3  방향벡터 --> R·cart_dir  (회전)  
    cart_dist: N     거리 --> 거리 그대로  
    jitter:    0.0 --> 0.02 Å
    """
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, data):
        # 1) 무작위 SO(3) 회전
        R = random_rotation_matrix(device=data.cart_dir.device)
        data.cart_dir = torch.matmul(data.cart_dir, R.T)

        # 2) 방향·거리 모두에 작은 Gaussian 노이즈 추가
        if self.jitter_std > 0:
            noise = torch.randn_like(data.cart_dir) * self.jitter_std
            data.cart_dir = torch.nn.functional.normalize(data.cart_dir + noise, dim=1)

            d_noise = torch.randn_like(data.cart_dist) * self.jitter_std
            data.cart_dist = (data.cart_dist + d_noise).clamp(min=0.0)

        return data

