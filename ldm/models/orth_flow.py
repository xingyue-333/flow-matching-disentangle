import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


def sample_conditional_path(z_0, t):
    """Linear path used by the velocity-matching objective.

    t is broadcast to image shape. Noise is 1 and data is 0:
    z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
    u = (1 - 1e-5) * z_1 - z_0
    """
    t = t.view(-1, 1, 1, 1)
    z_1 = torch.randn_like(z_0)
    z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
    u = (1 - 1e-5) * z_1 - z_0
    return z_t, u, t


def global_orth_loss(v_decomposed):
    """Off-diagonal mean absolute correlation of per-token velocity fields.

    v_decomposed: [B, K, C, H, W]
    """
    B, K, C, H, W = v_decomposed.shape
    v_long = v_decomposed.reshape(B, K, -1)
    v_long_norm = F.normalize(v_long, p=2, dim=-1)
    global_gram = torch.bmm(v_long_norm, v_long_norm.transpose(1, 2))
    identity = torch.eye(K, device=v_decomposed.device).unsqueeze(0)
    return (global_gram * (1 - identity)).abs().mean()


class FlowSplitterHead(nn.Module):
    def __init__(self, feature_channels, token_dim, K):
        super().__init__()
        self.K = K
        self.to_q = nn.Conv2d(feature_channels, 64, kernel_size=1)
        self.to_k = nn.Linear(token_dim, 64)

    def forward(self, x_feat, context, v_total):
        """
        x_feat: UNet feature map [B, C, H, W]
        context: concept tokens [B, K, D]
        v_total: predicted velocity field [B, 3, H, W]
        """
        B, C, H, W = x_feat.shape
        q = self.to_q(x_feat).reshape(B, 64, H * W).permute(0, 2, 1)
        k = self.to_k(context)
        attn = einsum("b n d, b k d -> b k n", q, k) * (64 ** -0.5)
        weights = attn.softmax(dim=1)
        weights = weights.reshape(B, self.K, H, W)
        v_decomposed = v_total.unsqueeze(1) * weights.unsqueeze(2)
        return v_decomposed
