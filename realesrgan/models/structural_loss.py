import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """Differentiable Structural Similarity (SSIM) Loss for Super-Resolution.
    Optimizes structural fidelity and contrast consistency to boost SSIM metric.
    """

    def __init__(self, loss_weight=1.0, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.register_buffer('window', self._create_window(window_size, self.channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, pred, target):
        if pred.size(1) != self.channel or self.window.device != pred.device or self.window.dtype != pred.dtype:
            self.channel = pred.size(1)
            self.window = self._create_window(self.window_size, self.channel).to(pred.device).type(pred.dtype)

        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            ssim_val = ssim_map.mean()
        else:
            ssim_val = ssim_map.mean(1).mean(1).mean(1)

        loss = 1.0 - ssim_val
        return self.loss_weight * loss

