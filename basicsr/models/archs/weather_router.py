"""
WeatherRouter: 极轻量 CNN 路由网络

3层 Conv + GAP → 1x7 logits
参数量 ~11K，GPU 推理 <0.5ms

forward() 返回 raw logits (不做 softmax)，
配合 F.cross_entropy 的 LogSumExp 数值稳定优化。
需要概率分布时: probs = torch.softmax(logits.detach(), dim=1)
"""
import torch
import torch.nn as nn


class WeatherRouter(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=4, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)  # raw logits, 不做 softmax
