import torch


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SESA_SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SESA_SEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 特征融合
        return x * y.expand_as(x)
    
    
class SESA(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=5):
        super(SESA, self).__init__()
        self.se_block = SESA_SEBlock(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        se_out = self.se_block(x)
        sa_out = self.spatial_attention(x)
        # 残差连接
        return x + se_out * sa_out