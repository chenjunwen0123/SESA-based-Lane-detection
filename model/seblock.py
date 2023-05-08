import torch


# Squeeze-and-Excitation
class SEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction_ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)

# DAN
class DualSEBlock(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super(DualSEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv1 = torch.nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.conv3 = torch.nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.conv4 = torch.nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        # global path
        y = self.avg_pool(x).view(b, c)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        # local path
        z = self.conv3(x)
        z = self.relu(z)
        z = self.conv4(z)
        z = self.sigmoid(z)
        return x * y.expand_as(x) + x * z.expand_as(x)


# DSE
class DoubleSEBlock(torch.nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(DoubleSEBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction_ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels),
            torch.nn.Sigmoid()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction_ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y1).view(b, c, 1, 1)
        y1 = x * y1.expand_as(x)

        y2 = self.avg_pool(y1).view(b, c)
        y2 = self.fc2(y2).view(b, c, 1, 1)
        y2 = y1 * y2.expand_as(y1)

        return x + y2 * x
