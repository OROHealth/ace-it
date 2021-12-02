from utils import *


def simple_cnn():
    cnn = sequential(
        (nn.Conv2d(in_channels=3, out_channels=8,
                   kernel_size=5,
                   stride=2,
                   padding=5//2)),  # 100
        (nn.ReLU()),
        (nn.BatchNorm2d(8)),

        (nn.Conv2d(in_channels=8, out_channels=16,
                   kernel_size=5,
                   stride=2,
                   padding=5//2)),  # 50
        (nn.ReLU()),
        (nn.BatchNorm2d(16)),

        (nn.Conv2d(in_channels=16, out_channels=32,
                   kernel_size=5,
                   stride=2,
                   padding=5//2)),  # 25
        (nn.ReLU()),
        (nn.BatchNorm2d(32)),

        (nn.Conv2d(in_channels=32, out_channels=64,
                   kernel_size=5,
                   stride=2,
                   padding=5//2)),  # 13
        (nn.ReLU()),
        (nn.BatchNorm2d(64)),

        (nn.Conv2d(in_channels=64, out_channels=128,
                   kernel_size=5,
                   stride=2,
                   padding=5//2)),  # 7
        (nn.ReLU()),
        (nn.BatchNorm2d(128)),

        (nn.Conv2d(in_channels=128, out_channels=256,
                   kernel_size=3,
                   stride=2,
                   padding=3//2)),  # 4
        (nn.ReLU()),
        (nn.BatchNorm2d(256)),

        (nn.Conv2d(in_channels=256, out_channels=512,
                   kernel_size=3,
                   stride=2,
                   padding=3//2)),  # 2
        (nn.ReLU()),
        (nn.BatchNorm2d(512)),

        (nn.Conv2d(in_channels=512, out_channels=3,
                   kernel_size=3,
                   stride=2,
                   padding=3//2)),  # 1

        (Flatten())
    )
    return cnn
