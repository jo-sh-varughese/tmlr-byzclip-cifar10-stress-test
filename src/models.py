"""Model architectures.

IMPORTANT — documented gap (see results_summary.md "Known Gaps" section): the source
paper's exact MNIST CNN/MLP architectures are not fully specified in the text available
to us. The architectures below are reasonable, standard, small choices for MNIST/CIFAR-10
federated-learning benchmarks, NOT a verified reproduction of the paper's exact layer
sizes. Parameter counts are reported so any future comparison against the paper's real
numbers (once available) is straightforward.

The CIFAR-10 "SmallCNN" is likewise built fresh for this study -- the task prompt referred
to a prior "OTCD" implementation with 227,594 parameters, but no such prior code or cached
data exists anywhere on this machine (verified by search before starting). This SmallCNN
is a new, independently-sized small CNN; its actual parameter count is reported below and
is NOT expected to equal 227,594.
"""

import torch
import torch.nn as nn


class MNIST_MLP(nn.Module):
    """Simple 2-hidden-layer MLP: 784 -> 200 -> 200 -> 10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MNIST_CNN(nn.Module):
    """Simple 2-conv-layer CNN: conv(1->16)-conv(16->32)-fc(1568->128)-fc(128->10)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10, built fresh for this study (see module docstring)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name, cls in [("MNIST_MLP", MNIST_MLP), ("MNIST_CNN", MNIST_CNN), ("SmallCNN", SmallCNN)]:
        m = cls()
        print(f"{name}: {count_params(m)} parameters")
