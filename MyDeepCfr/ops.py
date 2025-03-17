import torch.nn as nn


class DenseResidualBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseResidualBlock, self).__init__()
        self.dense1 = nn.Linear(input_dim, units)
        self.dense2 = nn.Linear(units, units)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.layer_norm = nn.LayerNorm(units)

        if input_dim != units:
            self.shortcut = nn.Linear(input_dim, units)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.dense1(x)
        x = self.leaky_relu(x)

        x = self.dense2(x)
        x = self.leaky_relu(x)

        x = x + shortcut
        x = self.leaky_relu(x)
        x = self.layer_norm(x)

        return x

