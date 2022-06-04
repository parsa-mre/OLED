from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=0,
        bias=True,
        batch_norm=False,
        dropout=None,
        activation=nn.LeakyReLU(0.2),
        upsample=False
    ):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if upsample:
            self.upsample = nn.UpsamplingNearest2d(scale_factor = upsample)
        else:
            self.upsample = False

        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.upsample:
            x = self.upsample(x)
        if self.bn:
            x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 64, (5, 5), (2, 2), 2, batch_norm=True),
            ConvBlock(64, 96, (5, 5), (2, 2), 2, batch_norm=True),
            ConvBlock(96, 128, (5, 5), (1, 1), 2, batch_norm=True),
            nn.Flatten(),
        )

        self.latent_fc = nn.Linear(6272, 6272)

        self.decoder = nn.Sequential(
            ConvBlock(128, 128, (5, 5), (1, 1), 'same', upsample=(2, 2)),
            ConvBlock(128, 96, (5, 5), (1, 1), 'same', upsample=(2, 2)),
            ConvBlock(96, 64, (5, 5), (1, 1), 'same'),
            ConvBlock(64, 1, (5, 5), (1, 1), 'same')
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.latent_fc(x)
        x = x.reshape(-1, 128, 7, 7)
        x = self.decoder(x)
        return x



class MaskModule(nn.Module):
    def __init__(self):
        super(MaskModule, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 32, (5, 5), (2, 2), 2),
            ConvBlock(32, 48, (5, 5), (2, 2), 2, batch_norm=True),
        )
        
        self.decoder = nn.Sequential(
            ConvBlock(48, 48, (5, 5), (1, 1), 'same', upsample=(2, 2)),
            ConvBlock(48, 32, (5, 5), (1, 1), 'same', upsample=(2, 2)),
            ConvBlock(32, 1, (5, 5), (1, 1), 'same'),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x