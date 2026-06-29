
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class CNNEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 4,
        base_ch: int = 32,
        embed_dim: int = 128,
        num_classes: int | None = None,
    ):
        super().__init__()
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.block1 = ConvBlock(in_channels, ch1)
        self.pool1  = nn.MaxPool2d(2)

        self.block2 = ConvBlock(ch1, ch2)
        self.pool2  = nn.MaxPool2d(2)

        self.block3 = ConvBlock(ch2, ch3)
        self.pool3  = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.classifier = (
            nn.Linear(embed_dim, num_classes) if num_classes is not None else None
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        e1 = self.block1(x)
        x  = self.pool1(e1)

        e2 = self.block2(x)
        x  = self.pool2(e2)

        e3 = self.block3(x)
        x  = self.pool3(e3)

        emb    = self.proj(self.gap(x))
        logits = self.classifier(emb) if self.classifier is not None else None

        return emb, logits, {'E1': e1, 'E2': e2, 'E3': e3}
