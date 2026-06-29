
import torch
import torch.nn as nn
import torch.nn.functional as F

class RCLLayer(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, T: int = 3):
        super().__init__()
        self.T   = T
        self.w_ff = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.w_rc = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bns  = nn.ModuleList([nn.BatchNorm2d(out_ch) for _ in range(T)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ff = self.w_ff(x)
        h  = F.relu(self.bns[0](ff), inplace=True)
        for t in range(1, self.T):
            h = F.relu(self.bns[t](ff + self.w_rc(h)), inplace=True)
        return h

class RCLBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, T: int = 3):
        super().__init__()
        self.rcl  = RCLLayer(in_ch, out_ch, T)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.rcl(x)
        out  = self.pool(feat)
        return feat, out

class RCLEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 7,
        base_ch: int = 32,
        embed_dim: int = 128,
        rcl_T: int = 3,
        num_classes: int | None = None,
    ):
        super().__init__()
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4

        self.rcl1 = RCLBlock(in_channels, ch1, rcl_T)
        self.rcl2 = RCLBlock(ch1, ch2, rcl_T)
        self.rcl3 = RCLBlock(ch2, ch3, rcl_T)

        self.gap  = nn.AdaptiveAvgPool2d(1)
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
        f1, x = self.rcl1(x)
        f2, x = self.rcl2(x)
        f3, x = self.rcl3(x)

        emb    = self.proj(self.gap(x))
        logits = self.classifier(emb) if self.classifier is not None else None

        return emb, logits, {'f1': f1, 'f2': f2, 'f3': f3}
