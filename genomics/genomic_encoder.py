
import torch
import torch.nn as nn

class SNVBranch(nn.Module):
    def __init__(self, in_dim: int = 20, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CNVBranch(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GenomicEncoder(nn.Module):

    def __init__(
        self,
        snv_dim: int = 716,
        cnv_dim: int = 4,
        branch_dim: int = 64,
        out_dim: int = 128,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.branch_dim  = branch_dim
        self.snv_branch = SNVBranch(snv_dim, branch_dim)
        self.cnv_branch = CNVBranch(cnv_dim, branch_dim)

        fused_dim = branch_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.classifier = (
            nn.Linear(out_dim, num_classes) if num_classes is not None else None
        )

    def forward(
        self,
        snv: torch.Tensor,
        cnv: torch.Tensor,
        use_snv: bool = True,
        use_cnv: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        f_snv = (self.snv_branch(snv) if use_snv
                 else snv.new_zeros(snv.size(0), self.branch_dim))
        f_cnv = (self.cnv_branch(cnv) if use_cnv
                 else cnv.new_zeros(cnv.size(0), self.branch_dim))
        fused = torch.cat([f_snv, f_cnv], dim=1)
        emb = self.fusion(fused)

        logits = self.classifier(emb) if self.classifier is not None else None
        return emb, logits
