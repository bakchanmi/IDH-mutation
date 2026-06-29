
import torch
import torch.nn as nn
import torch.nn.functional as F

from mri.cnn_encoder      import CNNEncoder
from mri.rcl_encoder      import RCLEncoder
from genomics.genomic_encoder import GenomicEncoder

class MLPHead(nn.Module):

    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        h1, h2 = in_dim // 2, in_dim // 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class LateFusionClassifier(nn.Module):

    def __init__(
        self,
        cnn_in_ch:    int   = 4,
        rcl_in_ch:    int   = 7,
        base_ch:      int   = 16,
        snv_dim:      int   = 64,
        cnv_dim:      int   = 4,
        embed_dim:    int   = 64,
        num_classes:  int   = 2,
        modal_drop_p: float = 0.0,
        ablation:     str   = 'full',
    ):
        super().__init__()
        self.modal_drop_p = modal_drop_p
        self.ablation     = ablation
        self.embed_dim    = embed_dim
        self.mri_dim      = base_ch * 7 * 2

        self.cnn_enc = CNNEncoder(
            in_channels=cnn_in_ch, base_ch=base_ch, embed_dim=embed_dim
        )
        self.rcl_enc = RCLEncoder(
            in_channels=rcl_in_ch, base_ch=base_ch, embed_dim=embed_dim, rcl_T=3
        )
        self.gen_enc = GenomicEncoder(
            snv_dim=snv_dim, cnv_dim=cnv_dim,
            branch_dim=embed_dim // 2, out_dim=embed_dim,
        )
        fused_dim = base_ch * 7 + base_ch * 7 + embed_dim
        self.fuse_norm = nn.BatchNorm1d(fused_dim)
        self.head = MLPHead(in_dim=fused_dim, num_classes=num_classes)

    def forward(
        self,
        full: torch.Tensor,
        roi:  torch.Tensor,
        snv:  torch.Tensor,
        cnv:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, cnn_feats = self.cnn_enc(full)
        _, _, rcl_feats = self.rcl_enc(roi)

        e1 = F.adaptive_avg_pool2d(cnn_feats['E1'], 1).flatten(1)
        e2 = F.adaptive_avg_pool2d(cnn_feats['E2'], 1).flatten(1)
        e3 = F.adaptive_avg_pool2d(cnn_feats['E3'], 1).flatten(1)

        f1 = F.adaptive_avg_pool2d(rcl_feats['f1'], 1).flatten(1)
        f2 = F.adaptive_avg_pool2d(rcl_feats['f2'], 1).flatten(1)
        f3 = F.adaptive_avg_pool2d(rcl_feats['f3'], 1).flatten(1)

        if self.ablation == 'mri_only':
            gen_emb, _ = self.gen_enc(snv, cnv)
            gen_emb = torch.zeros_like(gen_emb)
        elif self.ablation == 'genomic_only':
            gen_emb, _ = self.gen_enc(snv, cnv)
            e1 = torch.zeros_like(e1); e2 = torch.zeros_like(e2); e3 = torch.zeros_like(e3)
            f1 = torch.zeros_like(f1); f2 = torch.zeros_like(f2); f3 = torch.zeros_like(f3)
        elif self.ablation == 'mri_snv':
            gen_emb, _ = self.gen_enc(snv, cnv, use_cnv=False)
        elif self.ablation == 'mri_cnv':
            gen_emb, _ = self.gen_enc(snv, cnv, use_snv=False)
        else:
            gen_emb, _ = self.gen_enc(snv, cnv)

        if self.training and self.modal_drop_p > 0 and self.ablation != 'genomic_only':
            r = torch.rand(1).item()
            if r < self.modal_drop_p:
                e1 = torch.zeros_like(e1)
                e2 = torch.zeros_like(e2)
                e3 = torch.zeros_like(e3)
            elif r < 2 * self.modal_drop_p:
                f1 = torch.zeros_like(f1)
                f2 = torch.zeros_like(f2)
                f3 = torch.zeros_like(f3)
            elif r < 3 * self.modal_drop_p:
                gen_emb = torch.zeros_like(gen_emb)

        fused = torch.cat([e1, e2, e3, f1, f2, f3, gen_emb], dim=1)
        fused = self.fuse_norm(fused)

        if self.ablation == 'mri_only':
            fused = fused.clone()
            fused[:, self.mri_dim:] = 0.0
        elif self.ablation == 'genomic_only':
            fused = fused.clone()
            fused[:, :self.mri_dim] = 0.0

        logits = self.head(fused)

        return logits, fused
