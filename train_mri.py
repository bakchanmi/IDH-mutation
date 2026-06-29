
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from mri.mri_dataset  import MRIDataset
from mri.cnn_encoder  import CNNEncoder
from mri.rcl_encoder  import RCLEncoder

FINAL_CSV = os.path.join(BASE_DIR, 'TCGA_final_csv', 'final_dataset.csv')

EPOCHS      = 30
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 8
TOP_K       = 10
IMG_SIZE    = 128
ROI_SIZE    = 96
N_SPLITS    = 5
EMBED_DIM   = 128
DEVICE      = 'mps' if torch.backends.mps.is_available() else 'cpu'

class DualMRIModel(nn.Module):

    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()
        self.cnn = CNNEncoder(in_channels=4, embed_dim=embed_dim)
        self.rcl = RCLEncoder(in_channels=7, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, full, roi):
        e_cnn, _, _ = self.cnn(full)
        e_rcl, _, _ = self.rcl(roi)
        return self.head(torch.cat([e_cnn, e_rcl], dim=1))

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for full, roi, labels in loader:
        full, roi, labels = full.to(device), roi.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(full, roi)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for full, roi, labels in loader:
        full, roi, labels = full.to(device), roi.to(device), labels.to(device)
        preds = model(full, roi).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
    return correct / total

def main():
    print(f"Device: {DEVICE}")

    import pandas as pd
    df = pd.read_csv(FINAL_CSV)
    all_ids = df['submitter_id'].tolist()
    labels_all = df['IDH_label'].values
    N = len(all_ids)
    print(f"총 환자: {N}명  |  IDH+ {(labels_all==1).sum()}, IDH- {(labels_all==0).sum()}")

    n_pos = (labels_all == 1).sum()
    n_neg = (labels_all == 0).sum()
    class_weights = torch.tensor(
        [N / (2 * n_neg), N / (2 * n_pos)], dtype=torch.float32
    ).to(DEVICE)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), labels_all)):
        print(f"\n── Fold {fold+1}/{N_SPLITS} ──")
        train_ids = [all_ids[i] for i in train_idx]
        val_ids   = [all_ids[i] for i in val_idx]

        train_ds = MRIDataset(FINAL_CSV, top_k=TOP_K, img_size=IMG_SIZE, roi_size=ROI_SIZE,
                               patient_ids=train_ids, train=True, augment=True)
        val_ds   = MRIDataset(FINAL_CSV, top_k=TOP_K, img_size=IMG_SIZE, roi_size=ROI_SIZE,
                               patient_ids=val_ids,   train=False, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model     = DualMRIModel(embed_dim=EMBED_DIM, num_classes=2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_acc = 0.0
        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                acc = evaluate(model, val_loader, DEVICE)
                print(f"  Epoch {epoch+1:3d} | loss {loss:.4f} | val acc {acc:.3f}")
                best_acc = max(best_acc, acc)

        fold_accs.append(best_acc)
        print(f"  Best val acc: {best_acc:.3f}")

    print(f"\n{'='*40}")
    print(f"5-Fold 평균 val accuracy: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")

if __name__ == '__main__':
    main()
