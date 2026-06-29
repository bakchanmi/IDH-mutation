
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from genomics.genomic_dataset import GenomicDataset
from genomics.genomic_encoder import GenomicEncoder

FINAL_CSV = os.path.join(BASE_DIR, 'TCGA_final_csv', 'final_dataset.csv')
GDC_DIR   = os.path.join(BASE_DIR, 'GDCdata')
UUID_MAP  = os.path.join(GDC_DIR, 'uuid_to_patient_59.csv')

EPOCHS      = 50
LR          = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 16
N_SPLITS    = 5
DEVICE      = 'mps' if torch.backends.mps.is_available() else 'cpu'

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for snv, cnv, labels in loader:
        snv, cnv, labels = snv.to(device), cnv.to(device), labels.to(device)
        optimizer.zero_grad()
        _, logits = model(snv, cnv)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for snv, cnv, labels in loader:
        snv, cnv, labels = snv.to(device), cnv.to(device), labels.to(device)
        _, logits = model(snv, cnv)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
    return correct / total

def main():
    print(f"Device: {DEVICE}")

    full_ds = GenomicDataset(FINAL_CSV, GDC_DIR, UUID_MAP)
    N = len(full_ds)
    labels_all = full_ds.labels.numpy()
    print(f"총 환자: {N}명  |  IDH+ {(labels_all==1).sum()}명, IDH- {(labels_all==0).sum()}명")

    n_pos = (labels_all == 1).sum()
    n_neg = (labels_all == 0).sum()
    pos_w = N / (2 * n_pos)
    neg_w = N / (2 * n_neg)
    class_weights = torch.tensor([neg_w, pos_w], dtype=torch.float32).to(DEVICE)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), labels_all)):
        print(f"\n── Fold {fold+1}/{N_SPLITS} ──")

        train_patient_ids = [full_ds.patient_ids[i] for i in train_idx]
        val_patient_ids   = [full_ds.patient_ids[i] for i in val_idx]

        train_ds = GenomicDataset(FINAL_CSV, GDC_DIR, UUID_MAP, patient_ids=train_patient_ids)
        val_ds   = GenomicDataset(FINAL_CSV, GDC_DIR, UUID_MAP,
                                  patient_ids=val_patient_ids,
                                  snv_scaler=train_ds.snv_scaler,
                                  cnv_scaler=train_ds.cnv_scaler)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        model = GenomicEncoder(num_classes=2).to(DEVICE)
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
