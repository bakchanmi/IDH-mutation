
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc as sklearn_auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from fusion.fusion_model       import LateFusionClassifier
from fusion.multimodal_dataset import MultimodalDataset, K_MAJORITY

FINAL_CSV = os.path.join(BASE_DIR, 'TCGA_final_csv', 'final_dataset.csv')
GDC_DIR   = os.path.join(BASE_DIR, 'GDCdata')
UUID_MAP  = os.path.join(GDC_DIR, 'uuid_to_patient_59.csv')

EPOCHS          = 80
WARMUP_EPOCHS   = 10
LR              = 3e-4
WEIGHT_DECAY    = 1e-3
BATCH_SIZE      = 8
PATIENCE        = 30
TOP_K           = 10
SNV_K_BEST      = 20
MODAL_DROP_P    = 0.15
MIXUP_ALPHA     = 0.4
LABEL_SMOOTHING = 0.05
GENOMIC_NOISE   = 0.10
BASE_CH         = 4
EMBED_DIM       = 16
IMG_SIZE    = 128
ROI_SIZE    = 96
N_SPLITS    = 5
SEED        = 456
DEVICE      = 'mps' if torch.backends.mps.is_available() else 'cpu'

SAVE_DIR    = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(SAVE_DIR, exist_ok=True)

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pt = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        return ((1 - pt) ** self.gamma * ce).mean()

def train_one_epoch(
    model, loader, optimizer, criterion, device,
    mixup_alpha: float = 0.0,
    genomic_noise_sigma: float = 0.0,
):
    model.train()
    total_loss = 0.0
    for full, roi, snv, cnv, labels in loader:
        full, roi = full.to(device), roi.to(device)
        snv, cnv  = snv.to(device),  cnv.to(device)
        labels    = labels.to(device)

        if genomic_noise_sigma > 0:
            snv = snv + torch.randn_like(snv) * genomic_noise_sigma
            cnv = cnv + torch.randn_like(cnv) * genomic_noise_sigma

        optimizer.zero_grad()

        if mixup_alpha > 0 and np.random.rand() < 0.5:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(len(labels), device=device)
            snv_mix = lam * snv + (1 - lam) * snv[idx]
            cnv_mix = lam * cnv + (1 - lam) * cnv[idx]
            logits, _ = model(full, roi, snv_mix, cnv_mix)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[idx])
        else:
            logits, _ = model(full, roi, snv, cnv)
            loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    from collections import defaultdict
    model.eval()
    total_loss = 0.0
    raw_probs, raw_scores, raw_labels, raw_logits = [], [], [], []

    for full, roi, snv, cnv, labels in loader:
        full, roi = full.to(device), roi.to(device)
        snv, cnv  = snv.to(device),  cnv.to(device)
        labels    = labels.to(device)
        logits, _ = model(full, roi, snv, cnv)
        total_loss += criterion(logits, labels).item() * len(labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        scores = logits[:, 1] - logits[:, 0]
        raw_probs.extend(probs.cpu().tolist())
        raw_scores.extend(scores.cpu().tolist())
        raw_labels.extend(labels.cpu().tolist())
        raw_logits.extend(logits.cpu().tolist())

    val_loss = total_loss / len(loader.dataset)

    slice_scores = np.array(raw_scores, dtype=np.float32)
    slice_probs  = np.array(raw_probs,  dtype=np.float32)
    slice_labels = np.array(raw_labels, dtype=np.int64)

    sample_to_patient = loader.dataset._samples
    pt_probs_d:  dict = defaultdict(list)
    pt_label_d:  dict = {}
    pt_logits_d: dict = defaultdict(list)

    for i, (prob, label, lg_pair) in enumerate(zip(raw_probs, raw_labels, raw_logits)):
        pid = sample_to_patient[i]
        pt_probs_d[pid].append(prob)
        pt_label_d[pid] = label
        pt_logits_d[pid].append(lg_pair)

    pids           = sorted(pt_probs_d.keys())
    patient_probs  = np.array([np.mean(pt_probs_d[pid])            for pid in pids])
    patient_labels = [pt_label_d[pid]                               for pid in pids]
    patient_logits = np.array([np.mean(pt_logits_d[pid], axis=0)   for pid in pids])
    all_preds      = (patient_probs >= 0.5).astype(int).tolist()

    acc = sum(p == l for p, l in zip(all_preds, patient_labels)) / len(patient_labels)
    tp = sum(p==1 and l==1 for p,l in zip(all_preds, patient_labels))
    tn = sum(p==0 and l==0 for p,l in zip(all_preds, patient_labels))
    fp = sum(p==1 and l==0 for p,l in zip(all_preds, patient_labels))
    fn = sum(p==0 and l==1 for p,l in zip(all_preds, patient_labels))
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    f1   = 2 * tp / (2 * tp + fp + fn + 1e-9)

    if len(set(slice_labels.tolist())) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(slice_labels, slice_scores)

    return val_loss, acc, sens, spec, auc, f1, slice_scores, slice_labels, patient_logits, patient_probs

def verify_fold_splits(skf, labels_all, all_ids, n_splits):
    N = len(all_ids)
    fold_assignment: dict[int, int] = {}

    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(N), labels_all)):
        for idx in val_idx:
            if idx in fold_assignment:
                raise RuntimeError(
                    f"환자 {all_ids[idx]} 이미 fold {fold_assignment[idx]}에 배정됨 — 중복!"
                )
            fold_assignment[idx] = fold + 1

    missing = [i for i in range(N) if i not in fold_assignment]
    if missing:
        raise RuntimeError(f"폴드 미배정 환자 {len(missing)}명: {[all_ids[i] for i in missing]}")

    print(f"[폴드 구성 검증 (StratifiedKFold, n={n_splits}, shuffle=True, seed=42)]")
    print(f"  중복 없음 ✓  |  전원 배정 ✓  ({N}명, 각 환자 정확히 1회 검증)")
    for fn in range(1, n_splits + 1):
        fn_labels = [labels_all[i] for i, f in fold_assignment.items() if f == fn]
        n_pos = sum(fn_labels)
        n_neg = len(fn_labels) - n_pos
        print(f"  Fold {fn}: {len(fn_labels):2d}명  (IDH+ {n_pos}, IDH- {n_neg})")
    print()

def log_fold_diagnostics(fold, val_ds, pt_probs, pt_logits, slice_scores, slice_labels, auc_f):
    n_val = len(val_ds.patient_ids)
    val_labels_arr = np.array([val_ds.labels[k].item() for k in range(n_val)])

    print(f"\n  ┌─ Fold {fold+1} 검증 샘플 로짓·스코어 진단 {'─'*36}┐")
    hdr = f"  {'환자 ID':<32} {'레이블':>5} {'Logit[IDH-]':>12} {'Logit[IDH+]':>12} {'Prob(IDH+)':>11}"
    print(hdr)
    print(f"  {'─'*75}")
    for k, pid in enumerate(val_ds.patient_ids):
        lbl_str  = "IDH+" if val_labels_arr[k] == 1 else "IDH-"
        lg0, lg1 = float(pt_logits[k, 0]), float(pt_logits[k, 1])
        pr       = float(pt_probs[k])
        print(f"  {pid:<32} {lbl_str:>5} {lg0:>12.4f} {lg1:>12.4f} {pr:>11.4f}")

    n_slices = len(slice_scores)
    n_unique = len({round(float(s), 8) for s in slice_scores})
    print(f"\n  슬라이스 스코어: {n_slices}개 ({n_val}명 × top_k 슬라이스)")
    print(f"  고유 스코어:     {n_unique}/{n_slices}  "
          f"({'연속 스코어 ✓' if n_unique == n_slices else '중복 존재 ⚠'})")

    neg_probs = pt_probs[val_labels_arr == 0]
    pos_probs = pt_probs[val_labels_arr == 1]
    if len(neg_probs):
        print(f"  IDH- ({len(neg_probs)}명) 환자평균 Prob: {[f'{p:.4f}' for p in sorted(neg_probs)]}")
    if len(pos_probs):
        print(f"  IDH+ ({len(pos_probs)}명) 환자평균 Prob: [{pos_probs.min():.4f} ~ {pos_probs.max():.4f}]")

    if len(set(slice_labels.tolist())) > 1:
        fpr, tpr, _ = roc_curve(slice_labels, slice_scores)
        auc_trapz   = float(sklearn_auc(fpr, tpr))
        auc_direct  = float(auc_f)
        match       = abs(auc_direct - auc_trapz) < 1e-6
        print(f"\n  AUC 계산 검증 (로짓 차이 기반, {n_slices}포인트):")
        print(f"    roc_auc_score(slice_labels, slice_scores) = {auc_direct:.6f}")
        print(f"    roc_curve() → auc(fpr,tpr) [trapz]        = {auc_trapz:.6f}  "
              f"({'일치 ✓' if match else '불일치 ⚠'})")
        print(f"    ROC 포인트 수: {len(fpr)}개  (슬라이스 {n_slices}개 → 세밀한 곡선)")

    print(f"  └{'─'*77}┘")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Late Fusion IDH Classification')
    parser.add_argument(
        '--ablation', default='full',
        choices=['full', 'mri_only', 'genomic_only', 'mri_snv', 'mri_cnv'],
        help='모달리티 절제 모드 (default: full = 전체 모달리티 사용)',
    )
    args = parser.parse_args()
    ABLATION = args.ablation

    print(f"Device: {DEVICE}")
    print(f"Ablation: {ABLATION}")

    df = pd.read_csv(FINAL_CSV)
    all_ids    = df['submitter_id'].tolist()
    labels_all = df['IDH_label'].values
    N = len(all_ids)
    n_pos_all = int((labels_all==1).sum())
    n_neg_all = int((labels_all==0).sum())
    print(f"총 환자: {N}명  |  IDH+ {n_pos_all}, IDH- {n_neg_all}")

    print(f"\n[⚠ 방법론 주의사항]")
    print(f"  1. 소수 클래스 과소표현: IDH- {n_neg_all}명 → {N_SPLITS}-fold 폴드당 검증 ~{n_neg_all//N_SPLITS}명")
    print(f"     → 폴드별 AUC는 IDH- ~{n_neg_all//N_SPLITS}명으로 완전 분리 가능 → 1.0 수렴 주의.")
    print(f"     → Best 모델 선택/조기종료 기준: val_loss (낮을수록 good).")
    print(f"     → 폴드별 AUC는 참고용, Pooled Overall AUC를 주 지표로 사용.")
    print(f"  2. 유전체 피처는 IDH 상태와 강한 생물학적 상관성 가짐 (1p/19q co-deletion 등).")
    print(f"     완벽한 분리가 생물학적으로 가능함 → 1.0 지표가 반드시 leakage/overfitting")
    print(f"     은 아니나, 소규모 cohort에서는 과적합과 구별하기 어려움.")
    print(f"  3. SelectKBest(k={SNV_K_BEST}): fold 내부에서만 fit ✓. n_train~47 대비 적정 비율.")
    print(f"  4. 외부 테스트셋 없음: 진정한 일반화 성능은 독립 코호트(CGGA 등)로 검증 필요.")
    print(f"  5. StratifiedKFold(n={N_SPLITS}, stratify=IDH): 사용 중 ✓")
    print(f"  6. 클래스 가중치: FocalLoss(γ=2.5, weight=[w_neg, w_pos×1.5]) ✓")
    print(f"     슬라이스 수 기반 균형 가중치 + IDH+ 50% 보너스로 FN 감소 유도.")
    print(f"  7. 모달리티 드롭아웃(p={MODAL_DROP_P}): CNN/RCL/Genomic 브랜치 중 하나를 무작위 제거 ✓")
    print(f"  8. Genomic Mixup(α={MIXUP_ALPHA}): SNV/CNV 선형 보간, MRI 원본 유지 ✓")
    print(f"  9. Label Smoothing(ε={LABEL_SMOOTHING}): FocalLoss 내 과신 예측 억제 ✓")
    print(f" 10. Genomic Noise(σ={GENOMIC_NOISE}): 훈련 시 SNV/CNV에 Gaussian noise 추가.")
    print(f"     → genomic branch의 지나친 지배(memorization) 억제, MRI 의존도 균형화.")
    print(f"     → SNV_K_BEST={SNV_K_BEST}: mutual_info 상위 피처만 남겨 지나친 IDH 완전분리 억제.")
    print(f"\n  [Leakage 설계 원칙]")
    print(f"  SelectKBest / StandardScaler: 각 fold의 train 데이터로만 fit,")
    print(f"  val 데이터에는 transform만 적용 (multimodal_dataset.py:83~102 참조).")
    print(f"  CNV 피처: 환자별 독립 seg 파일 → 교차 통계 없음. MRI: 환자별 z-score.\n")

    id_to_idx = {pid: i for i, pid in enumerate(all_ids)}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    verify_fold_splits(skf, labels_all, all_ids, N_SPLITS)

    fold_results       = []
    all_slice_scores: list[float] = []
    all_slice_labels: list[int]   = []
    pt_probs  = np.zeros(N, dtype=np.float32)
    pt_preds  = np.zeros(N, dtype=np.int64)
    pt_labels = np.zeros(N, dtype=np.int64)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), labels_all)):
        fold_seed = SEED * 100 + fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)

        print(f"\n{'─'*45}")
        print(f"Fold {fold+1}/{N_SPLITS}  (train {len(train_idx)}, val {len(val_idx)})")
        print(f"{'─'*45}")

        train_ids = [all_ids[i] for i in train_idx]
        val_ids   = [all_ids[i] for i in val_idx]

        train_ds = MultimodalDataset(
            FINAL_CSV, GDC_DIR, UUID_MAP,
            top_k=TOP_K, img_size=IMG_SIZE, roi_size=ROI_SIZE,
            patient_ids=train_ids, train=True, augment=True,
            snv_k_best=SNV_K_BEST,
        )
        val_ds = MultimodalDataset(
            FINAL_CSV, GDC_DIR, UUID_MAP,
            top_k=TOP_K, img_size=IMG_SIZE, roi_size=ROI_SIZE,
            patient_ids=val_ids, train=False, augment=False,
            val_all_slices=True,
            snv_scaler=train_ds.snv_scaler,
            cnv_scaler=train_ds.cnv_scaler,
            snv_selector=train_ds.snv_selector,
        )

        n_train_samples = len(train_ds)
        n_val_patients  = len(val_ids)
        print(f"  Train samples: {n_train_samples} (슬라이스 확장 후) | "
              f"Val: {n_val_patients}명 × {TOP_K}슬라이스 집계")

        n_snv_total = train_ds.snv_selector.n_features_in_
        n_tr_pos = int((labels_all[train_idx] == 1).sum())
        n_tr_neg = len(train_idx) - n_tr_pos
        n_va_pos = int((labels_all[val_idx] == 1).sum())
        n_va_neg = len(val_idx) - n_va_pos
        print(f"  ── Leakage 감사 ──────────────────────────────────")
        print(f"  Train {len(train_idx)}명 (IDH+ {n_tr_pos}, IDH- {n_tr_neg})")
        print(f"    SelectKBest(k={train_ds.snv_dim}/{n_snv_total}): train으로만 fit ✓")
        print(f"    SNV StandardScaler: train으로만 fit ✓")
        print(f"    CNV StandardScaler: train으로만 fit ✓")
        print(f"  Val  {len(val_idx)}명  (IDH+ {n_va_pos}, IDH- {n_va_neg})")
        print(f"    SelectKBest / StandardScaler: transform only ✓")
        IDH_LEAK_COLS = {'IDH1_x', 'IDH2_x', 'IDH1_y', 'IDH2_y', 'IDH_label'}
        selected_cols = train_ds.selected_snv_cols
        leaked = [c for c in selected_cols if c in IDH_LEAK_COLS]
        print(f"  SelectKBest 선택 피처 ({len(selected_cols)}개):")
        for i, c in enumerate(selected_cols):
            tag = " ★IDH_LEAK★" if c in IDH_LEAK_COLS else ""
            print(f"    [{i+1:2d}] {c}{tag}")
        if leaked:
            print(f"  ⚠️  IDH leakage 피처 발견: {leaked}  → META_COLS 확인 필요!")
        else:
            print(f"  IDH 피처 미포함 확인 ✓")
        print(f"  ──────────────────────────────────────────────────")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        n_neg = int((labels_all[train_idx] == 0).sum())
        n_pos = int((labels_all[train_idx] == 1).sum())
        n_neg_sl = n_neg * TOP_K
        n_pos_sl = n_pos * K_MAJORITY
        n_sl     = n_neg_sl + n_pos_sl
        w_neg    = n_sl / (2 * n_neg_sl)
        w_pos    = n_sl / (2 * n_pos_sl) * 1.5
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(DEVICE)
        print(f"  클래스 가중치: IDH- {w_neg:.2f} (슬라이스 {n_neg_sl}개), "
              f"IDH+ {w_pos:.2f} (슬라이스 {n_pos_sl}개)")

        model = LateFusionClassifier(
            num_classes=2, snv_dim=train_ds.snv_dim, cnv_dim=train_ds.cnv_dim,
            modal_drop_p=MODAL_DROP_P, base_ch=BASE_CH, embed_dim=EMBED_DIM,
            ablation=ABLATION,
        ).to(DEVICE)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  SNV 피처 수: {train_ds.snv_dim}  |  파라미터 수: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
        )
        criterion = FocalLoss(
            gamma=2.5, weight=class_weights, label_smoothing=LABEL_SMOOTHING
        )

        best_loss  = float('inf')
        best_state = None
        no_improve = 0

        for epoch in range(EPOCHS):
            loss = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE,
                mixup_alpha=MIXUP_ALPHA, genomic_noise_sigma=GENOMIC_NOISE,
            )
            scheduler.step()

            val_loss, acc, sens, spec, auc, f1, *_ = evaluate(
                model, val_loader, criterion, DEVICE
            )
            improved = val_loss < best_loss
            flag = " ◀ best" if improved else ""
            if (epoch + 1) % 10 == 0 or improved:
                print(f"  Epoch {epoch+1:3d} | train {loss:.4f} | val {val_loss:.4f} | "
                      f"acc {acc:.4f} | sens {sens:.4f} | spec {spec:.4f} | "
                      f"auc {auc:.4f} | f1 {f1:.4f}{flag}")
            if improved:
                best_loss  = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            ckpt_path = os.path.join(SAVE_DIR, f'fusion_{ABLATION}_fold{fold+1}.pt')
            torch.save({'state_dict': best_state, 'val_loss': best_loss}, ckpt_path)

        if best_state is not None:
            model.load_state_dict(best_state)
        _, acc_f, sens_f, spec_f, auc_f, f1_f, slice_scores_f, slice_labels_f, logits_f, pt_probs_f = evaluate(
            model, val_loader, criterion, DEVICE
        )
        fold_results.append({'acc': acc_f, 'sens': sens_f, 'spec': spec_f, 'auc': auc_f, 'f1': f1_f})

        log_fold_diagnostics(fold, val_ds, pt_probs_f, logits_f, slice_scores_f, slice_labels_f, auc_f)

        all_slice_scores.extend(slice_scores_f.tolist())
        all_slice_labels.extend(slice_labels_f.tolist())

        for k, pid in enumerate(val_ds.patient_ids):
            orig_idx = id_to_idx[pid]
            pt_probs[orig_idx]  = pt_probs_f[k]
            pt_preds[orig_idx]  = int(pt_probs_f[k] >= 0.5)
            pt_labels[orig_idx] = labels_all[orig_idx]

        print(f"  ▶ Fold {fold+1} | best_val_loss {best_loss:.4f} | "
              f"acc {acc_f:.4f} | sens {sens_f:.4f} | spec {spec_f:.4f} | "
              f"auc {auc_f:.4f} | f1 {f1_f:.4f}")

    fold_aucs  = [r['auc']  for r in fold_results]
    fold_accs  = [r['acc']  for r in fold_results]
    fold_senss = [r['sens'] for r in fold_results]
    fold_specs = [r['spec'] for r in fold_results]
    fold_f1s   = [r['f1']   for r in fold_results]

    print(f"\n{'='*54}")
    print(f"  [{N_SPLITS}-Fold 평균 | Ablation: {ABLATION}]")
    print(f"  Fold AUCs    : {' / '.join(f'{a:.4f}' for a in fold_aucs)}")
    print(f"  AUC          : {np.nanmean(fold_aucs):.4f} ± {np.nanstd(fold_aucs):.4f}")
    print(f"  Accuracy     : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"  Sensitivity  : {np.mean(fold_senss):.4f} ± {np.std(fold_senss):.4f}")
    print(f"  Specificity  : {np.mean(fold_specs):.4f} ± {np.std(fold_specs):.4f}")
    print(f"  F1           : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    pooled_auc = roc_auc_score(all_slice_labels, all_slice_scores)

    p_tp = int(((pt_preds==1) & (pt_labels==1)).sum())
    p_tn = int(((pt_preds==0) & (pt_labels==0)).sum())
    p_fp = int(((pt_preds==1) & (pt_labels==0)).sum())
    p_fn = int(((pt_preds==0) & (pt_labels==1)).sum())
    p_acc  = (p_tp + p_tn) / N
    p_sens = p_tp / (p_tp + p_fn + 1e-9)
    p_spec = p_tn / (p_tn + p_fp + 1e-9)
    p_f1   = 2 * p_tp / (2 * p_tp + p_fp + p_fn + 1e-9)

    print(f"\n  [Pooled Overall ★ 주 평가지표 (threshold=0.5) | Ablation: {ABLATION}]")
    print(f"  Accuracy   : {p_acc:.4f}   (TP={p_tp} TN={p_tn} FP={p_fp} FN={p_fn})")
    print(f"  Sensitivity: {p_sens:.4f}")
    print(f"  Specificity: {p_spec:.4f}")
    print(f"  F1 Score   : {p_f1:.4f}")
    print(f"  AUC ★      : {pooled_auc:.4f}")
    print(f"{'='*54}")

    thresholds = np.linspace(0.05, 0.95, 91)
    best_j, best_thr = -1.0, 0.5
    for thr in thresholds:
        preds_t = (pt_probs >= thr).astype(int)
        tp_t = int(((preds_t == 1) & (pt_labels == 1)).sum())
        tn_t = int(((preds_t == 0) & (pt_labels == 0)).sum())
        fp_t = int(((preds_t == 1) & (pt_labels == 0)).sum())
        fn_t = int(((preds_t == 0) & (pt_labels == 1)).sum())
        s_t  = tp_t / (tp_t + fn_t + 1e-9)
        sp_t = tn_t / (tn_t + fp_t + 1e-9)
        j    = s_t + sp_t - 1
        if j > best_j:
            best_j, best_thr = j, thr

    opt  = (pt_probs >= best_thr).astype(int)
    o_tp = int(((opt == 1) & (pt_labels == 1)).sum())
    o_tn = int(((opt == 0) & (pt_labels == 0)).sum())
    o_fp = int(((opt == 1) & (pt_labels == 0)).sum())
    o_fn = int(((opt == 0) & (pt_labels == 1)).sum())
    o_acc  = (o_tp + o_tn) / N
    o_sens = o_tp / (o_tp + o_fn + 1e-9)
    o_spec = o_tn / (o_tn + o_fp + 1e-9)
    o_f1   = 2 * o_tp / (2 * o_tp + o_fp + o_fn + 1e-9)

    print(f"\n  [Optimal Threshold = {best_thr:.2f}  (Youden's J, in-sample 참고)]")
    print(f"  Accuracy   : {o_acc:.4f}   (TP={o_tp} TN={o_tn} FP={o_fp} FN={o_fn})")
    print(f"  Sensitivity: {o_sens:.4f}")
    print(f"  Specificity: {o_spec:.4f}")
    print(f"  F1 Score   : {o_f1:.4f}")
    print(f"  AUC ★      : {pooled_auc:.4f}  (임계값과 무관)")
    print(f"{'='*54}")

if __name__ == '__main__':
    main()
