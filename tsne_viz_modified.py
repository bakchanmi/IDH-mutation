
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fusion.fusion_model       import LateFusionClassifier
from fusion.multimodal_dataset import MultimodalDataset

FINAL_CSV  = os.path.join(BASE_DIR, 'TCGA_final_csv', 'final_dataset.csv')
GDC_DIR    = os.path.join(BASE_DIR, 'GDCdata')
UUID_MAP   = os.path.join(GDC_DIR, 'uuid_to_patient_59.csv')
SAVE_DIR   = os.path.join(BASE_DIR, 'checkpoints')

TOP_K      = 10
SNV_K_BEST = 20
IMG_SIZE   = 128
ROI_SIZE   = 96
BASE_CH    = 4
EMBED_DIM  = 16
BATCH_SIZE = 8
SEED       = 4
N_SPLITS   = 5
DEVICE     = 'mps' if torch.backends.mps.is_available() else 'cpu'

ABLATIONS = ['full', 'mri_only', 'genomic_only']
TITLES    = {
    'full':         'Ours',
    'mri_only':     'MRI Only',
    'genomic_only': 'Genomic Only',
}
COL_NEG = '#E63946'
COL_POS = '#1D6FA4'

@torch.no_grad()
def _run_one_fold(
    ablation: str,
    fold_idx: int,
    all_ids: list,
    labels_all: np.ndarray,
) -> tuple:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits    = list(skf.split(np.zeros(len(all_ids)), labels_all))
    train_idx, _ = splits[fold_idx]
    train_ids = [all_ids[i] for i in train_idx]

    train_ds = MultimodalDataset(
        FINAL_CSV, GDC_DIR, UUID_MAP,
        top_k=TOP_K, img_size=IMG_SIZE, roi_size=ROI_SIZE,
        patient_ids=train_ids, train=True, augment=False,
        snv_k_best=SNV_K_BEST,
    )
    all_ds = MultimodalDataset(
        FINAL_CSV, GDC_DIR, UUID_MAP,
        top_k=TOP_K, img_size=IMG_SIZE, roi_size=ROI_SIZE,
        patient_ids=all_ids, train=False, augment=False,
        val_all_slices=True,
        snv_scaler=train_ds.snv_scaler,
        cnv_scaler=train_ds.cnv_scaler,
        snv_selector=train_ds.snv_selector,
    )
    loader = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LateFusionClassifier(
        num_classes=2,
        snv_dim=train_ds.snv_dim,
        cnv_dim=train_ds.cnv_dim,
        modal_drop_p=0.0,
        base_ch=BASE_CH,
        embed_dim=EMBED_DIM,
        ablation=ablation,
    ).to(DEVICE)

    ckpt_path = os.path.join(SAVE_DIR, f'fusion_{ablation}_fold{fold_idx + 1}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    raw_fused, raw_labels = [], []
    for full_img, roi, snv, cnv, labels in loader:
        full_img, roi = full_img.to(DEVICE), roi.to(DEVICE)
        snv, cnv      = snv.to(DEVICE),  cnv.to(DEVICE)
        _, fused      = model(full_img, roi, snv, cnv)
        raw_fused.append(fused.cpu().numpy())
        raw_labels.extend(labels.tolist())

    all_fused  = np.concatenate(raw_fused)
    all_labels = np.array(raw_labels, dtype=np.int64)

    pt_feats: dict = defaultdict(list)
    pt_label: dict = {}
    for i, (fv, lbl) in enumerate(zip(all_fused, all_labels)):
        pid_idx = all_ds._samples[i]
        pt_feats[pid_idx].append(fv)
        pt_label[pid_idx] = int(lbl)

    idxs = sorted(pt_feats.keys())
    feat  = np.array([np.mean(pt_feats[k], axis=0) for k in idxs])
    label = np.array([pt_label[k]                  for k in idxs])
    return feat, label

def extract_patient_features(
    ablation: str,
    all_ids: list,
    labels_all: np.ndarray,
    folds: list[int],
) -> tuple:
    all_fold_feats = []
    label_arr = None
    for fi in folds:
        feat, lbl = _run_one_fold(ablation, fi, all_ids, labels_all)
        all_fold_feats.append(feat)
        label_arr = lbl
    feat_arr = np.mean(all_fold_feats, axis=0)
    return feat_arr, label_arr

_MRI_DIM     = BASE_CH * 7 * 2
_GENOMIC_DIM = EMBED_DIM
_ACTIVE_DIMS = {
    'full':         slice(None),
    'mri_only':     slice(0, _MRI_DIM),
    'genomic_only': slice(_MRI_DIM, None),
}

def _preprocess(X: np.ndarray, ablation: str) -> np.ndarray:
    X = X[:, _ACTIVE_DIMS[ablation]]
    std = X.std(axis=0)
    X = X[:, std > 1e-8]
    if X.shape[1] == 0:
        return np.zeros((len(X), 2))
    return StandardScaler().fit_transform(X)

def run_tsne(X: np.ndarray, perplexity: int = 15, max_iter: int = 3000,
             tsne_seed: int = 0) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=tsne_seed,
        learning_rate='auto',
        init='pca',
    ).fit_transform(X)

def _add_ellipse(ax, X: np.ndarray, color: str, n_std: float = 1.5):
    if len(X) < 3:
        return
    import numpy.linalg as la
    cov  = np.cov(X.T)
    vals, vecs = la.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(
        xy=X.mean(axis=0), width=w, height=h, angle=angle,
        edgecolor=color, facecolor=color,
        alpha=0.12, lw=2.0, linestyle='--', zorder=2,
    )
    ax.add_patch(ell)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', default='1,2,3,4,5',
                        help='앙상블할 fold 번호 목록, 쉼표 구분 (default: 1,2,3,4,5)')
    parser.add_argument('--out',   default='tsne_comparison.png',
                        help='출력 PNG 경로 (default: tsne_comparison.png)')
    args    = parser.parse_args()
    folds   = [int(f) - 1 for f in args.folds.split(',')]

    df = pd.read_csv(FINAL_CSV)
    all_ids    = df['submitter_id'].tolist()
    labels_all = df['IDH_label'].values
    n_pos = int((labels_all == 1).sum())
    n_neg = int((labels_all == 0).sum())

    print(f"Device : {DEVICE}")
    print(f"Folds  : {[f+1 for f in folds]}  (앙상블, SEED={SEED})")
    print(f"환자   : {len(all_ids)}명  (IDH+ {n_pos}, IDH- {n_neg})\n")

    feats, lbls = {}, {}
    for ab in ABLATIONS:
        print(f"  [{ab}] 피처 추출 중 ...", end=' ', flush=True)
        feats[ab], lbls[ab] = extract_patient_features(
            ab, all_ids, labels_all, folds
        )
        sil = silhouette_score(_preprocess(feats[ab], ab), lbls[ab])
        print(f"shape={feats[ab].shape}  Silhouette={sil:.4f}")

    print("\nt-SNE 투영 중 (perplexity=15, max_iter=3000) ...")
    coords = {}
    for ab in ABLATIONS:
        X_pre = _preprocess(feats[ab], ab)
        coords[ab] = run_tsne(X_pre, perplexity=15, max_iter=3000, tsne_seed=0)
        sil = silhouette_score(X_pre, lbls[ab])
        print(f"  [{ab}]  Silhouette = {sil:.4f}  (활성차원: {X_pre.shape[1]}D)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.patch.set_facecolor('white')

    for ax, ab in zip(axes, ABLATIONS):
        X2  = coords[ab]
        lbl = lbls[ab]
        sil = silhouette_score(_preprocess(feats[ab], ab), lbl)

        for cls, col in [(0, COL_NEG), (1, COL_POS)]:
            _add_ellipse(ax, X2[lbl == cls], col, n_std=1.5)

        scatter_cfg = [
            (0, 'X',  130, 0.95),
            (1, 'o',  110, 0.88),
        ]
        for cls, marker, ms, alpha in scatter_cfg:
            col  = COL_NEG if cls == 0 else COL_POS
            mask = lbl == cls
            ax.scatter(
                X2[mask, 0], X2[mask, 1],
                c=col, marker=marker, s=ms,
                edgecolors='white', linewidths=0.8,
                alpha=alpha, zorder=4,
            )

        for cls, col in [(0, COL_NEG), (1, COL_POS)]:
            mask = lbl == cls
            cx, cy = X2[mask, 0].mean(), X2[mask, 1].mean()
            name = 'IDH-' if cls == 0 else 'IDH+'
            ax.text(
                cx, cy, name, fontsize=10, fontweight='bold', color='white',
                ha='center', va='center', zorder=5,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=col)],
            )

        ax.set_title(TITLES[ab], fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(
            f'Silhouette Score = {sil:.4f}',
            fontsize=11, labelpad=8,
            color='#333333',
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#F5F5F5')

    legend_handles = [
        mpatches.Patch(color=COL_NEG, label='IDH Wild-Type  (IDH−)'),
        mpatches.Patch(color=COL_POS, label='IDH Mutant  (IDH+)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center', ncol=2,
        fontsize=12, frameon=True,
        bbox_to_anchor=(0.5, -0.06),
        edgecolor='#AAAAAA',
    )

    fig.suptitle(
        't-SNE Visualization of Pre-Fusion Feature Vectors  (Patient-Level, n=59)',
        fontsize=13, fontweight='bold', y=1.03,
    )

    plt.tight_layout(pad=1.8)
    out_path = os.path.join(BASE_DIR, args.out)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n저장 완료: {out_path}")

if __name__ == '__main__':
    main()
