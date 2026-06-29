
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import pandas as pd
from skimage.transform import resize as sk_resize
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fusion.fusion_model       import LateFusionClassifier
from fusion.multimodal_dataset import MultimodalDataset
from mri.mri_utils import load_volume, select_top_slices

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
SEED       = 62
N_SPLITS   = 5
DEVICE     = 'mps' if torch.backends.mps.is_available() else 'cpu'

_WT_COLOR = '#FFD700'
_TC_COLOR = '#FF4500'
_ET_COLOR = '#00BFFF'

COL_TITLES = ['T1ce (Original)', 'Tumor Masks', 'CNN Grad-CAM\n(Full Slice, E3)', 'RCL Grad-CAM\n(ROI → Full, f3)']

class GradCAM:

    def __init__(self, model: LateFusionClassifier):
        self.model = model
        self._acts: dict[str, torch.Tensor] = {}
        self._handles = [
            model.cnn_enc.block3.register_forward_hook(self._fwd('cnn')),
            model.rcl_enc.rcl3.rcl.register_forward_hook(self._fwd('rcl')),
        ]

    def _fwd(self, name: str):
        def hook(module, inp, out):
            out.retain_grad()
            self._acts[name] = out
        return hook

    def compute(
        self,
        full_img: torch.Tensor,
        roi:      torch.Tensor,
        snv:      torch.Tensor,
        cnv:      torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[dict[str, np.ndarray], int, float]:
        self.model.eval()
        self.model.zero_grad()

        logits, _ = self.model(full_img, roi, snv, cnv)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        logits[0, target_class].backward()

        cams = {}
        for name, act in self._acts.items():
            grad = act.grad
            if grad is None:
                continue
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act.detach()).sum(dim=1).squeeze(0)
            cam = F.relu(cam)
            if cam.max() > 1e-8:
                cam = (cam / cam.max()).cpu().numpy().astype(np.float32)
            else:
                cam = cam.cpu().numpy().astype(np.float32)
            cams[name] = cam

        return cams, target_class, float(probs[0, target_class].item())

    def remove(self):
        for h in self._handles:
            h.remove()

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _get_masks(seg2d: np.ndarray):
    wt = (seg2d > 0).astype(np.uint8)
    tc = ((seg2d == 1) | (seg2d == 4)).astype(np.uint8)
    et = (seg2d == 4).astype(np.uint8)
    return wt, tc, et

def _wt_bbox(seg2d: np.ndarray, pad: int = 10):
    ys, xs = np.where(seg2d > 0)
    if len(ys) == 0:
        h, w = seg2d.shape
        return h // 4, 3 * h // 4, w // 4, 3 * w // 4
    y1 = max(ys.min() - pad, 0)
    y2 = min(ys.max() + pad + 1, seg2d.shape[0])
    x1 = max(xs.min() - pad, 0)
    x2 = min(xs.max() + pad + 1, seg2d.shape[1])
    return y1, y2, x1, x2

def _rcl_cam_to_full(cam_small: np.ndarray, seg2d: np.ndarray,
                     full_h: int = 240, full_w: int = 240) -> np.ndarray:
    y1, y2, x1, x2 = _wt_bbox(seg2d)
    crop_h, crop_w  = y2 - y1, x2 - x1
    cam_crop = sk_resize(
        cam_small, (crop_h, crop_w),
        order=1, anti_aliasing=True, preserve_range=True,
    ).astype(np.float32)
    canvas = np.zeros((full_h, full_w), dtype=np.float32)
    canvas[y1:y2, x1:x2] = cam_crop
    return canvas

def _apply_gradcam_overlay(ax, bg: np.ndarray, cam: np.ndarray,
                           cmap: str = 'jet', thresh: float = 0.15):
    ax.imshow(bg, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    colormap   = plt.colormaps[cmap]
    cam_rgba   = colormap(cam).astype(np.float32)
    cam_rgba[..., 3] = np.where(cam > thresh, cam * 0.80, 0.0)
    ax.imshow(cam_rgba, interpolation='bilinear')

def _build_datasets(fold_idx: int, all_ids: list, labels_all: np.ndarray,
                    target_ids: list):
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
    target_ds = MultimodalDataset(
        FINAL_CSV, GDC_DIR, UUID_MAP,
        top_k=TOP_K, img_size=IMG_SIZE, roi_size=ROI_SIZE,
        patient_ids=target_ids, train=False, augment=False,
        snv_scaler=train_ds.snv_scaler,
        cnv_scaler=train_ds.cnv_scaler,
        snv_selector=train_ds.snv_selector,
    )
    return train_ds, target_ds

def _load_model(fold_idx: int, train_ds: MultimodalDataset) -> LateFusionClassifier:
    ckpt_path = os.path.join(SAVE_DIR, f'fusion_full_fold{fold_idx + 1}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")
    model = LateFusionClassifier(
        num_classes=2,
        snv_dim=train_ds.snv_dim,
        cnv_dim=train_ds.cnv_dim,
        modal_drop_p=0.0,
        base_ch=BASE_CH,
        embed_dim=EMBED_DIM,
        ablation='full',
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patients', default=None,
                        help='쉼표 구분 submitter_id 목록 (default: IDH+ 1명 + IDH- 1명 자동 선택)')
    parser.add_argument('--fold',    default=1, type=int,
                        help='체크포인트 fold 번호 (1~5, default: 1)')
    parser.add_argument('--out',     default='gradcam.png',
                        help='출력 PNG 경로 (default: gradcam.png)')
    args = parser.parse_args()

    df         = pd.read_csv(FINAL_CSV)
    all_ids    = df['submitter_id'].tolist()
    labels_all = df['IDH_label'].values
    fold_idx   = args.fold - 1

    if args.patients:
        target_sids = [s.strip() for s in args.patients.split(',')]
    else:
        idh_pos = df[df['IDH_label'] == 1]['submitter_id'].iloc[0]
        idh_neg = df[df['IDH_label'] == 0]['submitter_id'].iloc[0]
        target_sids = [idh_pos, idh_neg]
        print(f"기본 환자 선택: IDH+ {idh_pos},  IDH- {idh_neg}")

    print(f"Fold     : {args.fold}  |  Device: {DEVICE}")
    print(f"환자     : {target_sids}\n")

    print("스케일러 fit 중 ...")
    train_ds, target_ds = _build_datasets(fold_idx, all_ids, labels_all, target_sids)
    model = _load_model(fold_idx, train_ds)
    grad_cam = GradCAM(model)

    n = len(target_sids)
    fig, axes = plt.subplots(n, 4, figsize=(22, 5.5 * n))
    fig.patch.set_facecolor('white')
    if n == 1:
        axes = axes[np.newaxis, :]

    for row_idx, sid in enumerate(target_sids):
        row_info = df[df['submitter_id'] == sid]
        brats_id  = row_info['BraTS_2019_subject_ID'].values[0]
        idh_label = int(row_info['IDH_label'].values[0])
        idh_str   = 'IDH+ (Mutant)' if idh_label == 1 else 'IDH- (Wild-Type)'

        print(f"  [{sid}]  {brats_id}  |  {idh_str}")

        vol, seg = load_volume(brats_id)
        top_slices = select_top_slices(seg, k=TOP_K)
        z = int(top_slices[len(top_slices) // 2])
        print(f"    z = {z}")

        p_idx = target_ds.patient_ids.index(sid)
        snv   = target_ds.snv[p_idx].unsqueeze(0).to(DEVICE)
        cnv   = target_ds.cnv[p_idx].unsqueeze(0).to(DEVICE)

        from mri.mri_utils import extract_full_slice, extract_roi_slice
        full_arr = extract_full_slice(vol, z, IMG_SIZE)
        roi_arr  = extract_roi_slice(vol, seg, z, ROI_SIZE)
        full_img = torch.from_numpy(full_arr).unsqueeze(0).to(DEVICE)
        roi_img  = torch.from_numpy(roi_arr).unsqueeze(0).to(DEVICE)

        cams, pred_cls, pred_prob = grad_cam.compute(full_img, roi_img, snv, cnv)
        pred_str = 'IDH+' if pred_cls == 1 else 'IDH-'
        print(f"    pred = {pred_str}  (p={pred_prob:.3f})")

        bg   = vol[1, :, :, z]
        seg2d = seg[:, :, z]

        cnn_cam = sk_resize(
            cams['cnn'], (240, 240),
            order=1, anti_aliasing=True, preserve_range=True,
        ).astype(np.float32)
        if cnn_cam.max() > 1e-8:
            cnn_cam = cnn_cam / cnn_cam.max()

        rcl_cam = _rcl_cam_to_full(cams['rcl'], seg2d)
        if rcl_cam.max() > 1e-8:
            rcl_cam = rcl_cam / rcl_cam.max()

        wt, tc, et = _get_masks(seg2d)

        ax_row = axes[row_idx]

        ax_row[0].imshow(bg, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')

        ax_row[1].imshow(bg, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        for mask, color in [(wt, _WT_COLOR), (tc, _TC_COLOR), (et, _ET_COLOR)]:
            rgba = np.zeros((*bg.shape, 4), dtype=np.float32)
            r, g, b = _hex_to_rgb(color)
            rgba[mask > 0] = [r, g, b, 0.55]
            ax_row[1].imshow(rgba, interpolation='none')

        _apply_gradcam_overlay(ax_row[2], bg, cnn_cam)

        _apply_gradcam_overlay(ax_row[3], bg, rcl_cam)

        correct = (pred_cls == idh_label)
        pred_mark = '✓' if correct else '✗'
        ax_row[0].set_ylabel(
            f'{sid}\n{idh_str}\npred: {pred_str}  {pred_mark} ({pred_prob:.2f})',
            fontsize=9.5, rotation=0, ha='right', va='center',
            labelpad=8,
        )

        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_facecolor('black')

    for c, title in enumerate(COL_TITLES):
        axes[0, c].set_title(title, fontsize=12, fontweight='bold', pad=8)

    mask_legend = [
        mpatches.Patch(color=_WT_COLOR, label='WT (NCR+ED+ET)'),
        mpatches.Patch(color=_TC_COLOR, label='TC (NCR+ET)'),
        mpatches.Patch(color=_ET_COLOR, label='ET'),
    ]
    fig.legend(handles=mask_legend, loc='lower center', ncol=3,
               fontsize=10, frameon=True,
               bbox_to_anchor=(0.35, -0.03), edgecolor='#AAAAAA')

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.72, 0.05, 0.22, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Grad-CAM Activation', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    fig.suptitle(
        'Grad-CAM: CNN vs RCL Encoder Attention  |  Full Multimodal (IDH Classification)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    out_path = os.path.join(BASE_DIR, args.out)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    grad_cam.remove()
    print(f"\n저장 완료: {out_path}")

if __name__ == '__main__':
    main()
