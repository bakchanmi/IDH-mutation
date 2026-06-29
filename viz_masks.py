
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from mri.mri_utils import load_volume, select_top_slices

FINAL_CSV   = os.path.join(BASE_DIR, 'TCGA_final_csv', 'final_dataset.csv')
MODALITIES  = ['t1', 't1ce', 't2', 'flair']
MOD_LABELS  = ['T1', 'T1ce', 'T2', 'FLAIR']
TOP_K       = 10

_WT_COLOR = '#FFD700'
_TC_COLOR = '#FF4500'
_ET_COLOR = '#00BFFF'

def _get_masks(seg2d: np.ndarray):
    wt = (seg2d > 0).astype(np.uint8)
    tc = ((seg2d == 1) | (seg2d == 4)).astype(np.uint8)
    et = (seg2d == 4).astype(np.uint8)
    return wt, tc, et

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def _overlay(ax, bg: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.50):
    ax.imshow(bg, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    rgba = np.zeros((*bg.shape, 4), dtype=np.float32)
    r, g, b = _hex_to_rgb(color)
    rgba[mask > 0] = [r, g, b, alpha]
    ax.imshow(rgba, interpolation='none')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient',  default=None,
                        help='BraTS_2019_subject_ID (default: 첫 번째 환자)')
    parser.add_argument('--modality', default='t1ce',
                        choices=MODALITIES,
                        help='마스크 배경 모달리티 (default: t1ce)')
    parser.add_argument('--out',      default='mask_viz.png',
                        help='출력 PNG 경로 (default: mask_viz.png)')
    args = parser.parse_args()

    df = pd.read_csv(FINAL_CSV)
    brats_col = 'BraTS_2019_subject_ID'

    if args.patient:
        brats_id = args.patient
    else:
        brats_id = df[brats_col].iloc[0]

    row = df[df[brats_col] == brats_id]
    submitter = row['submitter_id'].values[0] if len(row) else '?'
    idh_label = int(row['IDH_label'].values[0]) if len(row) else -1
    idh_str   = 'IDH Mutant (IDH+)' if idh_label == 1 else 'IDH Wild-Type (IDH-)'

    print(f"Patient  : {brats_id}")
    print(f"ID       : {submitter}  |  {idh_str}")

    vol, seg = load_volume(brats_id)
    top_slices = select_top_slices(seg, k=TOP_K)
    z = int(top_slices[len(top_slices) // 2])

    print(f"Top-{TOP_K} slices : {top_slices.tolist()}")
    print(f"Central z-slice  : {z}\n")

    seg2d = seg[:, :, z]
    wt, tc, et = _get_masks(seg2d)
    bg_idx = MODALITIES.index(args.modality)
    bg = vol[bg_idx, :, :, z]

    print(f"WT pixels : {int(wt.sum()):>5}")
    print(f"TC pixels : {int(tc.sum()):>5}")
    print(f"ET pixels : {int(et.sum()):>5}")

    fig = plt.figure(figsize=(20, 9.5))
    fig.patch.set_facecolor('white')

    gs_top = gridspec.GridSpec(1, 4, figure=fig,
                               left=0.02, right=0.98,
                               top=0.90, bottom=0.52,
                               wspace=0.05)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.10, right=0.90,
                               top=0.46, bottom=0.08,
                               wspace=0.06)

    for c, (mod_idx, label) in enumerate(zip(range(4), MOD_LABELS)):
        ax = fig.add_subplot(gs_top[0, c])
        ax.imshow(vol[mod_idx, :, :, z], cmap='gray', vmin=0, vmax=1,
                  interpolation='bilinear')
        ax.set_title(label, fontsize=13, fontweight='bold', pad=6)
        ax.axis('off')

    mask_cfg = [
        ('WT\n(Whole Tumor)',   wt, _WT_COLOR),
        ('TC\n(Tumor Core)',    tc, _TC_COLOR),
        ('ET\n(Enhancing)',     et, _ET_COLOR),
    ]
    for c, (title, mask, color) in enumerate(mask_cfg):
        ax = fig.add_subplot(gs_bot[0, c])
        _overlay(ax, bg, mask, color)
        n_px = int(mask.sum())
        ax.set_title(f'{title}  ({n_px} px)',
                     fontsize=13, fontweight='bold', pad=6)
        ax.axis('off')

    legend_handles = [
        mpatches.Patch(color=_WT_COLOR, label='WT — Whole Tumor (NCR + ED + ET)'),
        mpatches.Patch(color=_TC_COLOR, label='TC — Tumor Core (NCR + ET)'),
        mpatches.Patch(color=_ET_COLOR, label='ET — Enhancing Tumor'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.01), edgecolor='#AAAAAA')

    fig.suptitle(
        f'BraTS Tumor Mask Visualization  |  {brats_id}  ({submitter})  |  {idh_str}  |  z = {z}',
        fontsize=12, fontweight='bold', y=0.97,
    )

    out_path = os.path.join(BASE_DIR, args.out)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n저장 완료: {out_path}")

if __name__ == '__main__':
    main()
