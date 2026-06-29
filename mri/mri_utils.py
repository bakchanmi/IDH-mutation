
import os
import numpy as np
import nibabel as nib

MODALITIES  = ['t1', 't1ce', 't2', 'flair']
BRATS_BASE  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BraTS_raw_2019')

_NCR_NET = 1
_ED      = 2
_ET      = 4

def _wt_mask(seg): return (seg > 0).astype(np.uint8)
def _tc_mask(seg): return ((seg == _NCR_NET) | (seg == _ET)).astype(np.uint8)
def _et_mask(seg): return (seg == _ET).astype(np.uint8)

def find_brats_path(brats_id: str) -> str:
    for grade in ('HGG', 'LGG'):
        p = os.path.join(BRATS_BASE, grade, brats_id)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"BraTS subject not found: {brats_id}")

def load_volume(brats_id: str) -> tuple[np.ndarray, np.ndarray]:
    dirp = find_brats_path(brats_id)
    modalities = []
    for mod in MODALITIES:
        fpath = os.path.join(dirp, f'{brats_id}_{mod}.nii')
        data = nib.load(fpath).get_fdata(dtype=np.float32)
        data = _zscore_norm(data)
        modalities.append(data)

    vol = np.stack(modalities, axis=0)
    seg_path = os.path.join(dirp, f'{brats_id}_seg.nii')
    seg = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.uint8)
    return vol, seg

def _zscore_norm(vol: np.ndarray) -> np.ndarray:
    brain = vol[vol > 0]
    if brain.size == 0:
        return vol
    mu, sigma = brain.mean(), brain.std()
    if sigma < 1e-6:
        return vol
    out = (vol - mu) / sigma
    out = np.clip(out, -3, 3)
    out = (out + 3) / 6.0
    out[vol == 0] = 0.0
    return out.astype(np.float32)

def select_top_slices(seg: np.ndarray, k: int = 10) -> np.ndarray:
    wt = _wt_mask(seg)
    areas = wt.sum(axis=(0, 1))
    order = np.argsort(areas)[::-1]
    top   = order[:k]
    if areas[top].sum() == 0:
        center = seg.shape[2] // 2
        top = np.arange(center - k // 2, center + k // 2)
    return np.sort(top)

def extract_full_slice(vol: np.ndarray, z: int, img_size: int = 128) -> np.ndarray:
    from skimage.transform import resize
    slc = vol[:, :, :, z]
    slc = np.stack([
        resize(slc[c], (img_size, img_size), order=1,
               anti_aliasing=True, preserve_range=True)
        for c in range(4)
    ], axis=0).astype(np.float32)
    return slc

def extract_roi_slice(
    vol: np.ndarray,
    seg: np.ndarray,
    z: int,
    roi_size: int = 96,
    pad: int = 10,
) -> np.ndarray:
    from skimage.transform import resize

    slc_vol = vol[:, :, :, z]
    slc_seg = seg[:, :, z]

    wt = _wt_mask(slc_seg)
    tc = _tc_mask(slc_seg)
    et = _et_mask(slc_seg)

    ys, xs = np.where(wt > 0)
    if len(ys) == 0:
        x1, x2, y1, y2 = 0, slc_seg.shape[1], 0, slc_seg.shape[0]
    else:
        y1 = max(ys.min() - pad, 0)
        y2 = min(ys.max() + pad + 1, slc_seg.shape[0])
        x1 = max(xs.min() - pad, 0)
        x2 = min(xs.max() + pad + 1, slc_seg.shape[1])

    crop_vol = slc_vol[:, y1:y2, x1:x2]
    crop_wt  = wt[y1:y2, x1:x2][np.newaxis]
    crop_tc  = tc[y1:y2, x1:x2][np.newaxis]
    crop_et  = et[y1:y2, x1:x2][np.newaxis]

    def _resize(arr):
        return np.stack([
            resize(arr[c], (roi_size, roi_size), order=1,
                   anti_aliasing=True, preserve_range=True)
            for c in range(arr.shape[0])
        ], axis=0).astype(np.float32)

    crop = np.concatenate([crop_vol, crop_wt, crop_tc, crop_et], axis=0)
    return _resize(crop)
