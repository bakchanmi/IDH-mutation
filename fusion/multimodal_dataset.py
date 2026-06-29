
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from mri.mri_utils import load_volume, select_top_slices, extract_full_slice, extract_roi_slice
from genomics.cnv_features import build_cnv_features

META_COLS = [
    'case_id', 'Grade', 'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID',
    'submitter_id', 'BraTS_2019_subject_ID', 'IDH_label',
    'IDH1_x', 'IDH2_x', 'IDH1_y', 'IDH2_y',
    'ATRX_x', 'ATRX_y', 'TP53_x', 'TP53_y',
    'CIC_x',  'CIC_y',  'FUBP1_x', 'FUBP1_y',
]

K_MAJORITY  = 4
_K_MAJORITY = K_MAJORITY

class MultimodalDataset(Dataset):

    def __init__(
        self,
        final_csv: str,
        gdc_data_dir: str,
        uuid_map_csv: str,
        top_k: int = 10,
        img_size: int = 128,
        roi_size: int = 96,
        patient_ids: list[str] | None = None,
        train: bool = True,
        augment: bool = True,
        val_all_slices: bool = False,
        snv_k_best: int = 64,
        snv_scaler: StandardScaler | None = None,
        cnv_scaler: StandardScaler | None = None,
        snv_selector: SelectKBest | None = None,
    ):
        df = pd.read_csv(final_csv)
        all_snv_cols = [c for c in df.columns if c not in META_COLS]

        cnv_map = build_cnv_features(gdc_data_dir, uuid_map_csv)
        valid_ids = sorted(set(df['submitter_id'].values) & set(cnv_map.keys()))
        if patient_ids is not None:
            valid_ids = sorted(set(valid_ids) & set(patient_ids))

        df = df[df['submitter_id'].isin(valid_ids)].set_index('submitter_id').loc[valid_ids]
        labels_np = df['IDH_label'].values.astype(np.int64)

        snv_raw = df[all_snv_cols].values.astype(np.float32)

        if snv_selector is None:
            from functools import partial
            _mi = partial(mutual_info_classif, random_state=42, discrete_features=True)
            selector = SelectKBest(_mi, k=snv_k_best)
            selector.fit(snv_raw, labels_np)
            snv_selector = selector
        snv_raw = snv_selector.transform(snv_raw).astype(np.float32)

        if snv_scaler is None:
            snv_scaler = StandardScaler().fit(snv_raw)
        snv_raw = snv_scaler.transform(snv_raw).astype(np.float32)

        cnv_raw = np.stack([cnv_map[pid] for pid in valid_ids], axis=0)
        if cnv_scaler is None:
            cnv_scaler = StandardScaler().fit(cnv_raw)
        cnv_raw = cnv_scaler.transform(cnv_raw).astype(np.float32)

        self.snv = torch.from_numpy(snv_raw)
        self.cnv = torch.from_numpy(cnv_raw)
        self.labels    = torch.from_numpy(labels_np)
        self.brats_ids = df['BraTS_2019_subject_ID'].values.tolist()
        self.patient_ids  = valid_ids
        self.snv_scaler   = snv_scaler
        self.cnv_scaler   = cnv_scaler
        self.snv_selector = snv_selector
        self.snv_dim      = snv_raw.shape[1]
        self.cnv_dim      = cnv_raw.shape[1]
        selected_idx = snv_selector.get_support(indices=True)
        self.selected_snv_cols = [all_snv_cols[i] for i in selected_idx]

        self.top_k          = top_k
        self.img_size       = img_size
        self.roi_size       = roi_size
        self.train          = train
        self.augment        = augment and train
        self.val_all_slices = val_all_slices and not train

        self._samples: list[int] = []
        self._slice_keys: list[int] = []
        if train:
            for i, lbl in enumerate(labels_np):
                n_slices = top_k if lbl == 0 else _K_MAJORITY
                self._samples.extend([i] * n_slices)
                self._slice_keys.extend([-1] * n_slices)
        elif self.val_all_slices:
            for i in range(len(labels_np)):
                self._samples.extend([i] * top_k)
                self._slice_keys.extend(list(range(top_k)))
        else:
            self._samples   = list(range(len(labels_np)))
            self._slice_keys = [-1] * len(labels_np)

        self._slice_cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        patient_idx = self._samples[idx]
        brats_id = self.brats_ids[patient_idx]
        label    = self.labels[patient_idx]
        snv      = self.snv[patient_idx]
        cnv      = self.cnv[patient_idx]

        vol, seg = load_volume(brats_id)

        if brats_id not in self._slice_cache:
            self._slice_cache[brats_id] = select_top_slices(seg, k=self.top_k)
        slices = self._slice_cache[brats_id]

        slice_key = self._slice_keys[idx]
        if self.train:
            z = int(np.random.choice(slices))
        elif slice_key == -1:
            z = int(slices[len(slices) // 2])
        else:
            z = int(slices[min(slice_key, len(slices) - 1)])

        full = extract_full_slice(vol, z, self.img_size)
        roi  = extract_roi_slice(vol, seg, z, self.roi_size)

        if self.augment:
            full, roi = _augment(full, roi)

        return (
            torch.from_numpy(full),
            torch.from_numpy(roi),
            snv,
            cnv,
            label,
        )

def _augment(full: np.ndarray, roi: np.ndarray):
    if np.random.rand() < 0.5:
        full = full[:, :, ::-1].copy()
        roi  = roi[:, :, ::-1].copy()
    if np.random.rand() < 0.5:
        full = full[:, ::-1, :].copy()
        roi  = roi[:, ::-1, :].copy()
    k = np.random.randint(0, 4)
    if k > 0:
        full = np.rot90(full, k=k, axes=(1, 2)).copy()
        roi  = np.rot90(roi,  k=k, axes=(1, 2)).copy()
    if np.random.rand() < 0.5:
        full = np.clip(full + np.random.normal(0, 0.02, full.shape).astype(np.float32), 0, 1)
        roi  = np.clip(roi  + np.random.normal(0, 0.02, roi.shape).astype(np.float32),  0, 1)
    if np.random.rand() < 0.5:
        f = np.float32(np.random.uniform(0.85, 1.15))
        full = np.clip(full * f, 0, 1)
        roi  = np.clip(roi  * f, 0, 1)
    return full, roi
