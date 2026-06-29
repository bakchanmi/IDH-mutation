
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mri.mri_utils import (
    load_volume, select_top_slices,
    extract_full_slice, extract_roi_slice
)

META_COLS = [
    'case_id', 'Grade', 'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID',
    'submitter_id', 'BraTS_2019_subject_ID', 'IDH_label',
]

class MRIDataset(Dataset):

    def __init__(
        self,
        final_csv: str,
        top_k:  int = 10,
        img_size: int = 128,
        roi_size: int = 96,
        patient_ids: list[str] | None = None,
        train: bool = True,
        augment: bool = True,
    ):
        df = pd.read_csv(final_csv)
        if patient_ids is not None:
            df = df[df['submitter_id'].isin(patient_ids)]

        self.records = df[['submitter_id', 'BraTS_2019_subject_ID', 'IDH_label']].reset_index(drop=True)
        self.top_k    = top_k
        self.img_size = img_size
        self.roi_size = roi_size
        self.train    = train
        self.augment  = augment and train

        self._slice_cache: dict[str, np.ndarray] = {}

    def _get_slices(self, brats_id: str, seg: np.ndarray) -> np.ndarray:
        if brats_id not in self._slice_cache:
            self._slice_cache[brats_id] = select_top_slices(seg, k=self.top_k)
        return self._slice_cache[brats_id]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row     = self.records.iloc[idx]
        brats_id = row['BraTS_2019_subject_ID']
        label   = int(row['IDH_label'])

        vol, seg = load_volume(brats_id)
        slices   = self._get_slices(brats_id, seg)

        if self.train:
            z = int(np.random.choice(slices))
        else:
            z = int(slices[len(slices) // 2])

        full = extract_full_slice(vol, z, self.img_size)
        roi  = extract_roi_slice(vol, seg, z, self.roi_size)

        if self.augment:
            full, roi = _augment(full, roi)

        return (
            torch.from_numpy(full),
            torch.from_numpy(roi),
            torch.tensor(label, dtype=torch.long),
        )

def get_all_slices(brats_id: str, top_k: int, img_size: int, roi_size: int):
    vol, seg = load_volume(brats_id)
    slices   = select_top_slices(seg, k=top_k)
    fulls, rois = [], []
    for z in slices:
        fulls.append(extract_full_slice(vol, int(z), img_size))
        rois.append(extract_roi_slice(vol, seg, int(z), roi_size))
    return (
        torch.from_numpy(np.stack(fulls)),
        torch.from_numpy(np.stack(rois)),
    )

def _augment(full: np.ndarray, roi: np.ndarray):
    if np.random.rand() < 0.5:
        full = full[:, :, ::-1].copy()
        roi  = roi[:, :, ::-1].copy()
    if np.random.rand() < 0.5:
        full = full[:, ::-1, :].copy()
        roi  = roi[:, ::-1, :].copy()
    return full, roi
