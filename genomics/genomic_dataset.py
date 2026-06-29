
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from genomics.cnv_features import build_cnv_features

META_COLS = [
    'case_id', 'Grade', 'BraTS_2017_subject_ID', 'BraTS_2018_subject_ID',
    'submitter_id', 'BraTS_2019_subject_ID', 'IDH_label',
    'IDH1_x', 'IDH2_x', 'IDH1_y', 'IDH2_y',
]

class GenomicDataset(Dataset):

    def __init__(
        self,
        final_csv: str,
        gdc_data_dir: str,
        uuid_map_csv: str,
        patient_ids: list[str] | None = None,
        snv_scaler: StandardScaler | None = None,
        cnv_scaler: StandardScaler | None = None,
    ):
        df = pd.read_csv(final_csv)
        snv_cols = [c for c in df.columns if c not in META_COLS]

        cnv_map = build_cnv_features(gdc_data_dir, uuid_map_csv)

        valid_ids = set(df['submitter_id'].values) & set(cnv_map.keys())
        if patient_ids is not None:
            valid_ids = valid_ids & set(patient_ids)
        valid_ids = sorted(valid_ids)

        df = df[df['submitter_id'].isin(valid_ids)].set_index('submitter_id')
        df = df.loc[valid_ids]

        snv_raw = df[snv_cols].values.astype(np.float32)
        cnv_raw = np.stack([cnv_map[pid] for pid in valid_ids], axis=0)
        labels  = df['IDH_label'].values.astype(np.int64)

        if snv_scaler is None:
            snv_scaler = StandardScaler()
            snv_raw = snv_scaler.fit_transform(snv_raw).astype(np.float32)
        else:
            snv_raw = snv_scaler.transform(snv_raw).astype(np.float32)

        if cnv_scaler is None:
            cnv_scaler = StandardScaler()
            cnv_raw = cnv_scaler.fit_transform(cnv_raw).astype(np.float32)
        else:
            cnv_raw = cnv_scaler.transform(cnv_raw).astype(np.float32)

        self.snv = torch.from_numpy(snv_raw)
        self.cnv = torch.from_numpy(cnv_raw)
        self.labels = torch.from_numpy(labels)
        self.patient_ids = valid_ids
        self.snv_scaler = snv_scaler
        self.cnv_scaler = cnv_scaler

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.snv[idx], self.cnv[idx], self.labels[idx]
