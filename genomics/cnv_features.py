
import os
import glob
import numpy as np
import pandas as pd

CHROMOSOMES = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6',
    'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
    'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18',
    'chr19', 'chr20', 'chr21', 'chr22', 'chrX'
]
_CNV_RAW_DIM = len(CHROMOSOMES) * 3
CNV_SUMMARY_DIM = 4

def _seg_to_vector(seg_df: pd.DataFrame) -> np.ndarray:
    vec = np.zeros(_CNV_RAW_DIM, dtype=np.float32)

    for i, chrom in enumerate(CHROMOSOMES):
        rows = seg_df[seg_df['Chromosome'] == chrom].copy()
        if rows.empty:
            vec[i * 3]     = 2.0
            vec[i * 3 + 1] = 1.0
            vec[i * 3 + 2] = 1.0
            continue

        lengths = (rows['End'] - rows['Start']).values.astype(np.float64)
        total = lengths.sum()
        if total == 0:
            lengths = np.ones(len(rows), dtype=np.float64)
            total = float(len(rows))

        w = lengths / total
        vec[i * 3]     = float(np.dot(w, rows['Copy_Number'].values))
        vec[i * 3 + 1] = float(np.dot(w, rows['Major_Copy_Number'].values))
        vec[i * 3 + 2] = float(np.dot(w, rows['Minor_Copy_Number'].values))

    return vec

def _summarize_vec(vec: np.ndarray) -> np.ndarray:
    cn    = vec[0::3]
    minor = vec[2::3]
    return np.array([
        np.mean(np.abs(cn - 2.0)),
        np.mean(cn > 2.3),
        np.mean(cn < 1.7),
        np.mean(minor < 0.3),
    ], dtype=np.float32)

def build_cnv_features(
    gdc_data_dir: str,
    uuid_map_csv: str,
) -> dict[str, np.ndarray]:
    utp = pd.read_csv(uuid_map_csv)
    aliquot_to_patient = dict(zip(utp['GDC_Aliquot'], utp['Patient_ID']))

    seg_files = glob.glob(os.path.join(gdc_data_dir, '*', '*.txt'))

    patient_vecs: dict[str, list[np.ndarray]] = {}

    for fpath in seg_files:
        try:
            df = pd.read_csv(fpath, sep='\t')
        except Exception:
            continue

        if 'GDC_Aliquot' not in df.columns:
            continue

        aliquot = df['GDC_Aliquot'].iloc[0]
        if aliquot not in aliquot_to_patient:
            continue

        patient_id = aliquot_to_patient[aliquot]
        vec = _seg_to_vector(df)

        patient_vecs.setdefault(patient_id, []).append(vec)

    return {pid: _summarize_vec(np.mean(vecs, axis=0)) for pid, vecs in patient_vecs.items()}
