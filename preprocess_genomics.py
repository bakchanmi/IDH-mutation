
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GDC_DIR  = os.path.join(BASE_DIR, 'GDCdata')
UUID_MAP = os.path.join(GDC_DIR, 'uuid_to_patient_59.csv')
OUT_DIR  = os.path.join(BASE_DIR, 'preprocessed')
os.makedirs(OUT_DIR, exist_ok=True)

from genomics.cnv_features import build_cnv_features

print("CNV 피처 추출 중...")
cnv_map = build_cnv_features(GDC_DIR, UUID_MAP)
print(f"  완료: {len(cnv_map)}명의 환자")

cnv_out = os.path.join(OUT_DIR, 'cnv_features.pkl')
with open(cnv_out, 'wb') as f:
    pickle.dump(cnv_map, f)
print(f"  저장: {cnv_out}")

sample_patient = next(iter(cnv_map))
vec = cnv_map[sample_patient]
print(f"\n샘플 환자: {sample_patient}")
print(f"  CNV 벡터 shape: {vec.shape}")
print(f"  값 범위: {vec.min():.3f} ~ {vec.max():.3f}")
print(f"  처음 6값 (chr1 x3, chr2 x3): {vec[:6]}")
