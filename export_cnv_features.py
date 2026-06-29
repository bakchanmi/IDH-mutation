
import os
import sys
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from genomics.cnv_features import build_cnv_features

GDC_DIR  = os.path.join(BASE_DIR, 'GDCdata')
UUID_MAP = os.path.join(GDC_DIR, 'uuid_to_patient_59.csv')
FINAL_CSV = os.path.join(BASE_DIR, 'TCGA_final_csv', 'final_dataset.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='cnv_features_4d.csv',
                        help='출력 CSV 경로 (default: cnv_features_4d.csv)')
    args = parser.parse_args()

    print("CNV 세그먼트 파일 읽는 중...")
    cnv_map = build_cnv_features(GDC_DIR, UUID_MAP)

    df_meta = pd.read_csv(FINAL_CSV, usecols=['submitter_id', 'IDH_label'])
    df_meta = df_meta[df_meta['submitter_id'].isin(cnv_map)].copy()

    rows = []
    for pid in sorted(cnv_map.keys()):
        vec = cnv_map[pid]
        rows.append({
            'patient_id':       pid,
            'mean_abs_cn_dev':  float(vec[0]),
            'gain_fraction':    float(vec[1]),
            'loss_fraction':    float(vec[2]),
            'loh_fraction':     float(vec[3]),
        })

    df_cnv = pd.DataFrame(rows)

    df_out = df_cnv.merge(
        df_meta.rename(columns={'submitter_id': 'patient_id'}),
        on='patient_id', how='left'
    )

    out_path = os.path.join(BASE_DIR, args.out)
    df_out.to_csv(out_path, index=False)

    print(f"\n저장 완료: {out_path}")
    print(f"환자 수: {len(df_out)}")
    print(f"\n컬럼 설명:")
    print(f"  mean_abs_cn_dev : mean(|CN_i - 2|)      전체 이수성(aneuploidy) 수준")
    print(f"  gain_fraction   : mean(CN_i > 2.3)      증폭된 염색체 비율")
    print(f"  loss_fraction   : mean(CN_i < 1.7)      결실된 염색체 비율")
    print(f"  loh_fraction    : mean(Minor_i < 0.3)   LOH(이형접합성 소실) 비율")
    print(f"\n피처 통계:")
    print(df_out[['mean_abs_cn_dev', 'gain_fraction', 'loss_fraction', 'loh_fraction']].describe().round(4))

    if 'IDH_label' in df_out.columns:
        print(f"\nIDH별 평균:")
        print(df_out.groupby('IDH_label')[
            ['mean_abs_cn_dev', 'gain_fraction', 'loss_fraction', 'loh_fraction']
        ].mean().round(4).rename(index={0: 'IDH- (wild)', 1: 'IDH+ (mutant)'}))

if __name__ == '__main__':
    main()
