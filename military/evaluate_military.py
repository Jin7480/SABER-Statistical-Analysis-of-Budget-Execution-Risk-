# -*- coding: utf-8 -*-
"""
evaluate_military.py — 국방부 시점 예측 백테스트 전체 실행

실행:
    python evaluate_military.py

전제: data_mil/ 폴더에 열린재정 국방부 월별 집행실적 xlsx 파일들이 있어야 한다.
(공개 데이터. 열린재정 openfiscaldata.go.kr '재정사업 집행실적(월별집행실적)',
 소관=국방부, 2020~2025년. 파일명은 자유.)

출력:
    - 시점별(3/6/9월) 예측오차 RMSE (통합모델 vs 보정없음)
    - 5개 방법 비교 + 계층 피처 ablation
    - 9월 예측구간 규모별 포함률
    - 9월 조기 식별률
"""
import numpy as np
import pandas as pd

from military_data import parse_files, build_asof_panel
from saber_temporal import SaberTemporal, conformal_interval
from compare_models import run_comparison, ablation_hierarchy

DATA_GLOB = 'data_mil/*.xlsx'
TRAIN_MAX, VALID_YEAR, TEST_YEAR = 2023, 2024, 2025   # 학습 ~2024, 검증 2024, 시험 2025


def _rmse(p, y):
    e = (p - y); e = e[~np.isnan(e)]
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    import os
    CACHE = 'data_mil/mil_panel.pkl'
    if os.path.exists(CACHE + '.gz'):
        import gzip, pickle
        with gzip.open(CACHE + '.gz', 'rb') as f:
            raw = pickle.load(f)             # 압축 전처리 캐시 (재현용)
    elif os.path.exists(CACHE):
        raw = pd.read_pickle(CACHE)          # 전처리 캐시 (재현용)
    else:
        raw = parse_files(DATA_GLOB)          # 원본 xlsx 파싱
        os.makedirs('data_mil', exist_ok=True)
        raw.to_pickle(CACHE)
    print(f"로드: {len(raw):,}행, 연도 {sorted(raw['연도'].unique())}")

    # ===== 실험 1: 시점별 RMSE =====
    print("\n[실험 1] 시점별 예측오차 (통합모델 vs 보정없음)")
    for m in [3, 6, 9]:
        P = build_asof_panel(raw, as_of_month=m)
        tr, te = P[P['연도'] <= TRAIN_MAX], P[P['연도'] == TEST_YEAR]
        if len(te) == 0:
            continue
        yte = te['집행률'].clip(0, 1).values
        saber = SaberTemporal().fit(tr)
        rmse_s = _rmse(saber.predict(te), yte)
        rmse_naive = _rmse(te['execA'].clip(0, 1).values, yte)
        print(f"  {m:>2}월: 세이버-K {rmse_s:.3f} | 보정없음 {rmse_naive:.3f}")

    # 9월 패널 (이후 실험 공통)
    P9 = build_asof_panel(raw, as_of_month=9)
    tr = P9[P9['연도'] <= TRAIN_MAX]
    va = P9[P9['연도'] == VALID_YEAR]
    te = P9[P9['연도'] == TEST_YEAR].reset_index(drop=True)
    yte = te['집행률'].clip(0, 1).values

    # ===== 실험 2: 5개 방법 비교 + ablation =====
    print("\n[실험 2] 5개 방법 비교 (9월 시점, RMSE)")
    for name, val in sorted(run_comparison(tr, va, te).items(), key=lambda x: x[1]):
        print(f"  {val:.3f}  {name}")
    print("  -- 계층 피처 ablation --")
    for name, val in ablation_hierarchy(tr, te).items():
        print(f"  {val:.3f}  {name}")

    # ===== 실험 3: 예측구간 규모별 포함률 =====
    print("\n[실험 3] 9월 예측구간(80%) 규모별 포함률")
    saber = SaberTemporal().fit(tr)
    lo, hi = conformal_interval(saber, va, te, coverage=0.8)
    pred = saber.predict(te)
    te = te.assign(pred=pred, lo=lo, hi=hi,
                   inside=(yte >= lo) & (yte <= hi),
                   bud_eok=te['예산현액'] / 100)   # 백만원→억
    for label, thr in [('전체', 0), ('100억+', 100), ('500억+', 500), ('1000억+', 1000)]:
        sub = te[te['bud_eok'] >= thr]
        cov = sub['inside'].mean() * 100
        rmse = _rmse(sub['pred'].values, sub['집행률'].clip(0, 1).values)
        print(f"  {label:>8}: n={len(sub):3d}  포함률 {cov:4.1f}%  RMSE {rmse:.3f}")
    wcov = te.loc[te['inside'], '예산현액'].sum() / te['예산현액'].sum() * 100
    print(f"  금액 가중 포함률: {wcov:.1f}%")

    # ===== 실험 4: 조기 식별률 =====
    print("\n[실험 4] 9월 시점 연말 불용 조기 식별률")
    actual_unspent = te['집행률'] < 0.9
    K = int(actual_unspent.sum())
    top = te['pred'].nsmallest(K).index
    recall = actual_unspent[top].sum() / K
    print(f"  연말 불용(집행률<90%) {K}건 / 전체 {len(te)}건")
    print(f"  예측 하위 {K}개 경보 → 실제 불용 적중률: {recall*100:.1f}%")


if __name__ == '__main__':
    main()
