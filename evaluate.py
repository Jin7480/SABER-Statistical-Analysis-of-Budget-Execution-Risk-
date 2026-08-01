# -*- coding: utf-8 -*-
"""
evaluate.py — 데이터 로드 → 백테스트 → 성능 표·그림 생성
사용법:
  1) data/ 폴더에 dataset_*.zip 압축 해제 (연도별 xlsx 10개)
  2) python evaluate.py
출력: results/ 폴더에 성능 표(csv)와 그림(png)

백테스트 설계: 2016~2023 학습 → 2024 검증(축소계수 추정) → 2025 시험
"""
import os, glob, zipfile
import numpy as np
import pandas as pd

from baseline_models import predict_baselines
from saber_model import SaberModel

DATA_DIR, RESULT_DIR, CACHE = 'data', 'results', 'data/panel_cache.pkl'
TRAIN_END, VAL_YEAR, TEST_YEAR = 2023, 2024, 2025
BINS = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 9)]
LABELS = ['신규(0년)', '1~2년', '3~4년', '5~6년', '7~9년']


# ---------- 데이터 준비 ----------
def load_panel() -> pd.DataFrame:
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    for z in glob.glob(f'{DATA_DIR}/dataset_*.zip'):
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_DIR)
    rows = []
    for path in sorted(glob.glob(f'{DATA_DIR}/*_세부사업별세출현황.xlsx')):
        year = int(os.path.basename(path)[:4])
        print(f'로드 중: {year} ...')
        df = pd.read_excel(path, header=None, skiprows=2,
                           usecols=[0, 1, 2, 3, 4, 9, 11, 12],
                           names=['지역', '자치단체', '회계', '사업명',
                                  '예산현액', '지출액', '분야', '부문'])
        df = df.dropna(subset=['지역', '사업명'])
        df['연도'] = year
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df['예산현액'] = pd.to_numeric(df['예산현액'], errors='coerce')
    df['지출액'] = pd.to_numeric(df['지출액'], errors='coerce')
    df = df.dropna(subset=['예산현액', '지출액'])
    df = df[df['예산현액'] >= 1_000_000]  # 100만원 미만 미세 항목 제외
    key = ['연도', '지역', '자치단체', '회계', '사업명']
    df = df.groupby(key, as_index=False).agg(
        {'예산현액': 'sum', '지출액': 'sum', '분야': 'first', '부문': 'first'})
    df['집행률'] = (df['지출액'] / df['예산현액']).clip(0, 1)
    df['uid'] = df['지역'] + '|' + df['자치단체'] + '|' + df['회계'].astype(str) + '|' + df['사업명']
    df.to_pickle(CACHE)
    return df


# ---------- 평가 ----------
def rmse(e):
    e = e.dropna()
    return float(np.sqrt((e ** 2).mean())) if len(e) else np.nan


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    df = load_panel()

    # SABER: 검증 연도로 축소계수 추정 후 시험 연도 예측
    saber = SaberModel().fit(df, TRAIN_END, VAL_YEAR)
    test = saber.predict(df, VAL_YEAR, TEST_YEAR)
    # 기준선 예측 결합
    test = predict_baselines(test, df[df.연도 <= VAL_YEAR])

    preds = {'SABER(계층 모형)': 'pred_saber',
             '기존① 개별 추정': 'pred_individual',
             '기존② 전년도': 'pred_last_year',
             '전체 평균': 'pred_global'}
    for name, col in preds.items():
        test[f'err_{col}'] = (test[col] - test['집행률']).abs()

    # 표 1: 이력 구간별 RMSE
    rows = []
    for (a, b), lab in zip(BINS, LABELS):
        m = (test.n >= a) & (test.n <= b)
        rows.append({'이력': lab, 'N': int(m.sum()),
                     **{name: rmse(test.loc[m, f'err_{col}']) for name, col in preds.items()}})
    t1 = pd.DataFrame(rows).round(4)
    t1.to_csv(f'{RESULT_DIR}/table1_rmse_by_history.csv', index=False, encoding='utf-8-sig')
    print('\n[이력 구간별 RMSE]\n', t1.to_string(index=False))

    # 표 2: 종합·대규모(10억+) RMSE
    h = test.n >= 1
    big = test['예산현액'] >= 1e9
    t2 = pd.DataFrame([
        {'구분': '이력 보유 전체', **{n: rmse(test.loc[h, f'err_{c}']) for n, c in preds.items()}},
        {'구분': '대규모(10억+)', **{n: rmse(test.loc[h & big, f'err_{c}']) for n, c in preds.items()}},
    ]).round(4)
    t2.to_csv(f'{RESULT_DIR}/table2_rmse_overall.csv', index=False, encoding='utf-8-sig')
    print('\n[종합 RMSE]\n', t2.to_string(index=False))

    # 표 3: 분야별 RMSE (SABER vs 기존 최선)
    rows = []
    for f in test['분야'].dropna().unique():
        m = (test['분야'] == f) & h
        if m.sum() < 5000:
            continue
        s = rmse(test.loc[m, 'err_pred_saber'])
        base = min(rmse(test.loc[m, 'err_pred_individual']), rmse(test.loc[m, 'err_pred_last_year']))
        rows.append({'분야': f, 'N': int(m.sum()), 'SABER': s, '기존(최선)': base,
                     '개선율%': round((1 - s / base) * 100, 1)})
    t3 = pd.DataFrame(rows).sort_values('개선율%', ascending=False).round(4)
    t3.to_csv(f'{RESULT_DIR}/table3_rmse_by_field.csv', index=False, encoding='utf-8-sig')
    print('\n[분야별 RMSE]\n', t3.to_string(index=False))

    # 신규 사업 커버리지
    n0 = ~h
    print(f"\n[콜드 스타트] 신규 사업 {n0.mean()*100:.1f}% — 기존 방식 예측 불가, "
          f"SABER RMSE {rmse(test.loc[n0,'err_pred_saber']):.4f} "
          f"vs 전체평균 {rmse(test.loc[n0,'err_pred_global']):.4f}")

    # 그림 (matplotlib 설치 시)
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(BINS)); w = 0.2
        colors = ['#4c72b0', '#c44e52', '#dd8452', '#999999']
        for j, (name, col) in enumerate(preds.items()):
            v = [t1.iloc[i][name] for i in range(len(BINS))]
            ax.bar(x + (j - 1.5) * w, v, w, label=name, color=colors[j])
        ax.set_xticks(x); ax.set_xticklabels(LABELS)
        ax.set_ylabel('RMSE'); ax.legend(); ax.grid(axis='y', alpha=0.3)
        ax.set_title(f'{TEST_YEAR}년 백테스트: 이력 구간별 예측오차')
        fig.tight_layout(); fig.savefig(f'{RESULT_DIR}/fig_rmse_by_history.png', dpi=150)
        print(f'\n그림 저장: {RESULT_DIR}/fig_rmse_by_history.png')
    except ImportError:
        print('matplotlib 미설치 — 그림 생략')


if __name__ == '__main__':
    main()
