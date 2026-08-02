# -*- coding: utf-8 -*-
"""
saber_model.py — 세이버-K(SABER-K): 계층적 정규화 기반 예산 집행률 예측 모형
Statistical Analysis of Budget Execution Risk

핵심 구성 (경험적 베이즈 근사 구현):
  1) 교차 계층 앵커 M: 전체 평균 + (동일 사업명 효과) + (소속 기관 효과)
     — 이력이 없는 신규 사업도 상위 계층 정보로 예측 (콜드 스타트 해소)
  2) 최신성 가중 개별 신호 m_ew: 지수 감쇠(λ) 가중 평균 — 직전 연도 정보를 상태로 통합
  3) 정보량 비례 축소: 이력 길이 구간별 축소계수 β를 '검증 연도'에서 회귀로 추정
     — 이력이 짧을수록 앵커(집단 정보) 비중이 커지는 부분 풀링(partial pooling)
  예측: pred = M + β_last(구간)·(last − M) + β_ew(구간)·(m_ew − M),  신규(n=0)는 pred = M
"""
import numpy as np
import pandas as pd

HIST_BINS = [(1, 1), (2, 2), (3, 4), (5, 6), (7, 99)]  # 이력 길이 구간


class SaberModel:
    def __init__(self, lam: float = 0.5, cP: float = 20, cG: float = 200):
        self.lam, self.cP, self.cG = lam, cP, cG
        self.betas = None  # 구간별 (β_last, β_ew)

    # ---------- 내부: 학습 데이터 통계 ----------
    def _stats(self, past: pd.DataFrame, train_end: int):
        past = past.sort_values('연도').copy()
        past['wt'] = self.lam ** (train_end - past['연도'])
        past['wy'] = past['wt'] * past['집행률']
        g = past.groupby('uid')
        unit = pd.DataFrame({
            'n': g['집행률'].size(),
            'm_ew': g['wy'].sum() / g['wt'].sum(),
            'last': g['집행률'].last()})
        aP = past.groupby('사업명')['집행률'].agg(['mean', 'size']); aP.columns = ['mP', 'sP']
        aG = past.groupby(['지역', '자치단체'])['집행률'].agg(['mean', 'size']); aG.columns = ['mG', 'sG']
        return past['집행률'].mean(), unit, aP, aG

    def _features(self, rows: pd.DataFrame, GLOBAL, unit, aP, aG) -> pd.DataFrame:
        t = rows.merge(unit, left_on='uid', right_index=True, how='left')
        t['n'] = t['n'].fillna(0).astype(int)
        t = t.merge(aP, left_on='사업명', right_index=True, how='left')
        t = t.merge(aG, left_on=['지역', '자치단체'], right_index=True, how='left')
        bP = t['sP'].fillna(0) / (t['sP'].fillna(0) + self.cP)
        bG = t['sG'].fillna(0) / (t['sG'].fillna(0) + self.cG)
        t['M'] = GLOBAL + bP * (t['mP'].fillna(GLOBAL) - GLOBAL) \
                        + bG * (t['mG'].fillna(GLOBAL) - GLOBAL)
        t['nb'] = t['n'].map(self._nbin)
        return t

    @staticmethod
    def _nbin(n):
        for i, (a, b) in enumerate(HIST_BINS):
            if a <= n <= b:
                return i
        return len(HIST_BINS) - 1

    # ---------- 학습: 검증 연도에서 구간별 축소계수 회귀 추정 ----------
    def fit(self, df: pd.DataFrame, train_end: int, val_year: int):
        """train_end 이전 자료를 학습, val_year 자료로 축소계수 β를 추정.
        시험 연도 데이터는 일절 접촉하지 않음 (시간 분할 검증 위생)."""
        GLOBAL, unit, aP, aG = self._stats(df[df.연도 <= train_end], train_end)
        val = self._features(df[df.연도 == val_year].copy(), GLOBAL, unit, aP, aG)
        val = val[val.n >= 1].dropna(subset=['last', 'm_ew'])
        self.betas = {}
        for i in range(len(HIST_BINS)):
            v = val[val.nb == i]
            X = np.column_stack([v['last'] - v['M'], v['m_ew'] - v['M']])
            y = (v['집행률'] - v['M']).values
            self.betas[i], *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    # ---------- 예측 ----------
    def predict(self, df: pd.DataFrame, train_end: int, test_year: int) -> pd.DataFrame:
        assert self.betas is not None, 'fit() 먼저 호출'
        GLOBAL, unit, aP, aG = self._stats(df[df.연도 <= train_end], train_end)
        t = self._features(df[df.연도 == test_year].copy(), GLOBAL, unit, aP, aG)
        b0 = np.array([self.betas[i][0] for i in range(len(HIST_BINS))])
        b1 = np.array([self.betas[i][1] for i in range(len(HIST_BINS))])
        t['pred_saber'] = np.where(
            t.n >= 1,
            t['M'] + b0[t['nb']] * (t['last'] - t['M']).fillna(0)
                   + b1[t['nb']] * (t['m_ew'] - t['M']).fillna(0),
            t['M'])  # 신규 사업: 교차 계층 앵커로 예측 (기존 방식은 예측 불가)
        return t


# ============================================================
# 캐시 생성 유틸리티 (독립 실행용, 자동 호출되지 않음)
# ============================================================
def build_cache(data_dir: str = 'data',
                cache_path: str = 'data/panel_cache.pkl',
                min_budget: int = 1_000_000) -> 'pd.DataFrame':
    """원본 엑셀(연도별 세부사업별 세출현황)을 전처리해 패널 캐시를 생성·저장한다.

    이 함수는 필요할 때 사용자가 직접 호출하는 유틸리티이며, 모듈 임포트 시
    자동으로 실행되지 않는다. evaluate.py의 load_panel()과 동일한 전처리
    (열 선택 → 정제 → 사업 단위 집계 → 집행률 계산)를 수행한다.

    처리 순서:
      1) data_dir 안의 dataset_*.zip 이 있으면 압축 해제
      2) *_세부사업별세출현황.xlsx 를 연도별로 순차 파싱 (메모리 절약: read-only 스트리밍)
      3) 연도별로 정제·집계 후 합쳐 집행률·uid 계산
      4) cache_path 로 저장

    매개변수:
      data_dir    : 엑셀(및 zip)이 있는 폴더
      cache_path  : 저장할 캐시 경로(.pkl)
      min_budget  : 이 금액 미만의 미세 예산 항목은 제외 (기본 100만 원)

    반환: 전처리된 pandas.DataFrame (저장과 동일한 내용)

    사용 예:
      python -c "from saber_model import build_cache; build_cache()"
    """
    import os, glob, zipfile
    from openpyxl import load_workbook

    # 1) zip 자동 해제
    for z in glob.glob(os.path.join(data_dir, 'dataset_*.zip')):
        with zipfile.ZipFile(z) as f:
            f.extractall(data_dir)

    # 2~3) 연도별 파싱 → 정제 → 집계
    parts = []
    xlsx_list = sorted(glob.glob(os.path.join(data_dir, '*_세부사업별세출현황.xlsx')))
    if not xlsx_list:
        raise FileNotFoundError(
            f"{data_dir} 에서 '*_세부사업별세출현황.xlsx' 를 찾을 수 없습니다. "
            f"엑셀 또는 dataset_*.zip 을 먼저 넣어주세요.")
    for path in xlsx_list:
        year = int(os.path.basename(path)[:4])
        print(f'로드 중: {year} ...', flush=True)
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        recs = []
        for r in ws.iter_rows(min_row=3, values_only=True):
            if r[0] is None or r[3] is None:
                continue
            recs.append((r[0], r[1], r[2], r[3], r[4], r[9], r[11], r[12]))
        wb.close()
        d = pd.DataFrame(recs, columns=['지역', '자치단체', '회계', '사업명',
                                        '예산현액', '지출액', '분야', '부문'])
        del recs
        d['예산현액'] = pd.to_numeric(d['예산현액'], errors='coerce')
        d['지출액'] = pd.to_numeric(d['지출액'], errors='coerce')
        d = d.dropna(subset=['예산현액', '지출액'])
        d = d[d['예산현액'] >= min_budget]
        d = d.groupby(['지역', '자치단체', '회계', '사업명'], as_index=False).agg(
            {'예산현액': 'sum', '지출액': 'sum', '분야': 'first', '부문': 'first'})
        d['연도'] = year
        parts.append(d)
        del d

    # 4) 합치기 → 집행률·uid → 저장
    df = pd.concat(parts, ignore_index=True)
    del parts
    df['집행률'] = (df['지출액'] / df['예산현액']).clip(0, 1)
    df['uid'] = (df['지역'] + '|' + df['자치단체'] + '|'
                 + df['회계'].astype(str) + '|' + df['사업명'])
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    df.to_pickle(cache_path)
    print(f'캐시 저장 완료: {cache_path}  ({len(df):,} 행)')
    return df


if __name__ == '__main__':
    # 이 파일을 직접 실행하면 캐시를 생성한다:  python saber_model.py
    build_cache()
