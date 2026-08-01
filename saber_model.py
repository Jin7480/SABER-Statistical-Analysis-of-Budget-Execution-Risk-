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
