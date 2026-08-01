# -*- coding: utf-8 -*-
"""
baseline_models.py — 기존 방식(비교 기준선) 모델
현행 실무 논리(과거·직전 연도 집행 실적 기반 판단)의 통계적 구현:
  1) 개별 추정: 사업 자신의 과거 평균 집행률
  2) 전년도 방식: 직전 연도 집행률
  3) 전체 평균: 전 사업 공통 평균 (완전 풀링, 참고용)
"""
import pandas as pd


def unit_history_stats(past: pd.DataFrame) -> pd.DataFrame:
    """uid별 과거 이력 통계 (n: 보유 연수, m: 평균 집행률, last: 직전 연도 집행률)"""
    past = past.sort_values('연도')
    g = past.groupby('uid')['집행률']
    return pd.DataFrame({'n': g.size(), 'm': g.mean(), 'v': g.var(), 'last': g.last()})


def predict_baselines(test: pd.DataFrame, past: pd.DataFrame) -> pd.DataFrame:
    """test 프레임에 세 기준선 예측 열을 추가해 반환.
    이력이 없는 신규 사업은 개별/전년도 예측이 NaN (예측 불가)."""
    unit = unit_history_stats(past)
    test = test.drop(columns=[c for c in ('n', 'm', 'v', 'last') if c in test.columns])
    out = test.merge(unit[['n', 'm', 'last']], left_on='uid', right_index=True, how='left')
    out['n'] = out['n'].fillna(0).astype(int)
    out['pred_individual'] = out['m']          # 기존 방식① 개별 추정(과거 평균)
    out['pred_last_year'] = out['last']        # 기존 방식② 전년도 집행률
    out['pred_global'] = past['집행률'].mean()  # 완전 풀링(참고)
    return out
