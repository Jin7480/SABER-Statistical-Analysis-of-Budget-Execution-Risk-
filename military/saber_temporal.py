# -*- coding: utf-8 -*-
"""
saber_temporal.py — 세이버-K 시점 예측 모델 (국방부 월별 데이터용)

지자체용 saber_model.py 가 '연 단위 집행률'을 예측한다면, 이 모듈은 '분기 시점
(as-of) → 연말 집행률'을 예측한다. 핵심 아이디어는 동일하다:

  ① 계층 구조로 파생 피처를 만든다 (계층 앵커 M, 시점 gap)
     - 계층: 회계 → 프로그램 → 단위사업. 이력이 적은 하위 계층일수록 상위 정보로 수축.
     - partial pooling: 가중 w = n / (n + c), 상수 c가 클수록 상위 계층을 더 신뢰.
  ② 그 계층 피처를 그래디언트 부스팅 트리로 통합해 최종 예측.

즉 '계층 베이지안(경험적 베이즈 근사) 골격 + 트리 부스팅 헤드'의 통합 파이프라인.
계층 피처 없이 부스팅만 쓰면 성능이 떨어지므로(ablation 참조), 성능의 핵심은
알고리즘이 아니라 계층 구조 설계에 있다.
"""
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# 계층과 축소 상수 (상위일수록 표본이 많아 c를 크게)
_LEVELS = [('회계명', 50), ('프로그램명', 15), ('단위사업명', 8)]


class SaberTemporal:
    """계층 피처 생성 + 부스팅 통합. fit(학습) / predict(예측)."""

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, random_state=random_state, n_jobs=2)
        self.model = None
        self._tables = None

    # ---------- 계층 피처 구성 ----------
    def _fit_hier(self, train):
        t = train.copy()
        t['gap'] = t['집행률'] - t['execA']
        gl = t['gap'].median()
        tabs = [(col, c, t.groupby(col)['gap'].median(), t.groupby(col).size()) for col, c in _LEVELS]
        self._tables = dict(
            gl=gl, tabs=tabs,
            prog_m=train.groupby('프로그램명')['집행률'].mean().to_dict(),
            acc_m=train.groupby('회계명')['집행률'].mean().to_dict(),
            unit_m=train.groupby('단위사업명')['집행률'].mean().to_dict(),
            prog_gap=t.groupby('프로그램명')['gap'].mean().to_dict())

    def _anchor(self, D):
        """계층 앵커 M: 시점 집행률에 계층별 gap을 부분 풀링으로 더한다."""
        gl, tabs = self._tables['gl'], self._tables['tabs']
        out = []
        for _, row in D.iterrows():
            gp = gl
            for col, c, g, n in tabs:
                k = row[col]
                if k in g.index and pd.notna(g[k]):
                    w = n[k] / (n[k] + c)
                    gp = gp + w * (g[k] - gp)
            out.append(np.clip(row['execA'] + gp, 0, 1))
        return np.array(out)

    def _features(self, D):
        T = self._tables
        X = pd.DataFrame({
            'execA': D['execA'].values,
            'logbud': np.log1p(D['예산현액'].astype(float)).values})
        X['anchor'] = self._anchor(D)  # 계층 앵커 (가장 중요한 피처)
        X['prog_m'] = D['프로그램명'].map(T['prog_m']).values
        X['acc_m'] = D['회계명'].map(T['acc_m']).values
        X['unit_m'] = D['단위사업명'].map(T['unit_m']).values
        X['prog_gap'] = D['프로그램명'].map(T['prog_gap']).values
        return X.ffill().fillna(0)

    # ---------- 학습 / 예측 ----------
    def fit(self, train):
        if not _HAS_XGB:
            raise ImportError("xgboost가 필요합니다: pip install xgboost")
        self._fit_hier(train)
        X = self._features(train).values.astype(np.float32)
        y = train['집행률'].clip(0, 1).values
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, test):
        X = self._features(test).values.astype(np.float32)
        return np.clip(self.model.predict(X), 0, 1)

    def anchor_only(self, test):
        """부스팅 없이 계층 앵커만으로 예측 (= 계층 베이지안 단독, ablation용)."""
        return self._anchor(test)

    def feature_importance(self):
        """변수 중요도 (계층 피처가 전체의 대부분을 차지함을 확인)."""
        names = ['execA', 'logbud', 'anchor', 'prog_m', 'acc_m', 'unit_m', 'prog_gap']
        return dict(zip(names, self.model.feature_importances_.round(4).tolist()))


def conformal_interval(model, valid, test, coverage=0.8, low_exec_thresh=0.6):
    """검증연도 잔차로 예측구간을 만든다(conformal). 9월 집행률이 낮아 불확실한
    사업에는 더 넓은 구간을 부여. 반환: (lo, hi) 배열."""
    pv = model.predict(valid); yv = valid['집행률'].clip(0, 1).values
    resid = np.abs(yv - pv)
    if len(resid)==0:
        raise ValueError("검증 데이터가 비어 있습니다. 학습/검증 연도 분리를 확인하세요.")
    low_v = valid['execA'].values < low_exec_thresh
    half_lo = np.quantile(resid[low_v], coverage) if low_v.sum() >= 10 else np.quantile(resid, coverage)
    half_hi = np.quantile(resid[~low_v], coverage) if (~low_v).sum() >= 10 else np.quantile(resid, coverage)
    pt = model.predict(test)
    low_t = test['execA'].values < low_exec_thresh
    half = np.where(low_t, half_lo, half_hi)
    return np.clip(pt - half, 0, 1), np.clip(pt + half, 0, 1)
