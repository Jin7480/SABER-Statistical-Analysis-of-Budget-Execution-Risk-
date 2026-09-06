# -*- coding: utf-8 -*-
"""
compare_models.py — 5개 방법 비교 (국방부 시점 예측)

공정 비교 원칙: 모든 모델에 동일한 '원본 피처'(시점별 집행률·예산)를 제공한다.
세이버-K만 여기에 '계층 파생 피처(앵커·gap·계층평균)'를 추가로 설계해 넣는다.
즉 차이는 알고리즘이 아니라 '계층 구조를 피처로 설계했는가'에 있다.

비교 대상:
  ① 계층 베이지안 + 트리 (세이버-K)   — 계층 피처 + XGBoost
  ② 트리 머신러닝 (XGBoost)            — 원본 피처만
  ③ 계층 베이지안 단독                  — 계층 앵커만 (부스팅 없음)
  ④ 딥러닝 (MLP 신경망)                — 원본 피처
  ⑤ 이외의 머신러닝 (선형회귀 Ridge)    — 원본 피처
"""
import numpy as np
import pandas as pd

from saber_temporal import SaberTemporal


def _rmse(pred, y):
    e = (pred - y)
    e = e[~np.isnan(e)]
    return float(np.sqrt(np.mean(e ** 2)))


def _raw_features(D):
    """원본 피처: 시점별 누계 집행률(execA 및 있으면 exec6·exec3) + 로그예산."""
    cols = {'execA': D['execA'].values, 'logbud': np.log1p(D['예산현액'].astype(float)).values}
    for extra in ['exec6', 'exec3']:
        if extra in D.columns:
            cols[extra] = D[extra].values
    return pd.DataFrame(cols).fillna(0)


def run_comparison(train, valid, test):
    """5개 방법을 학습·예측하고 {방법: RMSE} 딕셔너리를 반환."""
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    ytr = train['집행률'].clip(0, 1).values
    yte = test['집행률'].clip(0, 1).values
    Xr_tr, Xr_te = _raw_features(train).values.astype(np.float32), _raw_features(test).values.astype(np.float32)

    results = {}

    # ① 세이버-K (계층 베이지안 + 트리)
    saber = SaberTemporal().fit(train)
    results['① 계층 베이지안+트리 (세이버-K)'] = _rmse(saber.predict(test), yte)

    # ③ 계층 베이지안 단독 (부스팅 없이 앵커만) — 세이버 내부 재사용
    results['③ 계층 베이지안 단독'] = _rmse(saber.anchor_only(test), yte)

    # ② 트리 머신러닝 (원본 피처만)
    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=2, random_state=0)
    m.fit(Xr_tr, ytr)
    results['② 트리 머신러닝 (XGBoost)'] = _rmse(np.clip(m.predict(Xr_te), 0, 1), yte)

    # ④ 딥러닝 (MLP)
    sc = StandardScaler()
    m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=0)
    m.fit(sc.fit_transform(Xr_tr), ytr)
    results['④ 딥러닝 (MLP 신경망)'] = _rmse(np.clip(m.predict(sc.transform(Xr_te)), 0, 1), yte)

    # ⑤ 이외의 머신러닝 (선형회귀)
    sc = StandardScaler()
    m = Ridge(alpha=1.0)
    m.fit(sc.fit_transform(Xr_tr), ytr)
    results['⑤ 이외의 머신러닝 (선형회귀)'] = _rmse(np.clip(m.predict(sc.transform(Xr_te)), 0, 1), yte)

    return results


def ablation_hierarchy(train, test):
    """계층 피처의 기여 확인: 같은 XGBoost를 계층 피처 유/무로 비교."""
    import xgboost as xgb
    ytr = train['집행률'].clip(0, 1).values
    yte = test['집행률'].clip(0, 1).values

    # 계층 없음
    Xr_tr, Xr_te = _raw_features(train).values.astype(np.float32), _raw_features(test).values.astype(np.float32)
    m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=2, random_state=0)
    m.fit(Xr_tr, ytr)
    no_hier = _rmse(np.clip(m.predict(Xr_te), 0, 1), yte)

    # 계층 있음 (세이버)
    saber = SaberTemporal().fit(train)
    with_hier = _rmse(saber.predict(test), yte)
    return {'계층 피처 없음 (순수 부스팅)': no_hier, '계층 피처 있음 (세이버-K)': with_hier}
