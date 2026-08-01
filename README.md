# 세이버
**SABER (Statistical Analysis of Budget Execution Risk — 계층적 정규화 기반 예산 불용 조기경보 체계)**

2026 제3차 국방 AI 활용 아이디어 경연대회 출품작의 공개 데이터 검증 코드입니다.
군도(軍刀)를 뜻하는 이름 그대로, 연말에 발생할 예산 불용 위험을 분기 시점에 미리 베어내는 것을 목표로 합니다.

본 저장소의 data/ 데이터는 **행정안전부 지방재정365 공개 데이터**이며, 군 데이터가 아닙니다.

## 핵심 아이디어

부대×세부사업 단위의 예산 집행 이력은 극단적 소표본(조합당 수 년치)이라 딥러닝이 작동하지 않습니다.
세이버-K는 조직·예산 체계가 본래 지닌 **계층 구조**를 모형에 직접 반영하는
**계층적 정규화(Hierarchical Regularization, 부분 풀링)** 로 이 문제를 풉니다.

1. **교차 계층 앵커** — 같은 사업 유형의 타 기관 이력 + 소속 기관 특성 결합 → 이력 0년 신규 사업도 예측 (콜드 스타트 해소)
2. **정보량 비례 축소** — 이력이 짧을수록 집단 정보를, 길수록 자기 이력을 더 반영 (검증 연도에서 축소계수를 회귀로 추정)
3. **최신성 결합** — 지수 감쇠 가중으로 직전 연도 실적을 상태로 통합

GPU 불필요, 순수 오픈소스(Python), 모든 예측이 해석 가능 — 보안·설명책임이 요구되는 행정 환경을 위한 설계입니다.

## 검증 결과 (2025년 42.4만 개 세부사업 백테스트)

전국 지방자치단체 세출 결산 10개년: 2016-2025, 약 376만 건


2016-2023 학습 → 2024 검증(하이퍼파라미터·축소계수 선정) → 2025 시험의 엄격한 시간 분할.


| 예측오차 RMSE | SABER(계층) | 기존① 개별 추정 | 기존② 전년도 | 전체 평균 |
|---|---|---|---|---|
| 이력 보유 전체 | **0.170** | 0.213 (+25%) | 0.218 (+28%) | 0.226 (+33%) |
| 대규모(10억+) | **0.184** | 0.230 | 0.234 | 0.283 |

- 매년 전체 사업의 약 **20%는 이력 없는 신규 사업** → 기존 방식은 예측 자체가 불가능, SABER만 커버
- 심각 불용(집행률<60%) 사업의 **32%가 신규 사업** → 기존 방식은 문제의 1/3에 구조적으로 눈이 멂
- 분야별 개선 폭 최대: 국토·지역개발, 교통·물류, 문화·관광 등 **시설·인프라형 사업(24~25%)**

## 저장소 구조

```
├── README.md
├── baseline_models.py   # 기존 방식(개별 추정·전년도·전체 평균) 기준선
├── saber_model.py       # SABER-K 계층 모형 (경험적 베이즈 근사 구현)
├── evaluate.py          # 데이터 로드 → 백테스트 → 성능 표·그림 생성
└── data/
    ├── dataset_2016_2017.zip   # 행안부 지방재정365 연도별 세부사업별 세출현황
    ├── dataset_2018_2019.zip
    ├── dataset_2020_2021.zip
    ├── dataset_2022_2023.zip
    └── dataset_2024_2025.zip
```

## 실행 방법

```bash
pip install pandas numpy openpyxl matplotlib
# data/ 폴더에 zip 5개를 넣은 뒤 (자동 압축 해제됨)
python evaluate.py
```

첫 실행 시 xlsx 로드에 5~10분 소요되며 이후 캐시(`data/panel_cache.pkl`)를 사용합니다.
결과는 `results/` 폴더에 표(csv)와 그림(png)으로 저장됩니다.

## 데이터 출처

- 행정안전부, 지방재정365 「연도별 세부사업별 세출현황」, 공공데이터포털(data.go.kr) 개방 데이터
- 본 저장소에는 원본 xlsx를 연 2개년 단위 zip으로 재배포 (공공누리 개방 데이터)

## 참고 문헌

- Gelman & Hill (2007), *Data Analysis Using Regression and Multilevel/Hierarchical Models*, Cambridge Univ. Press
- Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on typical tabular data?", NeurIPS
- Shwartz-Ziv & Armon (2022), "Tabular data: Deep learning is not all you need", *Information Fusion*
- Makridakis et al. (2022), "M5 accuracy competition", *IJF*
- Rudin (2019), "Stop explaining black box machine learning models for high stakes decisions...", *Nature Machine Intelligence*
- Kim, Jeong, Kwak (2023), "HIER: Metric Learning Beyond Class Labels via Hierarchical Regularization", CVPR — 계층 구조의 명시적 반영 원리의 딥러닝 측 사례 (장기 확장 로드맵에서 비정형 데이터에 접목 검토)

## 유의사항
- 성능 수치는 위 백테스트 설계 기준이며 `evaluate.py`로 전 과정 재현 가능합니다.
