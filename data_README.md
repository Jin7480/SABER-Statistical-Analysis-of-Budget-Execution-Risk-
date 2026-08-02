# data 폴더 안내

이 폴더에는 용량이 큰 데이터 파일(원본 엑셀 zip 5개 + 전처리 캐시)이 들어갑니다.
GitHub 파일 용량 제한 때문에 저장소에는 직접 올리지 않고, 아래 **공개 Google Drive 폴더**에 두었습니다.

## 데이터 위치 (Google Drive, 공개)

https://drive.google.com/drive/folders/1455r3kIBEtm6gb0WTuAqWj4W6CmHADhR

폴더에 포함된 파일:

- `panel_cache.pkl.gz` — 전처리 완료 캐시 (엑셀 파싱 없이 즉시 재현용)
- `dataset_2016_2017.zip` ~ `dataset_2024_2025.zip` — 행정안전부 지방재정365
  「연도별 세부사업별 세출현황」 원본 (연 2개년 단위, 공공데이터포털 개방 데이터)

## 사용 방법

별도로 받을 필요 없습니다. 상위 폴더에서 아래를 실행하면, 이 `data/` 폴더가 비어 있을 때
`evaluate.py`가 위 Google Drive 폴더에서 자동으로 내려받습니다.

```bash
pip install pandas numpy openpyxl matplotlib gdown
python evaluate.py
```

직접 받고 싶다면 위 링크에서 파일을 내려받아 이 폴더에 넣으면 됩니다.
실행에는 캐시(`panel_cache.pkl.gz`) 하나만 있어도 충분하며, 원본 zip은 투명성·재현용입니다.
