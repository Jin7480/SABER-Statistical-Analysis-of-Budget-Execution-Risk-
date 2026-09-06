# -*- coding: utf-8 -*-
"""
military_data.py — 국방부 세부사업 월별 집행실적 데이터 로더

열린재정(openfiscaldata.go.kr) '재정사업 집행실적(월별집행실적)'에서 내려받은
국방부 소관 xlsx 파일들을 파싱한다. 정부 엑셀 특유의 스타일 태그가 openpyxl에서
오류를 일으키므로, 원시 XML을 직접 파싱하여 우회한다.

사용:
    from military_data import parse_files, build_asof_panel
    df = parse_files('data_mil/*.xlsx')       # 원자료 (사업×월)
    panel = build_asof_panel(df, as_of_month=9)  # 9월 시점 예측용 패널
"""
import zipfile, glob, os
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd

_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def _read_xlsx_raw(path):
    """스타일 파싱을 건너뛰고 셀 값만 원시 XML에서 추출."""
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        root = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in root.findall(f'{_NS}si'):
            shared.append(''.join(t.text or '' for t in si.iter(f'{_NS}t')))
    sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows = []
    for row in sheet.iter(f'{_NS}row'):
        cells = {}
        for c in row.findall(f'{_NS}c'):
            ref = c.get('r'); col = ''.join(ch for ch in ref if ch.isalpha())
            t = c.get('t'); v = c.find(f'{_NS}v')
            if v is None:
                val = None
            elif t == 's':
                val = shared[int(v.text)]
            else:
                val = v.text
            cells[col] = val
        rows.append(cells)
    return rows


def parse_files(file_glob):
    """여러 국방부 월별 xlsx를 하나의 DataFrame으로 병합.

    반환 컬럼: 회계연도, 집행월, 소관명, 회계명, 분야명, 부문명, 프로그램명,
    단위사업명, 세부사업명, 예산액, 예산현액, 집행액(당월), 집행액(연누계),
    그리고 파생 컬럼: 월(int), 연도(int), uid(회계|프로그램|단위사업|세부사업).
    """
    want = ['회계연도', '집행월', '소관명', '회계명', '분야명', '부문명', '프로그램명',
            '단위사업명', '세부사업명', '예산액', '예산현액', '집행액(당월)', '집행액(연누계)']
    all_rows = []
    for path in sorted(glob.glob(file_glob)):
        rows = _read_xlsx_raw(path)
        hdr = None
        for i, r in enumerate(rows):
            if any(v == '회계연도' for v in r.values()):
                hdr = i; break
        if hdr is None:
            continue
        colmap = {val: col for col, val in rows[hdr].items()}
        for r in rows[hdr + 1:]:
            rec = {w: (r.get(colmap[w]) if colmap.get(w) else None) for w in want}
            if rec['세부사업명']:
                all_rows.append(rec)
    df = pd.DataFrame(all_rows)
    for c in ['예산액', '예산현액', '집행액(당월)', '집행액(연누계)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['월'] = df['집행월'].astype(int)
    df['연도'] = df['회계연도'].astype(int)
    df['uid'] = df[['회계명', '프로그램명', '단위사업명', '세부사업명']].agg('|'.join, axis=1)
    return df


def build_asof_panel(df, as_of_month=9):
    """'as_of_month 시점 누계 집행률 → 연말 집행률' 예측용 패널을 만든다.

    각 (연도, 세부사업)에 대해:
      execA = as_of_month 누계집행액 / 예산현액   (예측 시점 정보)
      집행률 = 12월 누계집행액 / 예산현액         (예측 대상, 연말 실적)
    반환: 연도·uid·계층명·예산현액·execA·집행률 컬럼의 DataFrame.
    """
    recs = []
    for y in sorted(df['연도'].unique()):
        d = df[df['연도'] == y]
        piv = d.pivot_table(index='uid', columns='월', values='집행액(연누계)', aggfunc='last')
        if as_of_month not in piv.columns or 12 not in piv.columns:
            continue  # 미완성 연도(예: 진행 중인 당해연도)는 제외
        bud = d.groupby('uid')['예산현액'].last()
        meta = d.groupby('uid').agg(세부사업명=('세부사업명', 'last'), 프로그램명=('프로그램명', 'last'),
                                    회계명=('회계명', 'last'), 단위사업명=('단위사업명', 'last'),
                                    예산현액=('예산현액', 'last'))
        for u in bud[bud > 0].index:
            recs.append(dict(
                연도=y, uid=u, 세부사업명=meta.loc[u, '세부사업명'], 프로그램명=meta.loc[u, '프로그램명'],
                회계명=meta.loc[u, '회계명'], 단위사업명=meta.loc[u, '단위사업명'], 예산현액=meta.loc[u, '예산현액'],
                execA=min(piv[as_of_month].get(u, np.nan) / bud[u], 1.5),
                집행률=min(piv[12].get(u, np.nan) / bud[u], 1.5)))
    return pd.DataFrame(recs).dropna(subset=['execA', '집행률'])
