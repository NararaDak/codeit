# ==============================================
# data_check.py
# (Pandas DataFrame 기본 정보 확인 모듈)
# ==============================================

import pandas as pd

def basic_info(df):
    """DataFrame의 기초정보를 정리해 출력"""
    print('\n▶ 상위 5개 확인')
    print(df.head())

    print('\n▶ 데이터 전체 크기 (행, 열)')
    print(df.shape)

    print('\n▶ 데이터열(field, feature) 이름')
    print(df.columns.tolist())

    print('\n▶ 데이터 type(형식)')
    print(df.dtypes)

    print('\n▶ 데이터 info')
    df.info()

    print('\n▶ 문자열(object) 필드')
    string_cols = df.select_dtypes(include=['object']).columns
    print(string_cols.tolist())

    print('\n▶ 숫자열(number) 필드')
    numeric_cols = df.select_dtypes(include=['number']).columns
    print(numeric_cols.tolist())

    print('\n=== 완료 ===')
