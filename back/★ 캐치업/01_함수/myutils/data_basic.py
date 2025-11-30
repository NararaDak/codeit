# ==============================================
# myutils/data_basic.py
# ==============================================
import pandas as pd

def check_missing(df):
    """결측치 개수 출력"""
    print("\n▶ 결측치 개수")
    print(df.isnull().sum())
    return df.isnull().sum()

def group_summary(df, group_col, target_col, func='mean'):
    """그룹별 요약 통계"""
    print(f"\n▶ 그룹별 '{target_col}' {func}")
    result = df.groupby(group_col)[target_col].agg(func)
    print(result)
    return result

def quick_stats(df):
    """기초 통계 요약"""
    print("\n▶ describe() 통계 요약")
    print(df.describe(include='all'))
    return df.describe(include='all')
