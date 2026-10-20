"""
AI Data Insight Builder - 도구 카탈로그 (tools.py)
Pandas 기반 계산 전담 모듈 (Gemini에게는 계산을 시키지 않고 Pandas만 계산)

도구 목록:
1.  reshape_wide_to_long   연도 열을 행으로 접기
2.  quality_report         결측·중복·타입·이상한 열 이름
3.  year_coverage          연도별 값 개수와 권장 기준연도
4.  detect_aggregates      집계 행 후보 탐지
(도구 5~8, 9~12는 이후 단계에서 순차 추가)
"""

import math
import re
import numpy as np
import pandas as pd


# 표준 집계 행 탐지 키워드 (World Bank, UN 등 통계 데이터 공통)
DEFAULT_AGGREGATE_KEYWORDS = [
    "world", "oecd", "euro area", "european union", "income", "area",
    "total", "sub-saharan", "latin america", "caribbean", "middle east",
    "north america", "south asia", "east asia", "ida", "ibrd", "dividend",
    "africa eastern", "africa western", "small states", "demographic"
]


def sanitize_val(val):
    """NaN, None, Infinity 등을 안전하게 None 또는 기본값으로 변환하여 JSON 직렬화 보장"""
    if val is None:
        return None
    if isinstance(val, (float, int)):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def sanitize_dict_or_list(data):
    """딕셔너리 또는 리스트 내의 모든 결측/무한대 값을 안전하게 재귀 정제"""
    if isinstance(data, dict):
        return {k: sanitize_dict_or_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_dict_or_list(item) for item in data]
    else:
        return sanitize_val(data)


def find_year_columns(df):
    """4자리 연도 패턴(1900~2099)과 일치하는 열 목록 반환"""
    pattern = re.compile(r"^(19|20)\d{2}$")
    return [str(c) for c in df.columns if pattern.match(str(c).strip())]


def find_default_label_column(df):
    """국가명, 항목명 등 기준 레이블로 사용하기 적합한 문자열 열 반환"""
    cols = [str(c) for c in df.columns]
    for c in cols:
        c_lower = c.lower()
        if any(k in c_lower for k in ["country name", "국가명", "name", "country", "label", "항목명"]):
            return c
    # 없을 경우 첫 번째 object 타입 컬럼 반환
    obj_cols = [str(c) for c in df.select_dtypes(include=["object"]).columns]
    return obj_cols[0] if obj_cols else cols[0]


def filter_aggregates_df(df, label_column=None, exclude=True):
    """
    집계 행(World, 지역, 소득분위 등)을 분리하거나 제외하는 헬퍼 함수
    반환: (정제된_df, 제외된_행_수, 제외_이유)
    """
    if not exclude or label_column is None or label_column not in df.columns:
        return df, 0, "집계 행 제외 적용 안 됨"

    col_values = df[label_column].astype(str).str.lower()
    pattern = "|".join(DEFAULT_AGGREGATE_KEYWORDS)
    is_aggregate = col_values.str.contains(pattern, regex=True, na=False)

    excluded_count = int(is_aggregate.sum())
    filtered_df = df[~is_aggregate].copy()
    reason = f"국가 개별치가 아닌 다국가/지역/소득 그룹 집계 행 {excluded_count}개 제외"

    return filtered_df, excluded_count, reason


# -------------------------------------------------------------
# 도구 1: reshape_wide_to_long (연도 열을 행으로 접기)
# -------------------------------------------------------------
def reshape_wide_to_long(df, id_vars=None, value_vars=None, var_name="Year", value_name="Value"):
    """
    연도 열들이 가로로 나열된 광폭형(Wide) 데이터를 세로형(Long)으로 변환합니다.
    """
    year_cols = find_year_columns(df)

    if value_vars is None:
        value_vars = year_cols

    if id_vars is None:
        id_vars = [c for c in df.columns if c not in value_vars]

    if not value_vars:
        return {
            "error": "변환할 연도 열(숫자 4자리 형식)을 찾을 수 없습니다.",
            "tool": "reshape_wide_to_long"
        }

    initial_rows = len(df)
    melted = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name
    )

    # 결측치 정제 여부 파악
    total_melted = len(melted)
    non_null_melted = melted[melted[value_name].notna()].copy()
    excluded_nulls = total_melted - len(non_null_melted)

    # 미리보기 10행
    preview = non_null_melted.head(10).to_dict(orient="records")

    return {
        "tool": "reshape_wide_to_long",
        "reference_period": f"{min(value_vars)}~{max(value_vars)} (전체 연도)",
        "used_rows_count": len(non_null_melted),
        "excluded_rows_count": excluded_nulls,
        "exclusion_reason": f"변환 후 값이 비어있는(NaN) {excluded_nulls}개 행 제외",
        "used_columns": id_vars + [var_name, value_name],
        "unit": "행 (Long Format)",
        "caution": "연도별 결측치가 존재하는 행은 제외 후 미리보기를 제공했습니다.",
        "result": {
            "original_shape": [initial_rows, len(df.columns)],
            "melted_shape": [len(non_null_melted), len(non_null_melted.columns)],
            "year_range": [min(value_vars), max(value_vars)],
            "preview": sanitize_dict_or_list(preview)
        }
    }


# -------------------------------------------------------------
# 도구 2: quality_report (결측·중복·타입·이상한 열 이름)
# -------------------------------------------------------------
def quality_report(df):
    """
    데이터셋의 전반적인 데이터 품질(결측치 비율, 중복 행, 이상 컬럼명, 타입)을 종합 점검합니다.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    total_missing_cells = int(df.isna().sum().sum())
    missing_rate = round((total_missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())

    # 이상 컬럼명 감지 (공백, 특수문자, Unnamed)
    odd_columns = []
    for c in df.columns:
        c_str = str(c)
        if "unnamed" in c_str.lower():
            odd_columns.append({"column": c_str, "issue": "이름 없는 자동 생성 열"})
        elif c_str.strip() != c_str:
            odd_columns.append({"column": c_str, "issue": "양 끝에 불필요한 공백 포함"})
        elif not c_str:
            odd_columns.append({"column": c_str, "issue": "빈 열 이름"})

    # 결측치 상위 5개 열
    col_missing = df.isna().sum()
    top_missing_cols = [
        {"column": str(col), "missing_count": int(cnt), "percentage": round((cnt / total_rows * 100), 2)}
        for col, cnt in col_missing.sort_values(ascending=False).head(5).items()
        if cnt > 0
    ]

    return {
        "tool": "quality_report",
        "reference_period": "전체 데이터셋",
        "used_rows_count": total_rows,
        "excluded_rows_count": 0,
        "exclusion_reason": "전체 데이터 품질 검사를 위해 제외된 행 없음",
        "used_columns": [str(c) for c in df.columns],
        "unit": "건 / %",
        "caution": "결측치가 80% 이상인 열은 특정 분석 시 왜곡을 유발할 수 있으므로 주의해야 합니다.",
        "result": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "total_missing_cells": total_missing_cells,
            "overall_missing_rate_pct": missing_rate,
            "duplicate_rows": duplicate_rows,
            "odd_columns_count": len(odd_columns),
            "odd_columns": sanitize_dict_or_list(odd_columns),
            "top_missing_columns": sanitize_dict_or_list(top_missing_cols)
        }
    }


# -------------------------------------------------------------
# 도구 3: year_coverage (연도별 값 개수와 권장 기준연도)
# -------------------------------------------------------------
def year_coverage(df):
    """
    연도별 유효 데이터 개수를 집계하고, 가장 충실도가 높은 최신 권장 기준연도를 찾습니다.
    """
    year_cols = find_year_columns(df)
    if not year_cols:
        return {
            "error": "데이터셋에서 연도 열(4자리 숫자)을 감지할 수 없습니다.",
            "tool": "year_coverage"
        }

    total_rows = len(df)
    coverage_list = []
    labels = []
    counts = []

    for y in sorted(year_cols):
        cnt = int(df[y].notna().sum())
        pct = round((cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
        coverage_list.append({"year": y, "valid_count": cnt, "coverage_pct": pct})
        labels.append(y)
        counts.append(cnt)

    max_count = max(counts) if counts else 0
    threshold = max_count * 0.8

    # 최근 연도 중 충분한 데이터가 있는 연도
    recent_valid_years = [
        item["year"] for item in reversed(coverage_list)
        if item["valid_count"] >= threshold and item["valid_count"] > 0
    ]
    recommended_year = recent_valid_years[0] if recent_valid_years else (coverage_list[-1]["year"] if coverage_list else None)

    # Chart.js 시각화용 데이터
    chart_data = {
        "type": "bar",
        "labels": labels,
        "datasets": [
            {
                "label": "연도별 유효 데이터 건수",
                "data": counts,
                "backgroundColor": "#3b82f6"
            }
        ]
    }

    return {
        "tool": "year_coverage",
        "reference_period": f"{min(year_cols)}~{max(year_cols)}",
        "used_rows_count": total_rows,
        "excluded_rows_count": 0,
        "exclusion_reason": "전체 연도 충실도 계산을 위해 제외된 행 없음",
        "used_columns": year_cols,
        "unit": "개 (행 수)",
        "caution": "최신 연도일지라도 공공기관 집계 미완료로 결측치가 급증할 수 있으므로 권장 기준연도를 우선 고려하십시오.",
        "result": {
            "total_years": len(year_cols),
            "recommended_reference_year": recommended_year,
            "max_coverage_count": max_count,
            "coverage_table": sanitize_dict_or_list(coverage_list)
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 4: detect_aggregates (집계 행 후보 탐지)
# -------------------------------------------------------------
def detect_aggregates(df, label_column=None, keywords=None):
    """
    국가 개별 단위가 아닌 World, OECD, 소득 분위, 지역별 합계 등 집계 행을 탐지합니다.
    """
    if label_column is None or label_column not in df.columns:
        label_column = find_default_label_column(df)

    if keywords is None:
        keywords = DEFAULT_AGGREGATE_KEYWORDS

    total_rows = len(df)
    col_series = df[label_column].astype(str)
    
    detected = []
    for idx, val in col_series.items():
        v_lower = val.lower().strip()
        matched = [k for k in keywords if k in v_lower]
        if matched:
            detected.append({
                "row_index": int(idx),
                "label": val,
                "matched_keyword": matched[0]
            })

    detected_count = len(detected)
    country_like_count = total_rows - detected_count

    return {
        "tool": "detect_aggregates",
        "reference_period": "전체 데이터셋",
        "used_rows_count": country_like_count,
        "excluded_rows_count": detected_count,
        "exclusion_reason": f"집계성 그룹/지역 행 {detected_count}건 식별됨",
        "used_columns": [label_column],
        "unit": "건 (행 수)",
        "caution": "국가 간 순위 비교나 평균 계산 시 이 집계 행들을 포함하면 수치가 이중 계산되거나 심각하게 왜곡됩니다.",
        "result": {
            "label_column": label_column,
            "total_rows": total_rows,
            "aggregate_candidates_count": detected_count,
            "individual_candidates_count": country_like_count,
            "candidates": sanitize_dict_or_list(detected)
        }
    }


# -------------------------------------------------------------
# 도구 5: describe_numeric (개수·평균·중앙값·최소·최대·표준편차)
# -------------------------------------------------------------
def describe_numeric(df, reference_period=None, column=None, exclude_aggregates=True, label_column=None):
    """
    지정된 열(또는 기준연도)의 기본 기술통계(평균, 중앙값, 표준편차, 4분위수 등)를 계산합니다.
    """
    if label_column is None:
        label_column = find_default_label_column(df)

    # 타깃 열 결정 (reference_period 우선, 없으면 column, 없으면 첫 번째 숫자열)
    target_col = None
    if reference_period and str(reference_period) in df.columns:
        target_col = str(reference_period)
    elif column and str(column) in df.columns:
        target_col = str(column)
    else:
        # 권장 기준연도 또는 첫 번째 숫자형 열
        year_cols = find_year_columns(df)
        target_col = year_cols[-1] if year_cols else None

    if not target_col:
        return {
            "error": "분석할 수치형 열 또는 기준시점을 지정해야 합니다.",
            "tool": "describe_numeric"
        }

    # 1. 집계 행 제외 처리
    initial_rows = len(df)
    filtered_df, excl_agg, reason_agg = filter_aggregates_df(df, label_column, exclude=exclude_aggregates)

    # 2. 수치형 변환 및 결측치 제외
    numeric_series = pd.to_numeric(filtered_df[target_col], errors="coerce")
    valid_series = numeric_series.dropna()

    excl_null = len(filtered_df) - len(valid_series)
    total_excluded = excl_agg + excl_null
    exclusion_reasons = []
    if excl_agg > 0:
        exclusion_reasons.append(f"집계 행 {excl_agg}개")
    if excl_null > 0:
        exclusion_reasons.append(f"해당 시점 결측치 {excl_null}개")
    exclusion_str = ", ".join(exclusion_reasons) + " 제외" if exclusion_reasons else "제외된 행 없음"

    if valid_series.empty:
        return {
            "error": f"선택한 시점('{target_col}')에 유효한 수치 데이터가 존재하지 않습니다.",
            "tool": "describe_numeric"
        }

    count = int(valid_series.count())
    mean_val = round(float(valid_series.mean()), 3)
    std_val = round(float(valid_series.std()), 3) if count > 1 else 0.0
    min_val = round(float(valid_series.min()), 3)
    q25_val = round(float(valid_series.quantile(0.25)), 3)
    median_val = round(float(valid_series.median()), 3)
    q75_val = round(float(valid_series.quantile(0.75)), 3)
    max_val = round(float(valid_series.max()), 3)

    chart_data = {
        "type": "bar",
        "labels": ["최소값 (Min)", "하위 25% (Q1)", "중앙값 (Median)", "평균 (Mean)", "상위 75% (Q3)", "최대값 (Max)"],
        "datasets": [
            {
                "label": f"{target_col} 주요 통계 지표",
                "data": [min_val, q25_val, median_val, mean_val, q75_val, max_val],
                "backgroundColor": ["#64748b", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
            }
        ]
    }

    return {
        "tool": "describe_numeric",
        "reference_period": target_col,
        "used_rows_count": count,
        "excluded_rows_count": total_excluded,
        "exclusion_reason": exclusion_str,
        "used_columns": [target_col],
        "unit": "수치 단위 (원 데이터 기준)",
        "caution": "극단치(이상치)가 있는 경우 평균보다 중앙값을 주요 지표로 참고해야 합니다.",
        "result": {
            "column": target_col,
            "count": count,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "q25": q25_val,
            "median": median_val,
            "q75": q75_val,
            "max": max_val
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 6: trend_line (선택 항목들의 연도별 추이 - 선 차트)
# -------------------------------------------------------------
def trend_line(df, items=None, label_column=None, year_columns=None):
    """
    선택된 개별 항목(국가 등)의 연도별 시계열 변화 추이를 선 차트 데이터로 생성합니다.
    """
    if label_column is None:
        label_column = find_default_label_column(df)

    if year_columns is None:
        year_columns = find_year_columns(df)

    if not year_columns:
        return {
            "error": "시계열 추이를 분석할 연도 열을 찾을 수 없습니다.",
            "tool": "trend_line"
        }

    # 기본 비교 대상 항목 선정 (한국, 미국 등이 있으면 포함, 없으면 상위 4개)
    available_items = df[label_column].dropna().astype(str).unique().tolist()
    if not items:
        priority_candidates = ["Korea, Rep.", "United States", "China", "World", "Germany", "Japan"]
        chosen = [p for p in priority_candidates if p in available_items]
        if len(chosen) < 3:
            for itm in available_items:
                if itm not in chosen and not any(k in itm.lower() for k in ["income", "dividend"]):
                    chosen.append(itm)
                if len(chosen) >= 4:
                    break
        items = chosen

    # 선택된 항목 필터링
    df_items = df[df[label_column].astype(str).isin(items)].copy()
    used_items = df_items[label_column].tolist()
    excluded_items_count = len(items) - len(used_items)

    # Chart.js 데이터 세트 구성
    sorted_years = sorted(year_columns)
    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
    datasets = []

    for idx, (_, row) in enumerate(df_items.iterrows()):
        item_name = str(row[label_column])
        values = []
        for y in sorted_years:
            val = pd.to_numeric(row.get(y), errors="coerce")
            values.append(sanitize_val(round(float(val), 2) if pd.notna(val) else None))

        color = palette[idx % len(palette)]
        datasets.append({
            "label": item_name,
            "data": values,
            "borderColor": color,
            "backgroundColor": color,
            "fill": False,
            "tension": 0.2
        })

    chart_data = {
        "type": "line",
        "labels": sorted_years,
        "datasets": datasets
    }

    return {
        "tool": "trend_line",
        "reference_period": f"{min(sorted_years)}~{max(sorted_years)}",
        "used_rows_count": len(used_items),
        "excluded_rows_count": len(df) - len(used_items),
        "exclusion_reason": f"선택된 {len(used_items)}개 비교 항목 외 다른 행({len(df) - len(used_items)}개) 제외",
        "used_columns": [label_column] + sorted_years,
        "unit": "수치 (지표 기준)",
        "caution": "연도별 중간 결측치가 있는 경우 선이 끊어지거나 직전 값과 직선으로 연결될 수 있습니다.",
        "result": {
            "selected_items": used_items,
            "year_range": [min(sorted_years), max(sorted_years)],
            "data_points_per_item": len(sorted_years)
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 7: top_bottom_n (기준연도 기준 상위·하위 N - 수평 막대)
# -------------------------------------------------------------
def top_bottom_n(df, reference_period, n=10, label_column=None, exclude_aggregates=True, mode="top"):
    """
    [필수 규칙: 기준시점(reference_period) 필수 파라미터]
    지정된 기준시점에서 상위 N개 및 하위 N개 항목을 계산하고 수평 막대 차트 데이터를 생성합니다.
    mode: "top" (상위 N개 차트), "bottom" (하위 N개 차트)
    """
    if not reference_period:
        return {
            "error": "도구 7(top_bottom_n)은 기준 시점(reference_period)이 필수입니다.",
            "tool": "top_bottom_n"
        }

    ref_col = str(reference_period).strip()
    if ref_col not in df.columns:
        return {
            "error": f"데이터셋에서 기준 시점 열 '{ref_col}'을 찾을 수 없습니다.",
            "tool": "top_bottom_n"
        }

    if label_column is None:
        label_column = find_default_label_column(df)

    # 1. 집계 행 제외 필터링
    initial_rows = len(df)
    filtered_df, excl_agg, reason_agg = filter_aggregates_df(df, label_column, exclude=exclude_aggregates)

    # 2. 수치형 변환 및 결측치 제외
    val_series = pd.to_numeric(filtered_df[ref_col], errors="coerce")
    valid_mask = val_series.notna()
    valid_df = filtered_df[valid_mask].copy()
    valid_df["__val__"] = val_series[valid_mask]

    excl_null = len(filtered_df) - len(valid_df)
    total_excluded = excl_agg + excl_null
    reason_parts = []
    if excl_agg > 0:
        reason_parts.append(f"집계 행 {excl_agg}개")
    if excl_null > 0:
        reason_parts.append(f"{ref_col}년 결측 행 {excl_null}개")
    exclusion_reason = ", ".join(reason_parts) + " 제외" if reason_parts else "제외된 행 없음"

    if valid_df.empty:
        return {
            "error": f"기준 시점 '{ref_col}'에 값이 있는 유효한 행이 없습니다.",
            "tool": "top_bottom_n"
        }

    # 3. 상위 N 및 하위 N 계산
    top_df = valid_df.sort_values(by="__val__", ascending=False).head(n)
    bottom_df = valid_df.sort_values(by="__val__", ascending=True).head(n)

    top_list = [
        {"rank": i + 1, "label": str(r[label_column]), "value": round(float(r["__val__"]), 3)}
        for i, (_, r) in enumerate(top_df.iterrows())
    ]
    bottom_list = [
        {"rank": i + 1, "label": str(r[label_column]), "value": round(float(r["__val__"]), 3)}
        for i, (_, r) in enumerate(bottom_df.iterrows())
    ]

    # Chart.js 수평 막대 차트 (mode에 따라 상위 또는 하위)
    if str(mode).lower() == "bottom":
        chart_data = {
            "type": "bar",
            "indexAxis": "y",
            "labels": [item["label"] for item in reversed(bottom_list)],
            "datasets": [
                {
                    "label": f"{ref_col}년 하위 {len(bottom_list)}개국 (최저)",
                    "data": [item["value"] for item in reversed(bottom_list)],
                    "backgroundColor": "#f43f5e"
                }
            ]
        }
    else:
        chart_data = {
            "type": "bar",
            "indexAxis": "y",
            "labels": [item["label"] for item in reversed(top_list)],
            "datasets": [
                {
                    "label": f"{ref_col}년 상위 {len(top_list)}개국",
                    "data": [item["value"] for item in reversed(top_list)],
                    "backgroundColor": "#3b82f6"
                }
            ]
        }

    return {
        "tool": "top_bottom_n",
        "reference_period": ref_col,
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "exclusion_reason": exclusion_reason,
        "used_columns": [label_column, ref_col],
        "unit": "수치 (지표 기준)",
        "caution": f"기준시점({ref_col})에 값이 누락된 국가는 순위 집계에서 제외되었습니다.",
        "result": {
            "reference_period": ref_col,
            "n": n,
            "total_valid_evaluated": len(valid_df),
            "top_n": sanitize_dict_or_list(top_list),
            "bottom_n": sanitize_dict_or_list(bottom_list)
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 8: group_summary (그룹별 평균·합계·개수 - 막대)
# -------------------------------------------------------------
def group_summary(df, reference_period, group_column=None, value_column=None, agg_func="mean", exclude_aggregates=True):
    """
    [필수 규칙: 기준시점(reference_period) 필수 파라미터]
    그룹 열을 기준으로 기준시점의 지표를 그룹별 평균/합계/개수로 집계합니다.
    """
    if not reference_period:
        return {
            "error": "도구 8(group_summary)은 기준 시점(reference_period)이 필수입니다.",
            "tool": "group_summary"
        }

    val_col = str(reference_period).strip() if value_column is None else str(value_column).strip()
    if val_col not in df.columns:
        return {
            "error": f"데이터셋에서 집계할 수치 열 '{val_col}'을 찾을 수 없습니다.",
            "tool": "group_summary"
        }

    # 그룹 열이 지정되지 않은 경우 적합한 범주형 열 탐색
    if group_column is None or group_column not in df.columns:
        cat_candidates = [
            c for c in df.select_dtypes(include=["object"]).columns
            if c != val_col and 2 <= df[c].nunique() <= 30
        ]
        group_column = cat_candidates[0] if cat_candidates else None

    # 만약 적합한 그룹 컬럼이 없으면 지역 또는 소득 키워드 기반 가상 그룹 분류
    label_col = find_default_label_column(df)
    temp_df = df.copy()
    if group_column is None:
        # 그룹 컬럼이 없을 때: 국가명 앞글자 또는 단순 균등 분할
        temp_df["__group__"] = temp_df[label_col].astype(str).str[0].str.upper()
        group_column = "__group__"

    # 집계 행 제외
    filtered_df, excl_agg, _ = filter_aggregates_df(temp_df, label_col, exclude=exclude_aggregates)

    # 결측치 정제
    s = pd.to_numeric(filtered_df[val_col], errors="coerce")
    valid_mask = s.notna()
    valid_df = filtered_df[valid_mask].copy()
    valid_df["__num_val__"] = s[valid_mask]

    excl_null = len(filtered_df) - len(valid_df)
    total_excluded = excl_agg + excl_null

    if valid_df.empty:
        return {
            "error": f"기준 시점 '{val_col}'에 유효한 데이터가 없습니다.",
            "tool": "group_summary"
        }

    # 그룹별 집계
    grouped = valid_df.groupby(group_column)["__num_val__"].agg(["count", "mean", "sum", "median"]).reset_index()
    grouped = grouped.sort_values(by="mean", ascending=False).head(15)

    summary_list = [
        {
            "group": str(r[group_column]),
            "count": int(r["count"]),
            "mean": round(float(r["mean"]), 2),
            "sum": round(float(r["sum"]), 2),
            "median": round(float(r["median"]), 2)
        }
        for _, r in grouped.iterrows()
    ]

    chart_data = {
        "type": "bar",
        "labels": [item["group"] for item in summary_list],
        "datasets": [
            {
                "label": f"그룹별 {agg_func} ({val_col})",
                "data": [item["mean" if agg_func == "mean" else "sum"] for item in summary_list],
                "backgroundColor": "#10b981"
            }
        ]
    }

    return {
        "tool": "group_summary",
        "reference_period": val_col,
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "exclusion_reason": f"집계 행 {excl_agg}개 및 결측치 {excl_null}개 제외",
        "used_columns": [group_column, val_col],
        "unit": "수치 (지표 기준)",
        "caution": "그룹별 표본 수(count) 차이가 크면 단순 평균 해석에 주의가 필요합니다.",
        "result": {
            "group_column": group_column,
            "aggregation": agg_func,
            "summary_table": sanitize_dict_or_list(summary_list)
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 9: change_rate (두 시점 사이 변화량·변화율 - 막대)
# -------------------------------------------------------------
def change_rate(df, reference_period, compare_period=None, n=10, label_column=None, exclude_aggregates=True, mode="top"):
    """
    [필수 규칙: 기준시점(reference_period) 필수 파라미터]
    두 시점(예: 2024년 vs 2014년) 사이의 증감량(변화량)과 증감율(변화율 %)을 계산합니다.
    mode: "top" (상위 증가 N개), "bottom" (하위 증가/감소 N개)
    """
    if not reference_period:
        return {
            "error": "도구 9(change_rate)는 기준 시점(reference_period)이 필수입니다.",
            "tool": "change_rate"
        }

    ref_col = str(reference_period).strip()
    if ref_col not in df.columns:
        return {
            "error": f"데이터셋에서 기준 시점 열 '{ref_col}'을 찾을 수 없습니다.",
            "tool": "change_rate"
        }

    # 비교 시점(compare_period) 결정 (기본값: 기준연도 10년 전 또는 5년 전 또는 이전 연도)
    if compare_period is None:
        year_cols = find_year_columns(df)
        if ref_col in year_cols:
            idx = year_cols.index(ref_col)
            # 10년 전 또는 5년 전
            if idx >= 10:
                compare_period = year_cols[idx - 10]
            elif idx >= 5:
                compare_period = year_cols[idx - 5]
            elif idx > 0:
                compare_period = year_cols[idx - 1]
            else:
                compare_period = year_cols[-1]
        else:
            return {
                "error": "비교할 이전 시점(compare_period)을 지정해야 합니다.",
                "tool": "change_rate"
            }

    comp_col = str(compare_period).strip()
    if comp_col not in df.columns:
        return {
            "error": f"데이터셋에서 비교 시점 열 '{comp_col}'을 찾을 수 없습니다.",
            "tool": "change_rate"
        }

    if label_column is None:
        label_column = find_default_label_column(df)

    # 집계 행 제외
    filtered_df, excl_agg, _ = filter_aggregates_df(df, label_column, exclude=exclude_aggregates)

    # 두 시점 모두 유효한 행 필터링
    s_ref = pd.to_numeric(filtered_df[ref_col], errors="coerce")
    s_comp = pd.to_numeric(filtered_df[comp_col], errors="coerce")
    valid_mask = s_ref.notna() & s_comp.notna()

    valid_df = filtered_df[valid_mask].copy()
    valid_df["__ref__"] = s_ref[valid_mask]
    valid_df["__comp__"] = s_comp[valid_mask]
    valid_df["__diff__"] = valid_df["__ref__"] - valid_df["__comp__"]
    
    # 0으로 나누기 방지
    valid_df["__rate__"] = np.where(
        valid_df["__comp__"] != 0,
        (valid_df["__diff__"] / valid_df["__comp__"].abs()) * 100,
        np.nan
    )

    excl_null = len(filtered_df) - len(valid_df)
    total_excluded = excl_agg + excl_null

    if valid_df.empty:
        return {
            "error": f"두 시점('{comp_col}'과 '{ref_col}')에 모두 값이 존재하는 행이 없습니다.",
            "tool": "change_rate"
        }

    # 증감량 상위 N개 및 하위 N개
    top_diff = valid_df.sort_values(by="__diff__", ascending=False).head(n)
    bottom_diff = valid_df.sort_values(by="__diff__", ascending=True).head(n)

    top_list = [
        {
            "rank": i + 1,
            "label": str(r[label_column]),
            "base_val": round(float(r["__comp__"]), 2),
            "ref_val": round(float(r["__ref__"]), 2),
            "diff": round(float(r["__diff__"]), 2),
            "rate_pct": round(float(r["__rate__"]), 2) if pd.notna(r["__rate__"]) else None
        }
        for i, (_, r) in enumerate(top_diff.iterrows())
    ]

    bottom_list = [
        {
            "rank": i + 1,
            "label": str(r[label_column]),
            "base_val": round(float(r["__comp__"]), 2),
            "ref_val": round(float(r["__ref__"]), 2),
            "diff": round(float(r["__diff__"]), 2),
            "rate_pct": round(float(r["__rate__"]), 2) if pd.notna(r["__rate__"]) else None
        }
        for i, (_, r) in enumerate(bottom_diff.iterrows())
    ]

    if str(mode).lower() == "bottom":
        chart_data = {
            "type": "bar",
            "labels": [item["label"] for item in bottom_list],
            "datasets": [
                {
                    "label": f"변화량 최하위/감소 ({comp_col} -> {ref_col})",
                    "data": [item["diff"] for item in bottom_list],
                    "backgroundColor": "#f43f5e"
                }
            ]
        }
    else:
        chart_data = {
            "type": "bar",
            "labels": [item["label"] for item in top_list],
            "datasets": [
                {
                    "label": f"변화량 상위 ({comp_col} -> {ref_col})",
                    "data": [item["diff"] for item in top_list],
                    "backgroundColor": "#10b981"
                }
            ]
        }

    return {
        "tool": "change_rate",
        "reference_period": f"{comp_col} 대비 {ref_col}",
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "exclusion_reason": f"집계 행 {excl_agg}개 및 두 시점 중 결측치가 있는 {excl_null}개 행 제외",
        "used_columns": [label_column, comp_col, ref_col],
        "unit": "수치 변화량 (p% 또는 단위)",
        "caution": "초기 기준값(분모)이 매우 작은 경우 변화율(%)이 비정상적으로 과대하게 보일 수 있습니다.",
        "result": {
            "base_period": comp_col,
            "reference_period": ref_col,
            "evaluated_count": len(valid_df),
            "top_growers": sanitize_dict_or_list(top_list),
            "top_decliners": sanitize_dict_or_list(bottom_list)
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 10: distribution (구간별 분포 - 히스토그램)
# -------------------------------------------------------------
def distribution(df, reference_period, bins=5, exclude_aggregates=True, label_column=None):
    """
    [필수 규칙: 기준시점(reference_period) 필수 파라미터]
    기준시점의 수치 데이터를 등간격 구간(bins)으로 나누어 도수분포표와 히스토그램 차트를 생성합니다.
    """
    if not reference_period:
        return {
            "error": "도구 10(distribution)은 기준 시점(reference_period)이 필수입니다.",
            "tool": "distribution"
        }

    ref_col = str(reference_period).strip()
    if ref_col not in df.columns:
        return {
            "error": f"데이터셋에서 기준 시점 열 '{ref_col}'을 찾을 수 없습니다.",
            "tool": "distribution"
        }

    if label_column is None:
        label_column = find_default_label_column(df)

    filtered_df, excl_agg, _ = filter_aggregates_df(df, label_column, exclude=exclude_aggregates)

    s = pd.to_numeric(filtered_df[ref_col], errors="coerce").dropna()
    excl_null = len(filtered_df) - len(s)
    total_excluded = excl_agg + excl_null

    if s.empty:
        return {
            "error": f"기준 시점 '{ref_col}'에 유효한 수치 데이터가 없습니다.",
            "tool": "distribution"
        }

    counts, bin_edges = np.histogram(s, bins=bins)
    labels = []
    hist_table = []

    for i in range(len(counts)):
        low = round(float(bin_edges[i]), 1)
        high = round(float(bin_edges[i + 1]), 1)
        bin_label = f"{low} ~ {high}"
        cnt = int(counts[i])
        pct = round((cnt / len(s) * 100), 1)
        labels.append(bin_label)
        hist_table.append({
            "range": bin_label,
            "min_edge": low,
            "max_edge": high,
            "count": cnt,
            "percentage": pct
        })

    chart_data = {
        "type": "bar",
        "labels": labels,
        "datasets": [
            {
                "label": f"{ref_col}년 데이터 분포 (개수)",
                "data": [item["count"] for item in hist_table],
                "backgroundColor": "#8b5cf6"
            }
        ]
    }

    return {
        "tool": "distribution",
        "reference_period": ref_col,
        "used_rows_count": len(s),
        "excluded_rows_count": total_excluded,
        "exclusion_reason": f"집계 행 {excl_agg}개 및 결측치 {excl_null}개 제외",
        "used_columns": [ref_col],
        "unit": "개 (항목 빈도)",
        "caution": "특정 구간에 빈도가 과도하게 집중되어 있는지 왜도(치우침)를 함께 고려해야 합니다.",
        "result": {
            "bins_count": bins,
            "min_val": round(float(s.min()), 2),
            "max_val": round(float(s.max()), 2),
            "mean_val": round(float(s.mean()), 2),
            "median_val": round(float(s.median()), 2),
            "histogram": sanitize_dict_or_list(hist_table)
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 11: correlation_scatter (두 숫자 열 상관계수 - 산점도)
# -------------------------------------------------------------
def correlation_scatter(df, reference_period, col_x=None, col_y=None, label_column=None, exclude_aggregates=True):
    """
    [필수 규칙: 기준시점(reference_period) 필수 파라미터]
    [절대 규칙: 상관계수 계산 시 '상관관계는 인과관계가 아니다'를 반드시 주의사항에 명시]
    두 수치형 지표(또는 두 시점)의 피어슨 상관계수를 계산하고 산점도 차트 데이터를 생성합니다.
    """
    if not reference_period:
        return {
            "error": "도구 11(correlation_scatter)은 기준 시점(reference_period)이 필수입니다.",
            "tool": "correlation_scatter"
        }

    # col_y 기본값 = reference_period
    ref_col = str(reference_period).strip()
    if col_y is None:
        col_y = ref_col

    # col_x 기본값 = 10년 전 시점 또는 다른 수치 열
    if col_x is None:
        year_cols = find_year_columns(df)
        if ref_col in year_cols:
            idx = year_cols.index(ref_col)
            col_x = year_cols[idx - 10] if idx >= 10 else (year_cols[0] if len(year_cols) > 1 else ref_col)
        else:
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != col_y]
            col_x = num_cols[0] if num_cols else col_y

    if col_x not in df.columns or col_y not in df.columns:
        return {
            "error": f"상관관계를 분석할 두 열('{col_x}', '{col_y}')을 찾을 수 없습니다.",
            "tool": "correlation_scatter"
        }

    if label_column is None:
        label_column = find_default_label_column(df)

    filtered_df, excl_agg, _ = filter_aggregates_df(df, label_column, exclude=exclude_aggregates)

    sx = pd.to_numeric(filtered_df[col_x], errors="coerce")
    sy = pd.to_numeric(filtered_df[col_y], errors="coerce")
    valid_mask = sx.notna() & sy.notna()

    valid_df = filtered_df[valid_mask].copy()
    valid_df["__x__"] = sx[valid_mask]
    valid_df["__y__"] = sy[valid_mask]

    excl_null = len(filtered_df) - len(valid_df)
    total_excluded = excl_agg + excl_null

    if len(valid_df) < 3:
        return {
            "error": f"두 열('{col_x}', '{col_y}')에 유효한 공통 데이터가 3건 미만으로 상관계수를 산출할 수 없습니다.",
            "tool": "correlation_scatter"
        }

    # 피어슨 상관계수
    corr_coef = round(float(valid_df["__x__"].corr(valid_df["__y__"])), 3)

    # Chart.js 산점도 데이터 세트 (최대 100개 포인트 샘플링)
    scatter_points = [
        {"x": round(float(r["__x__"]), 2), "y": round(float(r["__y__"]), 2), "label": str(r[label_column])}
        for _, r in valid_df.head(100).iterrows()
    ]

    chart_data = {
        "type": "scatter",
        "datasets": [
            {
                "label": f"{col_x} vs {col_y} (r = {corr_coef})",
                "data": [{"x": p["x"], "y": p["y"]} for p in scatter_points],
                "backgroundColor": "#f59e0b"
            }
        ]
    }

    return {
        "tool": "correlation_scatter",
        "reference_period": f"{col_x} vs {col_y}",
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "exclusion_reason": f"집계 행 {excl_agg}개 및 두 열 중 결측치가 있는 {excl_null}개 행 제외",
        "used_columns": [label_column, col_x, col_y],
        "unit": "상관계수 (-1.0 ~ +1.0)",
        # [절대 규칙 준수]
        "caution": "상관관계는 인과관계가 아니다. 두 변수 간의 통계적 관련성이 원인과 결과를 의미하지는 않습니다.",
        "result": {
            "col_x": col_x,
            "col_y": col_y,
            "correlation_coefficient": corr_coef,
            "sample_size": len(valid_df),
            "points": sanitize_dict_or_list(scatter_points)
        },
        "chart_data": chart_data
    }


# -------------------------------------------------------------
# 도구 12: compare_one_vs_all (특정 항목 vs 전체 평균·중앙값 - 강조 막대)
# -------------------------------------------------------------
def compare_one_vs_all(df, reference_period, target_item=None, label_column=None, exclude_aggregates=True):
    """
    [필수 규칙: 기준시점(reference_period) 필수 파라미터]
    특정 1개 항목(예: 한국)의 수치를 전체 집단의 평균 및 중앙값과 직관적으로 비교합니다.
    """
    if not reference_period:
        return {
            "error": "도구 12(compare_one_vs_all)는 기준 시점(reference_period)이 필수입니다.",
            "tool": "compare_one_vs_all"
        }

    ref_col = str(reference_period).strip()
    if ref_col not in df.columns:
        return {
            "error": f"데이터셋에서 기준 시점 열 '{ref_col}'을 찾을 수 없습니다.",
            "tool": "compare_one_vs_all"
        }

    if label_column is None:
        label_column = find_default_label_column(df)

    filtered_df, excl_agg, _ = filter_aggregates_df(df, label_column, exclude=exclude_aggregates)

    s = pd.to_numeric(filtered_df[ref_col], errors="coerce")
    valid_mask = s.notna()
    valid_df = filtered_df[valid_mask].copy()
    valid_df["__val__"] = s[valid_mask]

    excl_null = len(filtered_df) - len(valid_df)
    total_excluded = excl_agg + excl_null

    if valid_df.empty:
        return {
            "error": f"기준 시점 '{ref_col}'에 유효한 데이터가 없습니다.",
            "tool": "compare_one_vs_all"
        }

    # 타깃 항목 선정 (기본: 한국, 없으면 1위 국가)
    available_items = valid_df[label_column].astype(str).tolist()
    if not target_item or target_item not in available_items:
        priority = ["Korea, Rep.", "South Korea", "대한민국", "United States"]
        found = [p for p in priority if p in available_items]
        target_item = found[0] if found else available_items[0]

    target_row = valid_df[valid_df[label_column].astype(str) == target_item]
    target_val = round(float(target_row["__val__"].values[0]), 2)

    all_mean = round(float(valid_df["__val__"].mean()), 2)
    all_median = round(float(valid_df["__val__"].median()), 2)

    # 순위 및 백분위 계산
    rank = int((valid_df["__val__"] > target_val).sum() + 1)
    percentile = round(((len(valid_df) - rank + 1) / len(valid_df) * 100), 1)

    chart_data = {
        "type": "bar",
        "labels": [target_item, "전체 평균", "전체 중앙값"],
        "datasets": [
            {
                "label": f"{ref_col}년 수치 비교",
                "data": [target_val, all_mean, all_median],
                "backgroundColor": ["#ef4444", "#94a3b8", "#64748b"]
            }
        ]
    }

    return {
        "tool": "compare_one_vs_all",
        "reference_period": ref_col,
        "used_rows_count": len(valid_df),
        "excluded_rows_count": total_excluded,
        "exclusion_reason": f"집계 행 {excl_agg}개 및 결측치 {excl_null}개 제외",
        "used_columns": [label_column, ref_col],
        "unit": "수치 (지표 기준)",
        "caution": "단일 항목과 전체 평균 비교 시 표본 내 극단치가 평균을 끌어올릴 수 있으므로 중앙값과의 격차도 함께 확인해야 합니다.",
        "result": {
            "target_item": target_item,
            "target_value": target_val,
            "overall_mean": all_mean,
            "overall_median": all_median,
            "diff_from_mean": round(target_val - all_mean, 2),
            "diff_from_median": round(target_val - all_median, 2),
            "rank": rank,
            "total_count": len(valid_df),
            "percentile": percentile
        },
        "chart_data": chart_data
    }


