import csv
import io
import json
import logging
import math
import os
import re
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
import tools

# .env 환경변수 로드
load_dotenv()

# 로깅 설정 (print() 대신 logging 사용, 포맷 표준화)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DataInsightBuilder")

app = Flask(__name__)

# 환경변수 로드 및 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() in ("true", "1", "yes")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# 메모리 데이터셋 보관 (서버 재시작 시 초기화됨, 디스크 영구 저장 안 함)
current_dataset = None


@app.before_request
def log_request_info():
    """모든 HTTP 요청 로깅 (API Key나 민감 정보는 로깅 제외)"""
    logger.info(f"[REQUEST] {request.method} {request.path}")


@app.after_request
def log_response_info(response):
    """모든 HTTP 응답 상태 코드 로깅"""
    logger.info(f"[RESPONSE] status={response.status_code}")
    return response


@app.route("/", methods=["GET"])
def index():
    """메인 페이지 반환"""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    서버 상태 및 API Key 설정 여부 확인 라우트
    * 절대 규칙: 실제 API Key 값은 반환하지 않고, 설정 여부(True/False)만 반환
    """
    is_api_key_configured = bool(
        GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"
    )

    return jsonify({
        "status": "healthy",
        "api_key_configured": is_api_key_configured,
        "model": GEMINI_MODEL,
        "max_upload_mb": MAX_UPLOAD_MB,
        "dataset_loaded": current_dataset is not None
    }), 200


def detect_encoding_and_decode(raw_bytes):
    """
    규칙: utf-8-sig -> utf-8 -> cp949 -> euc-kr 순차 시도
    성공한 인코딩과 디코딩된 텍스트 반환
    """
    candidate_encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for enc in candidate_encodings:
        try:
            decoded_text = raw_bytes.decode(enc)
            logger.info(f"[ENCODING] {enc} 디코딩 성공")
            return enc, decoded_text
        except UnicodeDecodeError:
            logger.info(f"[ENCODING] {enc} 디코딩 실패, 다음 인코딩 시도")
            continue
    return None, None


def detect_header_row(text, max_inspect_lines=20):
    """
    앞쪽 줄들을 검사하여 열 이름(헤더)으로 보이는 행을 탐지
    - 가장 유효한 열(빈 문자열이 아닌 컬럼)이 많고 헤더 특성을 띤 행을 탐색
    - 건너뛴 줄 수(skiprows) 반환
    """
    import io
    import csv

    reader = csv.reader(io.StringIO(text))
    rows = []
    for i, row in enumerate(reader):
        if i >= max_inspect_lines:
            break
        rows.append(row)

    if not rows:
        return 0

    best_idx = 0
    max_cols = 0
    for idx, row in enumerate(rows):
        non_empty = [c.strip() for c in row if c and c.strip()]
        # 메타데이터 행(1~2개 컬럼)보다 훨씬 열이 많은 행을 헤더로 식별
        if len(non_empty) > max_cols and len(non_empty) >= 2:
            max_cols = len(non_empty)
            best_idx = idx

    logger.info(f"[HEADER] 헤더 줄 감지 결과: 행 인덱스 {best_idx} (건너뛴 줄 수: {best_idx})")
    return best_idx


def sanitize_for_json(val):
    """NaN, None, Infinity 등을 JSON 표준 안전 포맷(None 또는 문자열)으로 변환"""
    import math
    if val is None:
        return None
    if isinstance(val, (float, int)):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def run_dataframe_diagnostics(df, filename, file_size_bytes, encoding, skiprows, initial_warnings):
    """
    진단(/inspect)에 명시된 13가지 필수 항목을 정밀 진단하는 함수
    """
    import math
    import re
    import numpy as np

    warnings = list(initial_warnings)
    total_rows = len(df)
    total_cols = len(df.columns)
    columns = [str(c) for c in df.columns]

    # 1. Unnamed 열 / 완전히 비어 있는 열
    unnamed_cols = [c for c in columns if "unnamed" in c.lower() or not c.strip()]
    empty_cols = [c for c in columns if df[c].isna().all()]
    if empty_cols:
        warnings.append(f"전체 값이 비어 있는 열이 {len(empty_cols)}개 감지되었습니다: {', '.join(empty_cols[:5])}")

    # 2. 열별 결측치 개수 및 비율
    missing_summary = {}
    for c in columns:
        missing_cnt = int(df[c].isna().sum())
        missing_pct = round((missing_cnt / total_rows * 100), 2) if total_rows > 0 else 0.0
        missing_summary[c] = {
            "missing_count": missing_cnt,
            "missing_percentage": missing_pct
        }
        if missing_pct >= 80.0:
            warnings.append(f"열 '{c}'의 결측치 비율이 {missing_pct}%로 매우 높습니다.")

    # 3. 중복 행 수
    dup_rows = int(df.duplicated().sum())
    if dup_rows > 0:
        warnings.append(f"완전히 일치하는 중복 행이 {dup_rows}개 있습니다.")

    # 4. 열별 데이터 타입 분류
    dtypes_map = {c: str(df[c].dtype) for c in columns}
    numeric_cols = []
    categorical_cols = []
    date_or_year_cols = []

    year_col_pattern = re.compile(r"^(19|20)\d{2}$")

    for c in columns:
        clean_c = c.strip()
        # 연도 컬럼명 확인 (예: 1960 ~ 2025 등 World Bank 광폭형 연도)
        if year_col_pattern.match(clean_c):
            date_or_year_cols.append(c)

        # 데이터 형태 파악 (숫자형 vs 문자형)
        # pandas dtypes 검사 또는 수치형 변환 가능 여부
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            # 문자형이지만 결측치를 제외한 값들이 숫자로 변환 가능한지 테스트
            non_null = df[c].dropna()
            if len(non_null) > 0 and len(date_or_year_cols) > 0 and c in date_or_year_cols:
                # 연도 열인데 object 타입으로 잘못 읽힌 경우 수치형 후보에 포함
                try:
                    pd.to_numeric(non_null.head(10))
                    numeric_cols.append(c)
                except Exception:
                    categorical_cols.append(c)
            else:
                categorical_cols.append(c)

    # 컬럼명에 year, date, 날짜, 연도가 포함된 경우도 날짜 후보군에 추가
    for c in columns:
        c_lower = c.lower()
        if any(k in c_lower for k in ["year", "date", "연도", "일자", "시점"]) and c not in date_or_year_cols:
            date_or_year_cols.append(c)

    # 5. 연도별 데이터 충실도 & 권장 기준연도 계산
    year_coverage = {}
    recommended_year = None

    # 연도 컬럼들이 존재하는 경우 (World Bank Wide format 등)
    pure_year_cols = [c for c in columns if year_col_pattern.match(c.strip())]
    if pure_year_cols:
        for y in pure_year_cols:
            non_na_cnt = int(df[y].notna().sum())
            year_coverage[y] = non_na_cnt

        if year_coverage:
            max_non_na = max(year_coverage.values()) if year_coverage else 0
            # 값이 충분히 채워져 있는(최대치의 80% 이상) 연도 중 가장 최근 연도 선정
            threshold = max_non_na * 0.8
            valid_recent_years = [
                y for y in sorted(pure_year_cols, reverse=True)
                if year_coverage[y] >= threshold and year_coverage[y] > 0
            ]
            if valid_recent_years:
                recommended_year = valid_recent_years[0]
            else:
                # 없을 경우 단순 최대 빈도 연도
                recommended_year = max(year_coverage, key=year_coverage.get)

    # 6. 집계 행 후보 탐지
    # World Bank 등에서 World, OECD, Income 그룹, Area 등이 국가명에 들어간 행
    agg_keywords = [
        "world", "oecd", "euro area", "european union", "income", "area",
        "total", "sub-saharan", "latin america", "caribbean", "middle east",
        "north america", "south asia", "east asia", "ida", "ibrd", "dividend"
    ]
    aggregate_candidates = []

    # 레이블 후보가 될 만한 문자형 열 탐색 (예: Country Name, 국가명, 항목명 등)
    label_col_candidates = [c for c in categorical_cols if any(k in c.lower() for k in ["name", "country", "명", "국가", "지역", "항목"])]
    target_label_col = label_col_candidates[0] if label_col_candidates else (categorical_cols[0] if categorical_cols else None)

    if target_label_col:
        for idx, val in df[target_label_col].dropna().items():
            val_str = str(val).strip()
            val_lower = val_str.lower()
            matched = [k for k in agg_keywords if k in val_lower]
            if matched:
                aggregate_candidates.append({
                    "row_index": int(idx),
                    "label": val_str,
                    "matched_keyword": matched[0]
                })

    if aggregate_candidates:
        warnings.append(f"집계/그룹 성격의 행(국가 개별치가 아닌 World, 지역 등)이 {len(aggregate_candidates)}건 감지되었습니다.")

    # 7. 데이터 미리보기 10행 (안전 직렬화)
    preview_df = df.head(10).copy()
    preview_rows = []
    for record in preview_df.to_dict(orient="records"):
        cleaned_record = {}
        for k, v in record.items():
            cleaned_record[k] = sanitize_for_json(v)
        preview_rows.append(cleaned_record)

    diagnosis = {
        "filename": filename,
        "file_size_bytes": file_size_bytes,
        "encoding": encoding,
        "skipped_header_rows": skiprows,
        "row_count": total_rows,
        "column_count": total_cols,
        "columns": columns,
        "dtypes": dtypes_map,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_or_year_columns": date_or_year_cols,
        "missing_summary": missing_summary,
        "duplicate_rows_count": dup_rows,
        "unnamed_columns": unnamed_cols,
        "empty_columns": empty_cols,
        "year_coverage": year_coverage,
        "recommended_reference_year": recommended_year,
        "target_label_column": target_label_col,
        "aggregate_candidates": aggregate_candidates,
        "preview_rows": preview_rows,
        "warnings": warnings,
        "memory_note": "데이터는 서버 메모리에만 임시 보관되며 서버를 재시작하면 사라집니다."
    }

    return diagnosis


@app.route("/inspect", methods=["POST"])
def inspect():
    """
    POST /inspect (완전체 13대 진단 항목)
    """
    global current_dataset
    warnings = []

    # 1. 파일 수신 여부 확인
    if "file" not in request.files:
        logger.warning("[ERROR] 파일이 전송되지 않았습니다.")
        return jsonify({"error": "업로드할 CSV 파일을 선택해 주세요."}), 400

    file = request.files["file"]
    filename = file.filename or ""

    if not filename:
        logger.warning("[ERROR] 파일명이 비어 있습니다.")
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    # 2. 파일 확장자 검사 (.csv 필수)
    if not filename.lower().endswith(".csv"):
        logger.warning(f"[ERROR] 허용되지 않은 확장자: {filename}")
        return jsonify({"error": "CSV 파일(.csv)만 업로드할 수 있습니다."}), 400

    # 3. 파일 내용 읽기 및 크기 확인
    raw_bytes = file.read()
    file_size_bytes = len(raw_bytes)
    logger.info(f"[FILE] name={filename} size={file_size_bytes} bytes")

    if file_size_bytes == 0:
        logger.warning("[ERROR] 파일 내용이 비어 있습니다.")
        return jsonify({"error": "업로드한 파일의 내용이 비어 있습니다."}), 400

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        logger.warning(f"[ERROR] 파일 크기 초과: {file_size_bytes} > {max_bytes}")
        return jsonify({"error": f"파일 크기가 최대 제한({MAX_UPLOAD_MB}MB)을 초과했습니다."}), 400

    # 4. 인코딩 순차 시도
    encoding, text = detect_encoding_and_decode(raw_bytes)
    if not encoding:
        logger.error("[ERROR] 지원되는 인코딩(utf-8-sig, utf-8, cp949, euc-kr) 디코딩 모두 실패")
        return jsonify({"error": "지원되는 인코딩(UTF-8, CP949, EUC-KR)으로 파일을 디코딩하지 못했습니다."}), 400

    # 5. 헤더 행 자동 감지
    skiprows = detect_header_row(text)
    if skiprows > 0:
        warnings.append(f"상단 메타데이터 {skiprows}개 행을 건너뛰고 헤더를 탐지했습니다.")

    # 6. Pandas 파싱
    import io
    import pandas as pd

    try:
        df = pd.read_csv(io.StringIO(text), skiprows=skiprows)
    except Exception as e:
        logger.error(f"[ERROR] CSV 파싱 실패: {str(e)}", exc_info=True)
        return jsonify({"error": f"CSV 파싱 중 오류가 발생했습니다: {str(e)}"}), 400

    # 빈 데이터셋 검사
    if df.empty:
        logger.warning("[ERROR] 파싱된 데이터프레임이 비어 있습니다.")
        return jsonify({"error": "헤더 이후에 유효한 데이터 행이 존재하지 않습니다."}), 400

    # 7. 13가지 정밀 진단 실행
    diagnosis = run_dataframe_diagnostics(
        df=df,
        filename=filename,
        file_size_bytes=file_size_bytes,
        encoding=encoding,
        skiprows=skiprows,
        initial_warnings=warnings
    )

    # 메모리에 1개 데이터셋 및 진단결과 보관
    current_dataset = {
        "filename": filename,
        "encoding": encoding,
        "skiprows": skiprows,
        "df": df,
        "file_size": file_size_bytes,
        "diagnosis": diagnosis
    }

    logger.info(f"[CSV] parsing 완료 rows={len(df)} columns={len(df.columns)}")
    logger.info(f"[DIAGNOSE] 진단 완료 warnings={len(diagnosis['warnings'])}")

    return jsonify({
        "status": "success",
        "diagnosis": diagnosis
    }), 200


# 도구 카탈로그 정의 (Gemini는 오직 이 카탈로그 안의 도구만 제안 가능)
TOOL_CATALOG = [
    {
        "name": "reshape_wide_to_long",
        "description": "연도별 열을 행으로 접어 세로형(Long) 시계열 데이터로 변환",
        "required_params": [],
        "optional_params": ["id_vars", "value_vars"]
    },
    {
        "name": "quality_report",
        "description": "결측치 비율, 중복 행, 이상 열 이름 등 종합 데이터 품질 보고서",
        "required_params": [],
        "optional_params": []
    },
    {
        "name": "year_coverage",
        "description": "연도별 유효 데이터 개수 및 권장 기준연도 진단 (막대 차트)",
        "required_params": [],
        "optional_params": []
    },
    {
        "name": "detect_aggregates",
        "description": "World, OECD 등 다국가/지역/소득 그룹 집계 행 식별",
        "required_params": [],
        "optional_params": ["label_column"]
    },
    {
        "name": "describe_numeric",
        "description": "기준시점의 개수, 평균, 중앙값, 최소, 최대, 표준편차 등 수치 요약",
        "required_params": ["reference_period"],
        "optional_params": ["exclude_aggregates"]
    },
    {
        "name": "trend_line",
        "description": "선택한 주요 국가/항목들의 연도별 시계열 변화 추이 (선 차트)",
        "required_params": [],
        "optional_params": ["items", "label_column"]
    },
    {
        "name": "top_bottom_n",
        "description": "기준연도 기준 상위 N개 및 하위 N개 국가 순위 비교 (수평 막대)",
        "required_params": ["reference_period"],
        "optional_params": ["n", "exclude_aggregates"]
    },
    {
        "name": "group_summary",
        "description": "그룹별 평균, 합계, 개수 비교 (막대)",
        "required_params": ["reference_period"],
        "optional_params": ["group_column", "agg_func", "exclude_aggregates"]
    },
    {
        "name": "change_rate",
        "description": "두 시점(예: 10년간) 사이의 변화량 및 변화율(%) 분석 (막대)",
        "required_params": ["reference_period"],
        "optional_params": ["compare_period", "n", "exclude_aggregates"]
    },
    {
        "name": "distribution",
        "description": "기준시점 데이터의 구간별 도수분포 및 히스토그램 (히스토그램)",
        "required_params": ["reference_period"],
        "optional_params": ["bins", "exclude_aggregates"]
    },
    {
        "name": "correlation_scatter",
        "description": "두 시점/지표 간 피어슨 상관계수 및 산점도 (주의: 상관관계는 인과관계가 아님)",
        "required_params": ["reference_period"],
        "optional_params": ["col_x", "col_y", "exclude_aggregates"]
    },
    {
        "name": "compare_one_vs_all",
        "description": "특정 국가(예: 한국) vs 전체 평균·중앙값 격차 및 순위 비교 (강조 막대)",
        "required_params": ["reference_period"],
        "optional_params": ["target_item", "exclude_aggregates"]
    }
]

VALID_TOOL_NAMES = {t["name"] for t in TOOL_CATALOG}


@app.route("/suggest", methods=["POST"])
def suggest():
    """
    POST /suggest
    - 진단 요약과 도구 카탈로그 목록을 Gemini에게 전달
    - CSV 전체는 절대 전송하지 않음
    - Gemini는 카탈로그 내 도구 중 실행 가능한 3~5개만 JSON으로 제안
    """
    global current_dataset
    if current_dataset is None or "diagnosis" not in current_dataset:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 /suggest 요청")
        return jsonify({"error": "먼저 CSV 파일을 업로드하고 진단(/inspect)을 완료해 주세요."}), 400

    # 최신 .env API Key 로드
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("[ERROR] Gemini API Key 미설정")
        return jsonify({
            "error": "Gemini API Key가 설정되지 않았습니다. .env 파일의 GEMINI_API_KEY에 API 키를 입력해 주세요."
        }), 400

    diagnosis = current_dataset["diagnosis"]
    rec_year = diagnosis.get("recommended_reference_year") or "2024"
    year_cols = diagnosis.get("date_or_year_columns", [])
    compare_candidate_year = year_cols[year_cols.index(rec_year) - 10] if rec_year in year_cols and year_cols.index(rec_year) >= 10 else None

    # 1. Gemini에게 보낼 요약 컨텍스트 생성 (CSV 원본 제외)
    summary_for_ai = {
        "filename": diagnosis.get("filename"),
        "total_rows": diagnosis.get("row_count"),
        "total_columns": diagnosis.get("column_count"),
        "columns_sample": diagnosis.get("columns", [])[:15],
        "year_columns_range": f"{year_cols[0]} ~ {year_cols[-1]}" if year_cols else "없음",
        "recommended_reference_year": rec_year,
        "suggested_compare_year": compare_candidate_year,
        "aggregate_candidates_count": len(diagnosis.get("aggregate_candidates", [])),
        "target_label_column": diagnosis.get("target_label_column")
    }

    prompt = f"""당신은 데이터 분석 컨설턴트입니다.
사용자가 업로드한 데이터의 진단 요약과 사용 가능한 도구 카탈로그를 확인하고, 이 데이터로 실제 실행 가능한 가장 의미 있는 분석을 3개에서 5개 골라 제안해 주세요.

[절대 규칙]
1. 반드시 아래 도구 카탈로그에 명시된 도구(tool)만 제안할 수 있습니다. 카탈로그에 없는 도구는 절대 제안하지 마세요.
2. 데이터에 없는 열은 제안할 수 없습니다. 기준 시점(reference_period)은 가급적 권장 기준연도('{rec_year}')를 사용하세요.
3. 계산은 Pandas가 수행하므로 당신은 계산을 할 필요가 없습니다.
4. 오직 순수한 JSON만 출력하세요. 마크다운 따옴표나 기타 텍스트를 포함하지 마세요.
5. 제안 개수는 반드시 3개 이상 5개 이하로 작성하세요.

[도구 카탈로그]
{json.dumps(TOOL_CATALOG, ensure_ascii=False, indent=2)}

[데이터 진단 요약]
{json.dumps(summary_for_ai, ensure_ascii=False, indent=2)}

[출력 JSON 규격]
{{
  "suggestions": [
    {{
      "tool": "top_bottom_n",
      "params": {{
        "reference_period": "{rec_year}",
        "n": 10,
        "exclude_aggregates": true
      }},
      "reference_period": "{rec_year}",
      "why": "이 분석이 이 데이터에서 의미 있는 이유",
      "caution": "해석할 때 주의할 점"
    }}
  ]
}}
"""

    logger.info(f"[AI] suggest 요청 시작 (model={model_name})")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        logger.info("[AI] suggest 응답 수신 완료")

        resp_text = response.text.strip()
        # 혹시 모를 마크다운 코드 블록 제거
        if resp_text.startswith("```json"):
            resp_text = resp_text[7:]
        if resp_text.startswith("```"):
            resp_text = resp_text[3:]
        if resp_text.endswith("```"):
            resp_text = resp_text[:-3]
        resp_text = resp_text.strip()

        parsed = json.loads(resp_text)
        raw_suggestions = parsed.get("suggestions", [])
        logger.info(f"[AI] JSON 파싱 결과: {len(raw_suggestions)}개 제안 수신")

    except json.JSONDecodeError as jde:
        logger.error(f"[ERROR] Gemini 응답 JSON 파싱 실패: {str(jde)}")
        return jsonify({"error": "AI의 응답을 JSON으로 파싱하지 못했습니다. 다시 시도해 주세요."}), 500
    except Exception as e:
        err_msg = str(e)
        logger.error(f"[ERROR] Gemini API 호출 실패: {err_msg}", exc_info=True)
        if "API_KEY_INVALID" in err_msg or "401" in err_msg or "unregistered" in err_msg.lower():
            return jsonify({"error": "Gemini API 인증에 실패했습니다. 올바른 API Key인지 확인해 주세요."}), 401
        elif "timeout" in err_msg.lower() or "deadline" in err_msg.lower():
            return jsonify({"error": "Gemini API 요청 시간이 초과되었습니다(Timeout). 잠시 후 다시 시도해 주세요."}), 504
        return jsonify({"error": f"Gemini API 호출 중 오류가 발생했습니다: {err_msg}"}), 500

    # 검증: 카탈로그에 존재하는 도구만 필터링
    valid_suggestions = []
    for s in raw_suggestions:
        tool_name = s.get("tool")
        if tool_name in VALID_TOOL_NAMES:
            # 기준시점 보장
            if not s.get("reference_period"):
                s["reference_period"] = rec_year
            if "params" not in s or not isinstance(s["params"], dict):
                s["params"] = {}
            if "reference_period" not in s["params"]:
                s["params"]["reference_period"] = s["reference_period"]
            valid_suggestions.append(s)

    # 제안 수가 부족할 경우 카탈로그의 핵심 분석 도구로 기본 보충 (3~5개 보장)
    if len(valid_suggestions) < 3:
        fallback_candidates = [
            {
                "tool": "top_bottom_n",
                "params": {"reference_period": rec_year, "n": 10, "exclude_aggregates": True},
                "reference_period": rec_year,
                "why": f"{rec_year}년 기준 국가별 최상위 및 최하위 격차를 비교 분석합니다.",
                "caution": "소득 수준이나 인프라 격차를 감안하여 해석해야 합니다."
            },
            {
                "tool": "trend_line",
                "params": {"label_column": summary_for_ai["target_label_column"]},
                "reference_period": f"{year_cols[0]}~{rec_year}",
                "why": "시간 흐름에 따른 주요 국가들의 시계열 성장 궤적을 비교합니다.",
                "caution": "연도별 결측치가 존재하는 국가는 해석에 주의가 필요합니다."
            },
            {
                "tool": "compare_one_vs_all",
                "params": {"reference_period": rec_year, "target_item": "Korea, Rep.", "exclude_aggregates": True},
                "reference_period": rec_year,
                "why": f"한국의 {rec_year}년 수치를 세계 전체 평균 및 중앙값과 직관적으로 비교합니다.",
                "caution": "단일 국가 수치와 비교 시 표본의 대표성을 확인하세요."
            }
        ]
        for fb in fallback_candidates:
            if not any(v["tool"] == fb["tool"] for v in valid_suggestions):
                valid_suggestions.append(fb)
            if len(valid_suggestions) >= 3:
                break

    valid_suggestions = valid_suggestions[:5]

    return jsonify({
        "status": "success",
        "model_used": model_name,
        "suggestions_count": len(valid_suggestions),
        "suggestions": valid_suggestions
    }), 200


# 12대 도구 실행 함수 매핑
TOOL_MAP = {
    "reshape_wide_to_long": tools.reshape_wide_to_long,
    "quality_report": tools.quality_report,
    "year_coverage": tools.year_coverage,
    "detect_aggregates": tools.detect_aggregates,
    "describe_numeric": tools.describe_numeric,
    "trend_line": tools.trend_line,
    "top_bottom_n": tools.top_bottom_n,
    "group_summary": tools.group_summary,
    "change_rate": tools.change_rate,
    "distribution": tools.distribution,
    "correlation_scatter": tools.correlation_scatter,
    "compare_one_vs_all": tools.compare_one_vs_all
}

# 기준시점이 필수로 요구되는 도구 목록 (7~12번 도구)
REFERENCE_PERIOD_REQUIRED_TOOLS = {
    "top_bottom_n", "group_summary", "change_rate",
    "distribution", "correlation_scatter", "compare_one_vs_all"
}


@app.route("/run", methods=["POST"])
def run_tool():
    """
    POST /run
    - 선택된 도구를 오직 Pandas(tools.py)로만 실행
    - [절대 규칙] Gemini에게 숫자 계산을 시키지 않고 Pandas가 전담
    - 기준시점 유효성, 제외 행 수 및 제외 사유 반환
    """
    global current_dataset
    if current_dataset is None or "df" not in current_dataset:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 /run 요청")
        return jsonify({"error": "먼저 CSV 파일을 업로드하고 진단(/inspect)을 완료해 주세요."}), 400

    req_data = request.get_json(silent=True) or {}
    tool_name = req_data.get("tool", "").strip()
    params = req_data.get("params") or {}

    # 1. 도구 존재 여부 검증
    if not tool_name or tool_name not in TOOL_MAP:
        logger.warning(f"[ERROR] 도구 카탈로그에 없는 도구 요청: {tool_name}")
        return jsonify({
            "error": f"도구 목록에 존재하지 않는 도구입니다: '{tool_name}'. 카탈로그에 등록된 도구만 실행할 수 있습니다."
        }), 400

    # 2. 7~12번 도구의 기준 시점 필수성 검증
    ref_period = params.get("reference_period")
    if tool_name in REFERENCE_PERIOD_REQUIRED_TOOLS:
        if not ref_period or not str(ref_period).strip():
            logger.warning(f"[ERROR] 도구 {tool_name}의 기준시점 미입력")
            return jsonify({
                "error": f"도구 '{tool_name}'은(는) 기준 시점(reference_period)이 필수입니다. 분석할 연도나 시점을 지정해 주세요."
            }), 400

    df = current_dataset["df"]
    fn = TOOL_MAP[tool_name]

    # 3. 도구 실행 파라미터 안전 필터링
    import inspect as py_inspect
    sig = py_inspect.signature(fn)
    valid_param_keys = set(sig.parameters.keys()) - {"df"}
    filtered_params = {k: v for k, v in params.items() if k in valid_param_keys}

    try:
        # Pandas 함수 실행 (순수 로컬 계산)
        result = fn(df, **filtered_params)
    except Exception as e:
        logger.error(f"[ERROR] 도구 '{tool_name}' 실행 중 예외 발생: {str(e)}", exc_info=True)
        return jsonify({"error": f"도구 실행 중 오류가 발생했습니다: {str(e)}"}), 500

    # 도구 내부 에러 반환 여부 점검
    if isinstance(result, dict) and result.get("error"):
        logger.warning(f"[ERROR] 도구 반환 오류: {result.get('error')}")
        return jsonify({"error": result.get("error")}), 400

    used_rows = result.get("used_rows_count", 0)
    excluded_rows = result.get("excluded_rows_count", 0)
    logger.info(f"[TOOL] {tool_name}, params={filtered_params}, 사용 행={used_rows}, 제외 행={excluded_rows}")

    # 차후 /explain에서 사용할 수 있도록 마지막 실행 결과 메모리 저장
    current_dataset["last_run_result"] = result

    return jsonify({
        "status": "success",
        "tool": tool_name,
        "run_result": result
    }), 200


@app.route("/explain", methods=["POST"])
def explain():
    """
    POST /explain
    - 모드 A (비교용, 의도적으로 허술하게): 진단 요약만 전달하고 보고서 요청 (Pandas 계산 수치 미제공)
    - 모드 B (정식): Pandas가 계산한 수치만 엄격히 전달하고 8대 규칙을 지켜 7대 헤더 보고서 작성
    """
    global current_dataset
    if current_dataset is None or "diagnosis" not in current_dataset:
        logger.warning("[ERROR] 데이터셋 미업로드 상태에서 /explain 요청")
        return jsonify({"error": "먼저 CSV 파일을 업로드하고 진단(/inspect)을 완료해 주세요."}), 400

    req_data = request.get_json(silent=True) or {}
    mode = str(req_data.get("mode", "B")).upper()
    if mode not in ("A", "B"):
        mode = "B"

    # 모드 B의 경우 Pandas 계산 결과 필요
    run_result = req_data.get("run_result") or current_dataset.get("last_run_result")
    if mode == "B" and not run_result:
        logger.warning("[ERROR] 모드 B 요청이나 실행된 계산 결과 없음")
        return jsonify({
            "error": "모드 B(정식 해석)를 실행하려면 먼저 도구를 실행(/run)하여 Pandas 계산 결과를 생성해야 합니다."
        }), 400

    # API Key 검증
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("[ERROR] Gemini API Key 미설정")
        return jsonify({
            "error": "Gemini API Key가 설정되지 않았습니다. .env 파일의 GEMINI_API_KEY에 API 키를 입력해 주세요."
        }), 400

    diagnosis = current_dataset["diagnosis"]

    if mode == "A":
        # [모드 A]: 계산 결과를 일절 주지 않고 진단 요약만 전달 (환각/추정 유도 비교용)
        prompt = f"""당신은 데이터 분석가입니다.
사용자가 올린 데이터셋의 진단 요약만 보고 이 데이터를 분석하여 종합 보고서를 자유롭게 작성해 주세요.

[데이터 진단 요약]
- 파일명: {diagnosis.get('filename')}
- 총 행 수: {diagnosis.get('row_count')}
- 총 열 수: {diagnosis.get('column_count')}
- 포함된 주요 열: {diagnosis.get('columns', [])[:10]}
- 감지된 연도: {diagnosis.get('date_or_year_columns', [])[:10]} ... 등

자유롭게 수치와 상황을 추정하여 분석 보고서를 작성해 주세요.
"""
    else:
        # [모드 B]: Pandas 계산 수치만 전달하고 8대 엄격 규칙 및 7대 지정 헤더 강제
        prompt = f"""당신은 데이터 분석 및 교육 전문가입니다.
아래에 Python Pandas가 직접 계산한 신뢰할 수 있는 계산 결과가 제공됩니다.
당신은 직접 계산하거나 추측하지 말고, 제공된 숫자에만 엄격히 근거하여 정확하고 알기 쉬운 해석 보고서를 작성하세요.

[8대 절대 준수 규칙]
1. 제공된 계산 결과에 있는 숫자만 사용하세요. 직접 계산하거나 추정하지 마세요.
2. 데이터에 없는 국가, 연도, 열을 절대 언급하지 마세요.
3. 기준 시점('{run_result.get('reference_period')}')과 제외된 행 수({run_result.get('excluded_rows_count')}개, 사유: {run_result.get('exclusion_reason')})를 반드시 본문에 명시하세요.
4. 상관관계를 인과관계로 표현하지 마세요 (상관관계는 인과관계가 아님).
5. 지표의 정의와 출처를 보고서 앞부분에 표기하세요.
6. 확인할 수 없는 것은 솔직하게 "이 데이터로는 확인할 수 없다"라고 쓰세요.
7. 중학생도 이해할 수 있게 친절하고 명확하게 쓰되 통계적 정확성을 희생하지 마세요.
8. 아래 출력 형식을 한 글자도 빠짐없이 정확한 마크다운 헤더(##)로 유지하세요.

[출력 형식]
## 무엇을 봤는가
## 데이터 출처와 정의
## 기준 시점과 제외된 항목
## 계산 결과
## 읽어낼 수 있는 것
## 이 데이터로는 알 수 없는 것
## 다음에 확인해볼 것

[Pandas 실제 계산 결과 컨텍스트]
- 파일명: {diagnosis.get('filename')}
- 실행 도구: {run_result.get('tool')}
- 기준 시점: {run_result.get('reference_period')}
- 사용한 행 수: {run_result.get('used_rows_count')}개
- 제외한 행 수: {run_result.get('excluded_rows_count')}개
- 제외 사유: {run_result.get('exclusion_reason')}
- 사용한 열: {run_result.get('used_columns')}
- 단위: {run_result.get('unit')}
- 통계 주의사항: {run_result.get('caution')}
- 상세 수치 테이블:
{json.dumps(run_result.get('result', {}), ensure_ascii=False, indent=2)}
"""

    logger.info(f"[AI] explain 요청 시작 (mode={mode}, model={model_name})")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3 if mode == "B" else 0.7
            )
        )
        logger.info(f"[AI] explain 응답 수신 완료 (mode={mode})")
        report_text = response.text.strip()

    except Exception as e:
        err_msg = str(e)
        logger.error(f"[ERROR] Gemini API 호출 실패 (/explain): {err_msg}", exc_info=True)
        if "API_KEY_INVALID" in err_msg or "401" in err_msg or "unregistered" in err_msg.lower():
            return jsonify({"error": "Gemini API 인증에 실패했습니다. 올바른 API Key인지 확인해 주세요."}), 401
        elif "timeout" in err_msg.lower() or "deadline" in err_msg.lower():
            return jsonify({"error": "Gemini API 요청 시간이 초과되었습니다(Timeout). 잠시 후 다시 시도해 주세요."}), 504
        return jsonify({"error": f"Gemini API 호출 중 오류가 발생했습니다: {err_msg}"}), 500

    return jsonify({
        "status": "success",
        "mode": mode,
        "model_used": model_name,
        "report": report_text
    }), 200


if __name__ == "__main__":
    logger.info("Starting AI Data Insight Builder server...")
    logger.info(f"Model configured: {GEMINI_MODEL}")
    logger.info(f"Max upload limit: {MAX_UPLOAD_MB}MB")
    app.run(host="127.0.0.1", port=5000, debug=FLASK_DEBUG)
