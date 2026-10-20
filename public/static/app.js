/**
 * AI Data Insight Builder - Frontend Logic (Vanilla JS)
 * 외부 프레임워크 없이 순수 JavaScript 및 Chart.js로만 구현
 */

// 전역 상태 관리 객체 (오류 발생 시 AI에게 전달 가능)
const AppState = {
  serverHealthy: false,
  apiKeyConfigured: false,
  model: "",
  selectedFile: null,
  diagnosis: null,
  suggestions: [],
  selectedTool: null,
  lastRunResult: null,
  lastReport: null,
  lastError: null
};

let currentChartInstance = null;

// ============================================================
// 유틸리티: 에러 표시 및 콘솔 로깅
// ============================================================
function showError(message, details = null) {
  AppState.lastError = {
    timestamp: new Date().toISOString(),
    message: message,
    details: details,
    state: {
      tool: AppState.selectedTool,
      file: AppState.selectedFile ? AppState.selectedFile.name : null,
      refPeriod: document.getElementById("referencePeriodInput")?.value || null
    }
  };

  console.error(`[ERROR] ${message}`, details || "");

  const errorCard = document.getElementById("errorCard");
  const errorMsgText = document.getElementById("errorMessageText");
  if (errorCard && errorMsgText) {
    errorMsgText.textContent = message;
    errorCard.classList.remove("hidden");
    errorCard.scrollIntoView({ behavior: "smooth" });
  }
}

function clearError() {
  AppState.lastError = null;
  const errorCard = document.getElementById("errorCard");
  if (errorCard) errorCard.classList.add("hidden");
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// 간단하고 안전한 마크다운 -> HTML 변환기 (## 헤더, 볼드, 목록 처리)
function renderMarkdown(mdText) {
  if (!mdText) return "";
  let html = mdText
    .replace(/^## (.*$)/gim, "<h2>$1</h2>")
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/gim, "<em>$1</em>")
    .replace(/^\- (.*$)/gim, "<li>$1</li>")
    .replace(/^\d+\. (.*$)/gim, "<li>$1</li>")
    .replace(/\n\n+/g, "</p><p>")
    .replace(/\n/g, "<br>");
  return `<p>${html}</p>`;
}

// ============================================================
// 0. 초기화: 서버 상태 점검 (/health)
// ============================================================
async function initServerStatus() {
  try {
    const res = await fetch("/health");
    const data = await res.json();

    AppState.serverHealthy = data.status === "healthy";
    AppState.apiKeyConfigured = data.api_key_configured;
    AppState.model = data.model;

    const serverBadge = document.getElementById("serverStatusBadge");
    const apiKeyBadge = document.getElementById("apiKeyBadge");
    const modelBadge = document.getElementById("modelBadge");

    if (AppState.serverHealthy) {
      serverBadge.textContent = "서버 정상 연결";
      serverBadge.className = "badge badge-success";
    } else {
      serverBadge.textContent = "서버 상태 이상";
      serverBadge.className = "badge badge-warning";
    }

    if (AppState.apiKeyConfigured) {
      apiKeyBadge.textContent = "Gemini API Key 설정됨";
      apiKeyBadge.className = "badge badge-success";
    } else {
      apiKeyBadge.textContent = "⚠️ Gemini API Key 미설정";
      apiKeyBadge.className = "badge badge-warning";
    }

    modelBadge.textContent = `Model: ${AppState.model}`;

    console.log("[PAGE] 페이지 로드 완료 (서버 상태: " + data.status + ", 모델: " + data.model + ")");
  } catch (err) {
    console.error("[ERROR] 서버 상태 확인 실패:", err);
    showError("서버와의 통신에 실패했습니다. Flask 서버가 켜져 있는지 확인하세요.");
  }
}

// ============================================================
// 1단계: CSV 파일 선택 및 진단 (/inspect)
// ============================================================
function setupUploadHandlers() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("csvFileInput");
  const inspectBtn = document.getElementById("inspectBtn");
  const selectedFileInfo = document.getElementById("selectedFileInfo");
  const selectedFileName = document.getElementById("selectedFileName");
  const selectedFileSize = document.getElementById("selectedFileSize");

  function handleFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showError("CSV 형식(.csv)의 파일만 업로드할 수 있습니다.");
      return;
    }

    clearError();
    AppState.selectedFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);
    selectedFileInfo.classList.remove("hidden");
    inspectBtn.disabled = false;

    console.log(`[UPLOAD] 파일 선택 (파일명: ${file.name}, 크기: ${formatBytes(file.size)})`);
  }

  fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  inspectBtn.addEventListener("click", runInspect);
}

async function runInspect() {
  if (!AppState.selectedFile) return;

  clearError();
  const inspectBtn = document.getElementById("inspectBtn");
  inspectBtn.disabled = true;
  inspectBtn.textContent = "진단 분석 중...";

  console.log(`[INSPECT] 진단 요청 시작 (${AppState.selectedFile.name})`);

  const formData = new FormData();
  formData.append("file", AppState.selectedFile);

  try {
    const res = await fetch("/inspect", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "데이터 진단 중 오류가 발생했습니다.");
    }

    const diag = data.diagnosis;
    AppState.diagnosis = diag;

    const warnCount = diag.warnings ? diag.warnings.length : 0;
    console.log(`[INSPECT] 진단 응답 수신 (행 수: ${diag.row_count}, 열 수: ${diag.column_count}, 경고 수: ${warnCount})`);

    // 진단 결과 UI 렌더링
    renderInspectResults(diag);

    // [신규] 자연어 질문 섹션 및 2단계(제안) 섹션 열기
    document.getElementById("nlQuerySection").classList.remove("hidden");
    document.getElementById("step2Section").classList.remove("hidden");
    document.getElementById("nlQuerySection").scrollIntoView({ behavior: "smooth" });

  } catch (err) {
    showError(err.message);
  } finally {
    inspectBtn.disabled = false;
    inspectBtn.textContent = "🔍 데이터 진단 시작";
  }
}

function renderInspectResults(diag) {
  const resultArea = document.getElementById("inspectResultArea");
  const grid = document.getElementById("diagnoseGrid");
  const warningsBox = document.getElementById("diagnoseWarnings");
  const tableContainer = document.getElementById("previewTableContainer");

  resultArea.classList.remove("hidden");

  // 13대 지표 그리드 채우기
  const items = [
    { label: "파일명", val: diag.filename },
    { label: "파일 크기", val: formatBytes(diag.file_size_bytes) },
    { label: "인코딩", val: diag.encoding },
    { label: "건너뛴 헤더 줄 수", val: `${diag.skipped_header_rows}행` },
    { label: "데이터 크기", val: `${diag.row_count}행 × ${diag.column_count}열` },
    { label: "권장 기준연도", val: diag.recommended_reference_year || "없음" },
    { label: "중복 행 수", val: `${diag.duplicate_rows_count}개` },
    { label: "집계 행 후보", val: `${diag.aggregate_candidates ? diag.aggregate_candidates.length : 0}개` }
  ];

  grid.innerHTML = items.map(item => `
    <div class="diag-card">
      <span class="label">${item.label}</span>
      <div class="value">${item.val}</div>
    </div>
  `).join("");

  // 경고 목록
  if (diag.warnings && diag.warnings.length > 0) {
    warningsBox.innerHTML = `
      <h4>⚠️ 데이터 품질 주의 경고 (${diag.warnings.length}건)</h4>
      <ul>${diag.warnings.map(w => `<li>${w}</li>`).join("")}</ul>
    `;
    warningsBox.classList.remove("hidden");
  } else {
    warningsBox.classList.add("hidden");
  }

  // 미리보기 10행 테이블 생성
  if (diag.preview_rows && diag.preview_rows.length > 0) {
    const cols = Object.keys(diag.preview_rows[0]);
    let tableHtml = "<table><thead><tr>";
    cols.forEach(c => { tableHtml += `<th>${c}</th>`; });
    tableHtml += "</tr></thead><tbody>";

    diag.preview_rows.forEach(row => {
      tableHtml += "<tr>";
      cols.forEach(c => {
        const val = row[c] !== null && row[c] !== undefined ? row[c] : "-";
        tableHtml += `<td>${val}</td>`;
      });
      tableHtml += "</tr>";
    });
    tableHtml += "</tbody></table>";
    tableContainer.innerHTML = tableHtml;
  }
}

// ============================================================
// 2단계: Gemini 도구 제안 (/suggest)
// ============================================================
function setupSuggestHandler() {
  const suggestBtn = document.getElementById("suggestBtn");
  suggestBtn.addEventListener("click", runSuggest);
}

async function runSuggest() {
  clearError();
  const suggestBtn = document.getElementById("suggestBtn");
  suggestBtn.disabled = true;
  suggestBtn.textContent = "Gemini 추천 생성 중...";

  console.log("[SUGGEST] 제안 요청 시작");

  try {
    const res = await fetch("/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "분석 제안 생성 중 오류가 발생했습니다.");
    }

    AppState.suggestions = data.suggestions || [];
    console.log(`[SUGGEST] 제안 응답 수신 (제안 개수: ${AppState.suggestions.length}개)`);

    renderSuggestions(AppState.suggestions);

  } catch (err) {
    showError(err.message);
  } finally {
    suggestBtn.disabled = false;
    suggestBtn.textContent = "💡 AI 추천 분석 받기";
  }
}

function renderSuggestions(suggestions) {
  const area = document.getElementById("suggestResultArea");
  const list = document.getElementById("suggestionsList");

  area.classList.remove("hidden");

  if (!suggestions || suggestions.length === 0) {
    list.innerHTML = "<p>제안된 분석이 없습니다.</p>";
    return;
  }

  list.innerHTML = suggestions.map((s, idx) => `
    <div class="suggestion-card">
      <div class="sugg-content">
        <h4>
          <span>${idx + 1}. ${s.tool}</span>
          <span class="sugg-badge">${s.reference_period || "전체"}</span>
        </h4>
        <p class="sugg-why"><strong>추천 이유:</strong> ${s.why}</p>
        <div class="sugg-caution">⚠️ <strong>주의:</strong> ${s.caution}</div>
      </div>
      <button class="btn btn-primary btn-sm select-tool-btn" data-index="${idx}">이 분석 선택 👉</button>
    </div>
  `).join("");

  // '이 분석 선택' 버튼 이벤트 바인딩
  document.querySelectorAll(".select-tool-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const idx = parseInt(e.target.dataset.index, 10);
      selectSuggestion(suggestions[idx]);
    });
  });
}

function selectSuggestion(suggestion) {
  AppState.selectedTool = suggestion.tool;

  // 3단계 폼 자동 채우기
  document.getElementById("activeToolName").value = suggestion.tool;

  const refInput = document.getElementById("referencePeriodInput");
  const params = suggestion.params || {};

  refInput.value = suggestion.reference_period || params.reference_period || AppState.diagnosis?.recommended_reference_year || "";

  if (params.compare_period) {
    document.getElementById("comparePeriodInput").value = params.compare_period;
  }
  if (params.n) {
    document.getElementById("nParamInput").value = params.n;
  }

  // 필요 없는 파라미터 행 숨김/표시 처리
  const compareRow = document.getElementById("comparePeriodRow");
  const nRow = document.getElementById("nParamRow");

  compareRow.style.display = (suggestion.tool === "change_rate" || suggestion.tool === "correlation_scatter") ? "flex" : "none";
  nRow.style.display = (suggestion.tool === "top_bottom_n" || suggestion.tool === "change_rate") ? "flex" : "none";

  // 3단계 섹션 열기
  const step3 = document.getElementById("step3Section");
  step3.classList.remove("hidden");
  step3.scrollIntoView({ behavior: "smooth" });
}

// ============================================================
// 3단계: 도구 실행 (/run)
// ============================================================
function setupRunHandler() {
  const form = document.getElementById("runToolForm");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    const toolName = document.getElementById("activeToolName").value.trim();
    const refPeriod = document.getElementById("referencePeriodInput").value.trim();
    const compPeriod = document.getElementById("comparePeriodInput").value.trim();
    const nVal = parseInt(document.getElementById("nParamInput").value, 10) || 10;
    const exclAgg = document.getElementById("excludeAggregatesCheckbox").checked;

    if (!toolName) {
      showError("실행할 도구를 먼저 선택해 주세요.");
      return;
    }

    const payload = {
      tool: toolName,
      params: {
        reference_period: refPeriod,
        compare_period: compPeriod || null,
        n: nVal,
        exclude_aggregates: exclAgg
      }
    };

    console.log(`[RUN] 도구 실행 시작 (도구명: ${toolName}, 기준시점: ${refPeriod})`);

    const runBtn = document.getElementById("runBtn");
    runBtn.disabled = true;
    runBtn.textContent = "계산 실행 중...";

    try {
      const res = await fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "도구 계산 중 오류가 발생했습니다.");
      }

      AppState.lastRunResult = data.run_result;
      console.log(`[RUN] 결과 수신 (사용 행: ${data.run_result.used_rows_count}, 제외 행: ${data.run_result.excluded_rows_count})`);

      // 4단계(결과) 렌더링
      renderRunResult(data.run_result);

      // 4단계 및 5단계 섹션 표시
      document.getElementById("step4Section").classList.remove("hidden");
      document.getElementById("step5Section").classList.remove("hidden");
      document.getElementById("step4Section").scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      showError(err.message);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "⚙️ Pandas 분석 실행";
    }
  });
}

// ============================================================
// 4단계: 계산 결과 & Chart.js 렌더링
// ============================================================
function renderRunResult(runResult) {
  // 신뢰성 배지 업데이트
  document.getElementById("auditRefPeriod").textContent = runResult.reference_period || "-";
  document.getElementById("auditUsedRows").textContent = `${runResult.used_rows_count}개 행`;
  document.getElementById("auditExcludedRows").textContent = `${runResult.excluded_rows_count}개 행`;
  document.getElementById("auditExclusionReason").textContent = runResult.exclusion_reason || "제외 사유 없음";

  // Chart.js 렌더링
  if (runResult.chart_data) {
    renderChart(runResult.chart_data);
  } else {
    // 차트 데이터가 없는 경우 캔버스 숨김 처리
    document.getElementById("chartWrapper").style.display = "none";
  }

  // 상세 계산 수치 표기 (친절한 비교 표 및 상세 요약 렌더링)
  const calcDisplay = document.getElementById("calculationResultDisplay");
  calcDisplay.innerHTML = renderCalculationSummary(runResult.result);
}

function renderCalculationSummary(result) {
  if (!result || typeof result !== "object") {
    return `<pre><code>${JSON.stringify(result, null, 2)}</code></pre>`;
  }

  // 1) 순위 비교 테이블 (ranking_comparison_list가 있는 경우)
  if (Array.isArray(result.ranking_comparison_list) && result.ranking_comparison_list.length > 0) {
    let html = "";

    // 비교 대상 또는 최하위 요약 카드
    if (result.comparison_target) {
      const ct = result.comparison_target;
      const isBottom = String(ct.rank).includes("최하위") || String(ct.name).includes("최하위");
      const cardBg = isBottom ? "#fff1f2" : "#eff6ff";
      const cardBorder = isBottom ? "#fecdd3" : "#bfdbfe";
      const titleColor = isBottom ? "#e11d48" : "#1d4ed8";
      const icon = isBottom ? "🔻" : "🎯";
      const diff1Str = ct.diff_from_1st > 0 ? `+${ct.diff_from_1st}` : `${ct.diff_from_1st}`;
      const diff10Str = ct.diff_from_10th > 0 ? `+${ct.diff_from_10th}` : `${ct.diff_from_10th}`;

      html += `
        <div style="background: ${cardBg}; border: 1px solid ${cardBorder}; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
          <h4 style="margin: 0 0 6px 0; color: ${titleColor}; font-size: 15px; font-weight: 700;">${icon} ${ct.name} 분석 요약</h4>
          <p style="margin: 0; font-size: 13.5px; color: #1f2937; line-height: 1.6;">
            전체 <strong>${ct.total_count}개국</strong> 중 <strong>${ct.rank}</strong>에 위치하며,
            수치는 <strong style="color: ${isBottom ? '#e11d48' : '#dc2626'}; font-size: 15px;">${ct.value}%</strong>입니다.<br>
            • ${isBottom ? '1위(100%)와의 격차' : '1위와의 격차'}: <strong>${diff1Str}%p</strong> | • ${isBottom ? '하위 10위 국가와의 격차' : '10위와의 격차'}: <strong>${diff10Str}%p</strong>
          </p>
        </div>
      `;
    }

    html += `
      <div style="overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 8px;">
        <table class="table-preview" style="width: 100%; border-collapse: collapse; font-size: 13.5px;">
          <thead>
            <tr style="background: #f8fafc; text-align: left; border-bottom: 2px solid #e2e8f0;">
              <th style="padding: 10px 14px; width: 80px;">순위</th>
              <th style="padding: 10px 14px;">국가 / 대상</th>
              <th style="padding: 10px 14px; text-align: right; width: 120px;">수치 (%)</th>
              <th style="padding: 10px 14px; text-align: center; width: 110px;">구분</th>
            </tr>
          </thead>
          <tbody>
    `;

    result.ranking_comparison_list.forEach(row => {
      const isTarget = row.is_target;
      const isBottomRow = String(row.rank).includes("하위");
      const rowStyle = isTarget
        ? "background: #fee2e2; font-weight: 700; border-left: 4px solid #ef4444;"
        : isBottomRow ? "background: #fff5f5; border-bottom: 1px solid #fecdd3;" : "border-bottom: 1px solid #f1f5f9;";
      
      let badge = `<span style="background: #e2e8f0; color: #475569; padding: 3px 8px; border-radius: 12px; font-size: 11px;">상위권</span>`;
      if (isTarget) {
        badge = `<span style="background: #ef4444; color: white; padding: 3px 8px; border-radius: 12px; font-size: 11px;">⭐ 비교 대상</span>`;
      } else if (isBottomRow) {
        badge = `<span style="background: #ffe4e6; color: #e11d48; padding: 3px 8px; border-radius: 12px; font-size: 11px;">🔻 최하위권</span>`;
      }

      const rankDisplay = String(row.rank).includes("위") ? row.rank : `${row.rank}위`;

      html += `
        <tr style="${rowStyle}">
          <td style="padding: 9px 14px;">${rankDisplay}</td>
          <td style="padding: 9px 14px;">${row.label}</td>
          <td style="padding: 9px 14px; text-align: right; font-weight: ${isTarget || isBottomRow ? '700' : 'normal'}; color: ${isTarget ? '#dc2626' : (isBottomRow ? '#e11d48' : 'inherit')};">${row.value}%</td>
          <td style="padding: 9px 14px; text-align: center;">${badge}</td>
        </tr>
      `;
    });

    html += `
          </tbody>
        </table>
      </div>
      <details style="margin-top: 14px; font-size: 12px; color: #64748b;">
        <summary style="cursor: pointer; padding: 4px 0;">🔍 개발자용 상세 원본 데이터(JSON) 보기</summary>
        <pre style="margin-top: 8px; background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0;"><code>${JSON.stringify(result, null, 2)}</code></pre>
      </details>
    `;

    return html;
  }

  // 2) 기본 top_n만 있는 경우
  if (Array.isArray(result.top_n) && result.top_n.length > 0) {
    let html = `
      <div style="overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 8px;">
        <table class="table-preview" style="width: 100%; border-collapse: collapse; font-size: 13.5px;">
          <thead>
            <tr style="background: #f8fafc; text-align: left; border-bottom: 2px solid #e2e8f0;">
              <th style="padding: 10px 14px; width: 70px;">순위</th>
              <th style="padding: 10px 14px;">국가 / 대상</th>
              <th style="padding: 10px 14px; text-align: right; width: 120px;">수치</th>
            </tr>
          </thead>
          <tbody>
    `;
    result.top_n.forEach(row => {
      html += `
        <tr style="border-bottom: 1px solid #f1f5f9;">
          <td style="padding: 9px 14px;">${row.rank}위</td>
          <td style="padding: 9px 14px;">${row.label}</td>
          <td style="padding: 9px 14px; text-align: right;">${row.value}</td>
        </tr>
      `;
    });
    html += `
          </tbody>
        </table>
      </div>
      <details style="margin-top: 14px; font-size: 12px; color: #64748b;">
        <summary style="cursor: pointer; padding: 4px 0;">🔍 상세 원본 데이터(JSON) 보기</summary>
        <pre style="margin-top: 8px; background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0;"><code>${JSON.stringify(result, null, 2)}</code></pre>
      </details>
    `;
    return html;
  }

  // 3) 특정 국가 vs 전체 평균/중앙값 (compare_one_vs_all)
  if (result.target_item && result.overall_mean !== undefined) {
    const diffM = result.diff_from_mean > 0 ? `+${result.diff_from_mean}` : `${result.diff_from_mean}`;
    const diffMed = result.diff_from_median > 0 ? `+${result.diff_from_median}` : `${result.diff_from_median}`;
    return `
      <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
        <h4 style="margin: 0 0 10px 0; color: #1d4ed8; font-size: 16px; font-weight: 700;">🎯 ${result.target_item} vs 전 세계 통계 비교</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 10px;">
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">${result.target_item} 수치</div>
            <div style="font-size: 18px; font-weight: 800; color: #dc2626;">${result.target_value}%</div>
          </div>
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">전 세계 순위</div>
            <div style="font-size: 18px; font-weight: 800; color: #1e293b;">${result.rank}위 <span style="font-size: 12px; font-weight: 500; color: #64748b;">/ ${result.total_count}개국</span></div>
          </div>
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">전 세계 평균</div>
            <div style="font-size: 18px; font-weight: 800; color: #475569;">${result.overall_mean}% <span style="font-size: 12px; color: #16a34a;">(${diffM}%p)</span></div>
          </div>
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">전 세계 중앙값</div>
            <div style="font-size: 18px; font-weight: 800; color: #475569;">${result.overall_median}% <span style="font-size: 12px; color: #16a34a;">(${diffMed}%p)</span></div>
          </div>
        </div>
      </div>
      <details style="margin-top: 12px; font-size: 12px; color: #64748b;">
        <summary style="cursor: pointer;">🔍 상세 원본 데이터(JSON) 보기</summary>
        <pre style="margin-top: 8px; background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0;"><code>${JSON.stringify(result, null, 2)}</code></pre>
      </details>
    `;
  }

  // 4) 기술통계 요약 (describe_numeric)
  if (result.mean !== undefined && result.median !== undefined && result.min !== undefined) {
    return `
      <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
        <h4 style="margin: 0 0 10px 0; color: #334155; font-size: 16px; font-weight: 700;">📊 ${result.column || '기준시점'} 전 세계 수치 기술통계 요약</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-top: 10px;">
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">표본 국가 수</div>
            <div style="font-size: 17px; font-weight: 800; color: #0f172a;">${result.count}개국</div>
          </div>
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">평균 (Mean)</div>
            <div style="font-size: 17px; font-weight: 800; color: #2563eb;">${result.mean}%</div>
          </div>
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">중앙값 (Median)</div>
            <div style="font-size: 17px; font-weight: 800; color: #16a34a;">${result.median}%</div>
          </div>
          <div style="background: white; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
            <div style="font-size: 11px; color: #64748b;">최솟값 ~ 최댓값</div>
            <div style="font-size: 15px; font-weight: 800; color: #475569;">${result.min}% ~ ${result.max}%</div>
          </div>
        </div>
      </div>
      <details style="margin-top: 12px; font-size: 12px; color: #64748b;">
        <summary style="cursor: pointer;">🔍 상세 원본 데이터(JSON) 보기</summary>
        <pre style="margin-top: 8px; background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0;"><code>${JSON.stringify(result, null, 2)}</code></pre>
      </details>
    `;
  }

  // 5) 시계열 추이 비교 요약 (trend_line)
  if (result.selected_items && Array.isArray(result.selected_items)) {
    return `
      <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;">
        <h4 style="margin: 0 0 8px 0; color: #15803d; font-size: 16px; font-weight: 700;">📈 시계열 비교 분석 대상</h4>
        <p style="margin: 0; font-size: 14px; color: #1f2937;">
          선택된 항목: <strong style="color: #16a34a;">${result.selected_items.join(', ')}</strong><br>
          분석 연도 범위: <strong>${result.year_range ? result.year_range[0] + '년 ~ ' + result.year_range[1] + '년' : '전체 시계열'}</strong>
        </p>
      </div>
      <details style="margin-top: 12px; font-size: 12px; color: #64748b;">
        <summary style="cursor: pointer;">🔍 상세 원본 데이터(JSON) 보기</summary>
        <pre style="margin-top: 8px; background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0;"><code>${JSON.stringify(result, null, 2)}</code></pre>
      </details>
    `;
  }

  // 6) 구간별 분포 (distribution)
  const histList = result.histogram || result.bins;
  if (Array.isArray(histList) && histList.length > 0) {
    let bHtml = `
      <div style="overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 14px;">
        <table class="table-preview" style="width: 100%; border-collapse: collapse; font-size: 13.5px;">
          <thead>
            <tr style="background: #f8fafc; text-align: left; border-bottom: 2px solid #e2e8f0;">
              <th style="padding: 10px 14px;">구간 범위 (%)</th>
              <th style="padding: 10px 14px; text-align: right;">속한 국가 수</th>
              <th style="padding: 10px 14px; text-align: right;">비율 (%)</th>
            </tr>
          </thead>
          <tbody>
    `;
    histList.forEach(b => {
      bHtml += `
        <tr style="border-bottom: 1px solid #f1f5f9;">
          <td style="padding: 9px 14px; font-weight: 600;">${b.range}%</td>
          <td style="padding: 9px 14px; text-align: right;">${b.count}개국</td>
          <td style="padding: 9px 14px; text-align: right;">${b.percentage}%</td>
        </tr>
      `;
    });
    bHtml += `
          </tbody>
        </table>
      </div>
      <details style="margin-top: 12px; font-size: 12px; color: #64748b;">
        <summary style="cursor: pointer;">🔍 상세 원본 데이터(JSON) 보기</summary>
        <pre style="margin-top: 8px; background: #f8fafc; padding: 12px; border-radius: 6px; border: 1px solid #e2e8f0;"><code>${JSON.stringify(result, null, 2)}</code></pre>
      </details>
    `;
    return bHtml;
  }

  // 7) 그 외 기본 JSON 표기
  return `<pre><code>${JSON.stringify(result, null, 2)}</code></pre>`;
}

function renderChart(chartConfig) {
  const canvas = document.getElementById("insightChart");
  const wrapper = document.getElementById("chartWrapper");
  wrapper.style.display = "block";

  // 기존 차트 인스턴스 파괴 (메모리 누수 및 잔상 방지)
  if (currentChartInstance) {
    currentChartInstance.destroy();
    currentChartInstance = null;
  }

  const ctx = canvas.getContext("2d");

  // Chart.js 인스턴스 생성
  currentChartInstance = new Chart(ctx, {
    type: chartConfig.type || "bar",
    data: {
      labels: chartConfig.labels || [],
      datasets: chartConfig.datasets || []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: chartConfig.indexAxis || "x",
      plugins: {
        legend: {
          display: true,
          position: "top"
        },
        tooltip: {
          enabled: true
        }
      }
    }
  });

  console.log(`[CHART] 차트 렌더 완료 (차트 종류: ${chartConfig.type})`);
}

// ============================================================
// 5단계: Gemini AI 심층 해석 (/explain)
// ============================================================
function setupExplainHandler() {
  const explainBtn = document.getElementById("explainBtn");
  const copyReportBtn = document.getElementById("copyReportBtn");

  explainBtn.addEventListener("click", async () => {
    clearError();

    const selectedMode = document.querySelector('input[name="explainMode"]:checked')?.value || "B";
    console.log(`[EXPLAIN] 해석 요청 시작 (모드: ${selectedMode})`);

    explainBtn.disabled = true;
    explainBtn.textContent = "AI 보고서 작성 중...";

    try {
      const res = await fetch("/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: selectedMode,
          run_result: AppState.lastRunResult
        })
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "해석 보고서 생성 중 오류가 발생했습니다.");
      }

      AppState.lastReport = data.report;
      console.log(`[EXPLAIN] 해석 응답 수신 완료 (모드: ${selectedMode})`);

      // 보고서 마크다운 렌더링
      const area = document.getElementById("explainResultArea");
      const content = document.getElementById("reportContent");
      content.innerHTML = renderMarkdown(data.report);
      area.classList.remove("hidden");
      area.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      showError(err.message);
    } finally {
      explainBtn.disabled = false;
      explainBtn.textContent = "📝 AI 해석 보고서 생성";
    }
  });

  copyReportBtn.addEventListener("click", () => {
    if (AppState.lastReport) {
      navigator.clipboard.writeText(AppState.lastReport).then(() => {
        alert("보고서 내용이 클립보드에 복사되었습니다!");
      });
    }
  });

  // 쉬운 비유 설명 버튼 핸들러
  const easyExplainBtn = document.getElementById("easyExplainBtn");
  const easySummaryBox = document.getElementById("easySummaryBox");
  const easySummaryContent = document.getElementById("easySummaryContent");
  const closeEasyBtn = document.getElementById("closeEasyBtn");

  if (easyExplainBtn) {
    easyExplainBtn.addEventListener("click", async () => {
      clearError();
      easyExplainBtn.disabled = true;
      easyExplainBtn.textContent = "🐣 쉬운 설명 생성 중...";

      console.log("[EXPLAIN] 쉬운 비유 설명 요청 시작");

      try {
        const res = await fetch("/easy-explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_result: AppState.lastRunResult
          })
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "쉬운 설명 생성 중 오류가 발생했습니다.");
        }

        easySummaryContent.innerHTML = renderMarkdown(data.easy_explanation);
        easySummaryBox.classList.remove("hidden");
        easySummaryBox.scrollIntoView({ behavior: "smooth" });
        console.log("[EXPLAIN] 쉬운 비유 설명 렌더링 완료");

      } catch (err) {
        showError(err.message);
      } finally {
        easyExplainBtn.disabled = false;
        easyExplainBtn.textContent = "🐣 쉬운 비유로 설명해줘";
      }
    });
  }

  if (closeEasyBtn) {
    closeEasyBtn.addEventListener("click", () => {
      easySummaryBox.classList.add("hidden");
    });
  }
}

// ============================================================
// 후속 질문 핸들러 (/ask)
// ============================================================
function setupFollowUpHandler() {
  const askBtn = document.getElementById("askFollowUpBtn");
  const questionInput = document.getElementById("followUpQuestionInput");
  const answerArea = document.getElementById("followUpAnswerArea");
  const answerText = document.getElementById("followUpAnswerText");

  async function handleAsk(customQuestion = null) {
    const question = (customQuestion || questionInput.value).trim();
    if (!question) {
      alert("질문 내용을 입력해 주세요.");
      return;
    }

    if (customQuestion) {
      questionInput.value = customQuestion;
    }

    clearError();
    askBtn.disabled = true;
    askBtn.textContent = "AI 답변 생성 중...";

    console.log(`[EXPLAIN] 후속 질문 요청 시작 (질문: ${question.substring(0, 30)}...)`);

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question,
          report: AppState.lastReport,
          run_result: AppState.lastRunResult
        })
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "답변 생성 중 오류가 발생했습니다.");
      }

      console.log("[EXPLAIN] 후속 질문 답변 수신 완료");

      // 답변 렌더링
      answerText.innerHTML = renderMarkdown(data.answer);
      answerArea.classList.remove("hidden");
      answerArea.scrollIntoView({ behavior: "smooth" });

    } catch (err) {
      showError(err.message);
    } finally {
      askBtn.disabled = false;
      askBtn.textContent = "질문하기 🚀";
    }
  }

  askBtn.addEventListener("click", () => handleAsk());
  questionInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      handleAsk();
    }
  });

  // 추천 빠른 질문 칩 버튼 클릭 연동
  document.querySelectorAll(".chip-btn").forEach(chip => {
    chip.addEventListener("click", (e) => {
      const q = e.target.dataset.q;
      if (q) {
        handleAsk(q);
      }
    });
  });
}

// ============================================================
// 자연어 데이터 질문 핸들러 (/query)
// ============================================================
function setupNaturalLanguageQueryHandler() {
  const queryInput = document.getElementById("nlQueryInput");
  const queryBtn = document.getElementById("nlQueryBtn");
  const answerArea = document.getElementById("nlAnswerArea");
  const answerText = document.getElementById("nlAnswerText");
  const visualizeBtn = document.getElementById("nlVisualizeBtn");

  let queryRunResult = null;
  let queryToolName = null;
  let queryParams = null;

  async function handleQuery(customText = null) {
    const text = (customText || queryInput.value).trim();
    if (!text) {
      alert("데이터에 질문할 내용을 입력해 주세요.");
      return;
    }

    if (customText) {
      queryInput.value = customText;
    }

    clearError();
    queryBtn.disabled = true;
    queryBtn.textContent = "AI 통계 분석 중...";

    console.log(`[EXPLAIN] 자연어 데이터 질문 시작: '${text}'`);

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "자연어 질문 분석 중 오류가 발생했습니다.");
      }

      console.log(`[EXPLAIN] 자연어 데이터 답변 수신 완료 (도구: ${data.tool})`);

      // 답변 렌더링
      answerText.innerHTML = renderMarkdown(data.answer);
      answerArea.classList.remove("hidden");
      answerArea.scrollIntoView({ behavior: "smooth" });

      // 차후 시각화에 사용할 계산 결과 저장
      queryRunResult = data.run_result;
      queryToolName = data.tool;
      queryParams = data.params;
      AppState.lastRunResult = data.run_result;
      AppState.selectedTool = data.tool;

    } catch (err) {
      showError(err.message);
    } finally {
      queryBtn.disabled = false;
      queryBtn.textContent = "데이터 질문하기 🚀";
    }
  }

  queryBtn.addEventListener("click", () => handleQuery());
  queryInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      handleQuery();
    }
  });

  // 예시 질문 칩 클릭 연동
  document.querySelectorAll(".nl-sample-chip").forEach(chip => {
    chip.addEventListener("click", (e) => {
      const q = e.target.dataset.q;
      if (q) {
        handleQuery(q);
      }
    });
  });

  // [시각화 버튼 클릭 시] -> 4단계 차트 렌더링 및 5단계 보고서 단계로 직행
  visualizeBtn.addEventListener("click", () => {
    if (!queryRunResult) return;

    // 3단계 폼 값 동기화
    document.getElementById("activeToolName").value = queryToolName || "";
    if (queryParams && queryParams.reference_period) {
      document.getElementById("referencePeriodInput").value = queryParams.reference_period;
    }
    if (queryParams && queryParams.n) {
      document.getElementById("nParamInput").value = queryParams.n;
    }

    // 4단계 차트 렌더링
    renderRunResult(queryRunResult);

    // 4단계, 5단계 섹션 표시 및 부드러운 스크롤 이동
    document.getElementById("step3Section").classList.remove("hidden");
    document.getElementById("step4Section").classList.remove("hidden");
    document.getElementById("step5Section").classList.remove("hidden");

    document.getElementById("step4Section").scrollIntoView({ behavior: "smooth" });
  });
}

// ============================================================
// 오류 복사 핸들러 (오류 발생 시 AI에게 전달)
// ============================================================
function setupErrorCopyHandler() {
  const copyBtn = document.getElementById("copyErrorBtn");
  copyBtn.addEventListener("click", () => {
    const errorPayload = {
      appState: AppState,
      userAgent: navigator.userAgent,
      url: window.location.href
    };
    navigator.clipboard.writeText(JSON.stringify(errorPayload, null, 2)).then(() => {
      alert("현재 상태와 오류 로그가 클립보드에 복사되었습니다. AI에게 그대로 붙여넣어 질문하세요!");
    });
  });
}

// ============================================================
// PWA Service Worker 등록
// ============================================================
function initServiceWorker() {
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/static/sw.js')
        .then(reg => console.log('[PWA] Service Worker registered with scope:', reg.scope))
        .catch(err => console.warn('[PWA] Service Worker registration failed:', err));
    });
  }
}

// ============================================================
// 접속 비밀번호 4가지 및 보안 게이트 관리
// ============================================================
const DEFAULT_PASSWORDS = ['insight2024', 'gemini35', 'analytics77', 'alpha99'];
let activePasswords = [...DEFAULT_PASSWORDS];

// 로컬 스토리지에 저장된 이전 생성 비밀번호 불러오기
try {
  const saved = localStorage.getItem('allowed_passwords');
  if (saved) {
    const parsed = JSON.parse(saved);
    if (Array.isArray(parsed) && parsed.length >= 4) {
      activePasswords = parsed;
    }
  }
} catch (e) {}

async function loadAccessPasswords() {
  try {
    const res = await fetch('/auth/passwords');
    const data = await res.json();
    if (res.ok && Array.isArray(data.passwords) && data.passwords.length >= 4) {
      activePasswords = data.passwords;
    }
  } catch (err) {
    console.warn('[AUTH] 비밀번호 목록 로드 기본값 유지');
  }
  renderPasswordChips(activePasswords);
}

function renderPasswordChips(passwords) {
  const containers = [
    document.getElementById('activePasswordChips'),
    document.getElementById('modalPasswordChips')
  ];

  containers.forEach(container => {
    if (!container) return;
    container.innerHTML = '';
    passwords.forEach((pwd, idx) => {
      const chip = document.createElement('div');
      chip.className = 'pw-chip';
      chip.style.cursor = 'pointer';
      chip.title = '클릭하면 자동 입력되고 즉시 접속됩니다';
      chip.innerHTML = `
        <span><strong>#${idx + 1}</strong>: <code style="background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-weight: 700; color: #1e293b;">${pwd}</code></span>
        <button type="button" class="copy-btn" data-pw="${pwd}" style="font-weight: 700; color: #2563eb; background: none; border: none; cursor: pointer;">접속하기 👉</button>
      `;

      // 칩 클릭 시 비밀번호 입력 후 즉시 접속 허용
      chip.addEventListener('click', () => {
        const input = document.getElementById('authPasswordInput');
        if (input) {
          input.value = pwd;
        }
        const authModal = document.getElementById('authGateModal');
        sessionStorage.setItem('data_insight_auth', 'true');
        if (authModal) {
          authModal.classList.add('hidden');
          authModal.style.setProperty('display', 'none', 'important');
        }
      });

      container.appendChild(chip);
    });
  });
}

function setupAuthGate() {
  const authModal = document.getElementById('authGateModal');
  const authForm = document.getElementById('authGateForm');
  const authInput = document.getElementById('authPasswordInput');
  const errorMsg = document.getElementById('authErrorMsg');
  const genBtn = document.getElementById('generateNewPwBtn');
  const modalGenBtn = document.getElementById('modalGeneratePwBtn');
  const lockBtn = document.getElementById('lockScreenNowBtn');
  const manageModal = document.getElementById('passwordManageModal');
  const openManageBtn = document.getElementById('managePwBtn');
  const closeManageBtn = document.getElementById('closePwManageBtn');

  function unlockUI() {
    sessionStorage.setItem('data_insight_auth', 'true');
    if (authModal) {
      authModal.classList.add('hidden');
      authModal.style.setProperty('display', 'none', 'important');
    }
  }

  function lockUI() {
    sessionStorage.removeItem('data_insight_auth');
    if (authModal) {
      authModal.classList.remove('hidden');
      authModal.style.removeProperty('display');
    }
  }

  // 인증 상태 확인 (sessionStorage)
  const isAuth = sessionStorage.getItem('data_insight_auth') === 'true';
  if (isAuth) {
    unlockUI();
  } else {
    lockUI();
  }

  // HTML에 이미 렌더링된 초기 칩 버튼들에도 클릭 이벤트 바인딩
  document.querySelectorAll('#activePasswordChips .copy-btn, #activePasswordChips .pw-chip').forEach(el => {
    el.addEventListener('click', (e) => {
      const pw = el.getAttribute('data-pw') || el.querySelector('.copy-btn')?.getAttribute('data-pw');
      if (pw && authInput) authInput.value = pw;
      unlockUI();
    });
  });

  // 폼 제출 검증 (클라이언트 1차 즉시 승인 + 서버 검증 지원으로 100% 안정성 보장)
  if (authForm) {
    authForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (errorMsg) errorMsg.classList.add('hidden');
      const pwd = authInput ? authInput.value.trim() : '';
      if (!pwd) return;

      // 1. 유효 비밀번호 목록 대조 (기본 4가지 + 현재 활성 비밀번호 + 로컬 저장 비밀번호)
      const allValidPasswords = new Set([
        ...DEFAULT_PASSWORDS,
        ...activePasswords
      ]);
      try {
        const saved = localStorage.getItem('allowed_passwords');
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed)) parsed.forEach(p => allValidPasswords.add(p));
        }
      } catch (e) {}

      // 대소문자 무시 대조
      let matched = false;
      for (const validPw of allValidPasswords) {
        if (validPw.toLowerCase() === pwd.toLowerCase()) {
          matched = true;
          break;
        }
      }

      if (matched) {
        // 즉시 접속 허용!
        unlockUI();
        return;
      }

      // 2. 서버 백엔드 검증 시도
      const submitBtn = document.getElementById('authSubmitBtn');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '인증 중...';
      }

      try {
        const res = await fetch('/auth/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: pwd })
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || '접속 비밀번호가 일치하지 않습니다.');
        }
        unlockUI();
      } catch (err) {
        if (errorMsg) {
          errorMsg.textContent = '비밀번호가 올바르지 않습니다. 아래 표시된 4가지 비밀번호 중 하나를 클릭해 주세요.';
          errorMsg.classList.remove('hidden');
        }
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = '접속하기 🚀';
        }
      }
    });
  }

  // 새로운 비밀번호 4가지 무작위 생성 핸들러 (즉시 생성 및 100% 작동 보장)
  function handleGeneratePasswords() {
    if (!confirm('새로운 4가지 접속 비밀번호를 자동 생성하시겠습니까?')) return;

    const prefixes = ['data', 'insight', 'gemini', 'stat', 'vision', 'alpha', 'smart', 'matrix'];
    const newPwds = [];
    while (newPwds.length < 4) {
      const p = `${prefixes[Math.floor(Math.random() * prefixes.length)]}${Math.floor(1000 + Math.random() * 9000)}`;
      if (!newPwds.includes(p)) newPwds.push(p);
    }

    activePasswords = newPwds;
    try {
      localStorage.setItem('allowed_passwords', JSON.stringify(newPwds));
    } catch (e) {}
    renderPasswordChips(activePasswords);

    // 서버에도 동기화 시도
    fetch('/auth/generate-passwords', { method: 'POST' }).catch(() => {});

    alert(`새로운 4가지 접속 비밀번호가 생성되었습니다!\n아래 번호 중 하나를 클릭하여 바로 접속하세요:\n\n${newPwds.map((p, i) => `#${i+1}: ${p}`).join('\n')}`);
  }

  if (genBtn) genBtn.addEventListener('click', handleGeneratePasswords);
  if (modalGenBtn) modalGenBtn.addEventListener('click', handleGeneratePasswords);

  // 화면 잠금 버튼
  if (lockBtn) {
    lockBtn.addEventListener('click', () => {
      sessionStorage.removeItem('data_insight_auth');
      if (manageModal) manageModal.classList.add('hidden');
      authModal.classList.remove('hidden');
      authInput.value = '';
    });
  }

  // 비밀번호 관리 모달 열기/닫기
  if (openManageBtn && manageModal) {
    openManageBtn.addEventListener('click', () => {
      manageModal.classList.remove('hidden');
    });
  }
  if (closeManageBtn && manageModal) {
    closeManageBtn.addEventListener('click', () => {
      manageModal.classList.add('hidden');
    });
  }

  // 초기 비밀번호 렌더링
  renderPasswordChips(activePasswords);
  loadAccessPasswords();
}

// ============================================================
// Safari 및 모바일 PWA 안내 모달
// ============================================================
function setupSafariPwaGuide() {
  const guideModal = document.getElementById('safariGuideModal');
  const openBtn = document.getElementById('safariPwaBtn');
  const closeBtn = document.getElementById('closeSafariGuideBtn');
  const confirmBtn = document.getElementById('confirmSafariGuideBtn');

  if (openBtn && guideModal) {
    openBtn.addEventListener('click', () => {
      guideModal.classList.remove('hidden');
    });
  }

  const closeModal = () => {
    if (guideModal) guideModal.classList.add('hidden');
  };

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  if (confirmBtn) confirmBtn.addEventListener('click', closeModal);
}

// ============================================================
// DOM 준비 완료 시 실행
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  initServiceWorker();
  setupAuthGate();
  setupSafariPwaGuide();
  initServerStatus();
  setupUploadHandlers();
  setupNaturalLanguageQueryHandler();
  setupSuggestHandler();
  setupRunHandler();
  setupExplainHandler();
  setupFollowUpHandler();
  setupErrorCopyHandler();
});
