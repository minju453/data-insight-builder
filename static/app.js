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

    // 2단계(제안) 섹션 열기
    document.getElementById("step2Section").classList.remove("hidden");
    document.getElementById("step2Section").scrollIntoView({ behavior: "smooth" });

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

  // 상세 계산 수치 표기
  const calcDisplay = document.getElementById("calculationResultDisplay");
  calcDisplay.innerHTML = `<pre><code>${JSON.stringify(runResult.result, null, 2)}</code></pre>`;
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
// DOM 준비 완료 시 실행
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  initServerStatus();
  setupUploadHandlers();
  setupSuggestHandler();
  setupRunHandler();
  setupExplainHandler();
  setupErrorCopyHandler();
});
