const state = {
  manifest: null,
  translationCache: {},
  showZh: true,
  selectedModels: [],
  validRecords: [],
  recordIndex: 0,
  manualEvalEnabled: false,
  showEvalResult: true,
  evaluations: {},
  blindOrders: {},
  evalMessage: "",
  savingEvaluation: false,
  modal: {
    open: false,
    scale: 1,
    minScale: 0.2,
    maxScale: 5,
    step: 0.1,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    scrollStartLeft: 0,
    scrollStartTop: 0,
    naturalWidth: 0,
    naturalHeight: 0,
  },
};

const dom = {
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  sampleIdInput: document.getElementById("sampleIdInput"),
  jumpBtn: document.getElementById("jumpBtn"),
  jumpMsg: document.getElementById("jumpMsg"),
  addModelBtn: document.getElementById("addModelBtn"),
  promptText: document.getElementById("promptText"),
  promptZhText: document.getElementById("promptZhText"),
  promptZhBlock: document.getElementById("promptZhBlock"),
  promptZhToggle: document.getElementById("promptZhToggle"),
  validInfo: document.getElementById("validInfo"),
  indexInput: document.getElementById("indexInput"),
  indexTotal: document.getElementById("indexTotal"),
  modelSelectors: document.getElementById("modelSelectors"),
  grid: document.getElementById("grid"),
  emptyState: document.getElementById("emptyState"),
  manualEvalToggle: document.getElementById("manualEvalToggle"),
  showEvalResultToggle: document.getElementById("showEvalResultToggle"),
  manualEvalHint: document.getElementById("manualEvalHint"),
  manualEvalPanel: document.getElementById("manualEvalPanel"),
  manualEvalStatus: document.getElementById("manualEvalStatus"),
  manualEvalChoices: document.getElementById("manualEvalChoices"),
  manualEvalSummary: document.getElementById("manualEvalSummary"),
  evalWinnerBanner: document.getElementById("evalWinnerBanner"),
  imageModal: document.getElementById("imageModal"),
  imageModalBackdrop: document.getElementById("imageModalBackdrop"),
  imageModalTitle: document.getElementById("imageModalTitle"),
  imageModalViewport: document.getElementById("imageModalViewport"),
  imageModalImg: document.getElementById("imageModalImg"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  zoomResetBtn: document.getElementById("zoomResetBtn"),
  closeImageModalBtn: document.getElementById("closeImageModalBtn"),
};

async function loadManifest() {
  const resp = await fetch("./data/manifest.json");
  if (!resp.ok) {
    throw new Error(`读取 manifest 失败: ${resp.status}`);
  }
  return resp.json();
}

async function loadTranslationCache() {
  try {
    const resp = await fetch("./data/translation_cache.json");
    if (!resp.ok) {
      return {};
    }
    const data = await resp.json();
    if (data && typeof data === "object") {
      return data;
    }
    return {};
  } catch (_) {
    return {};
  }
}

async function loadEvaluations() {
  try {
    const resp = await fetch("/api/human-evaluations");
    if (!resp.ok) {
      return {};
    }
    const data = await resp.json();
    return data?.evaluations && typeof data.evaluations === "object"
      ? data.evaluations
      : {};
  } catch (_) {
    return {};
  }
}

function lookupZh(record) {
  if (!record) {
    return "";
  }
  if (record.prompt_zh) {
    return record.prompt_zh;
  }
  const en = record.prompt || "";
  if (!en) {
    return "";
  }
  return state.translationCache[en] || "";
}

function initSelection(models) {
  const preferred = ["Nano_Banana_Pro", "GPT_Image_1p5", "Seedream4p5"];
  const picked = preferred.filter((model) => models.includes(model));
  const fallback = models.filter((model) => !picked.includes(model));
  const defaultCount = Math.min(3, models.length);
  state.selectedModels = [...picked, ...fallback].slice(0, defaultCount);
}

function recomputeValidRecords() {
  const records = state.manifest.records;
  const selected = state.selectedModels;
  state.validRecords = records.filter((record) => {
    if (!record.original) {
      return false;
    }
    return selected.every((model) => Boolean(record.candidates[model]));
  });
  if (state.recordIndex >= state.validRecords.length) {
    state.recordIndex = 0;
  }
}

function makePairKey(models = state.selectedModels) {
  return [...models].sort().join("::");
}

function makeEvalKey(record, models = state.selectedModels) {
  if (!record || models.length !== 2) {
    return "";
  }
  return `${record.key}::${makePairKey(models)}`;
}

function currentEvaluation(record = state.validRecords[state.recordIndex]) {
  const key = makeEvalKey(record);
  return key ? state.evaluations[key] : null;
}

function blindOrderForRecord(record) {
  const key = makeEvalKey(record);
  if (!key) {
    return state.selectedModels;
  }
  if (!state.blindOrders[key]) {
    const models = [...state.selectedModels];
    if (Math.random() < 0.5) {
      models.reverse();
    }
    state.blindOrders[key] = models;
  }
  return state.blindOrders[key];
}

function displayModelsForRecord(record) {
  if (state.manualEvalEnabled && state.selectedModels.length === 2) {
    return blindOrderForRecord(record);
  }
  return state.selectedModels;
}

function isManualEvalAvailable() {
  return state.selectedModels.length === 2 && state.validRecords.length > 0;
}

function isShowEvalResultActive() {
  return state.showEvalResult && isManualEvalAvailable();
}

function ensureManualEvalState() {
  if (!isManualEvalAvailable()) {
    state.manualEvalEnabled = false;
  }
}

function selectedPairEvaluations() {
  if (state.selectedModels.length !== 2) {
    return [];
  }
  return state.validRecords
    .map((record) => state.evaluations[makeEvalKey(record)])
    .filter(Boolean);
}

function buildCompletionSummary() {
  const total = state.validRecords.length;
  if (!state.manualEvalEnabled || state.selectedModels.length !== 2 || !total) {
    return null;
  }
  const evaluations = selectedPairEvaluations();
  if (evaluations.length !== total) {
    return null;
  }

  const counts = {
    [state.selectedModels[0]]: 0,
    [state.selectedModels[1]]: 0,
    tie: 0,
  };
  evaluations.forEach((evaluation) => {
    counts[evaluation.winner] = (counts[evaluation.winner] || 0) + 1;
  });

  return {
    total,
    counts,
  };
}

function availableUnusedModels() {
  return state.manifest.models.filter((m) => !state.selectedModels.includes(m));
}

function buildModelSelectors() {
  dom.modelSelectors.innerHTML = "";
  state.selectedModels.forEach((selectedModel, index) => {
    const wrap = document.createElement("div");
    wrap.className = "selector-item";

    const select = document.createElement("select");
    state.manifest.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      option.selected = model === selectedModel;
      if (model !== selectedModel && state.selectedModels.includes(model)) {
        option.disabled = true;
      }
      select.appendChild(option);
    });
    select.addEventListener("change", (e) => {
      const oldModel = state.selectedModels[index];
      const nextModel = e.target.value;
      if (nextModel === oldModel) {
        return;
      }
      state.selectedModels[index] = nextModel;
      recomputeValidRecords();
      ensureManualEvalState();
      render();
    });

    const removeBtn = document.createElement("button");
    removeBtn.textContent = "删除";
    removeBtn.disabled = state.selectedModels.length <= 1;
    removeBtn.addEventListener("click", () => {
      if (state.selectedModels.length <= 1) {
        return;
      }
      state.selectedModels.splice(index, 1);
      recomputeValidRecords();
      ensureManualEvalState();
      render();
    });

    wrap.appendChild(select);
    wrap.appendChild(removeBtn);
    dom.modelSelectors.appendChild(wrap);
  });
}

function displayModelName(model, record = null) {
  const tool = record?.agent_final_tools?.[model];
  if (tool) {
    return `${model} (${tool})`;
  }
  return model;
}

function labelForWinner(winner, record = null) {
  if (winner === "tie") {
    return "平局";
  }
  if (!winner) {
    return "未评测";
  }
  return displayModelName(winner, record);
}

function renderManualEval() {
  const available = isManualEvalAvailable();
  const total = state.validRecords.length;
  const evaluatedCount =
    state.selectedModels.length === 2 ? selectedPairEvaluations().length : 0;

  dom.manualEvalToggle.disabled = !available;
  dom.manualEvalToggle.checked = state.manualEvalEnabled;
  dom.showEvalResultToggle.disabled = !available;
  dom.showEvalResultToggle.checked = state.showEvalResult;
  dom.manualEvalHint.textContent = available
    ? `当前模型对已评测 ${evaluatedCount}/${total}。可通过「显示评测结果」在图片区查看胜出模型。`
    : "仅选择两个模型且存在有效样本时可使用人工评测相关功能。";

  if (!state.manualEvalEnabled) {
    dom.manualEvalPanel.classList.add("hidden");
    return;
  }
  dom.manualEvalPanel.classList.remove("hidden");
  dom.manualEvalChoices.innerHTML = "";

  const record = state.validRecords[state.recordIndex];
  const displayModels = displayModelsForRecord(record);
  const evaluation = currentEvaluation(record);
  const currentText = evaluation
    ? `当前结果：${labelForWinner(evaluation.winner, record)}`
    : "当前图片尚未评测";
  dom.manualEvalStatus.textContent = `${currentText}。进度：${evaluatedCount}/${total}${
    state.evalMessage ? `。${state.evalMessage}` : ""
  }`;

  const choices = [
    { value: displayModels[0], label: "左侧更优" },
    { value: displayModels[1], label: "右侧更优" },
    { value: "tie", label: "平局" },
  ];
  choices.forEach((choice) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "eval-choice-btn";
    btn.textContent = choice.label;
    btn.disabled = state.savingEvaluation;
    if (evaluation?.winner === choice.value) {
      btn.classList.add("selected");
    }
    btn.addEventListener("click", () => {
      saveManualEvaluation(choice.value);
    });
    dom.manualEvalChoices.appendChild(btn);
  });

  const summary = buildCompletionSummary();
  if (!summary) {
    dom.manualEvalSummary.classList.add("hidden");
    dom.manualEvalSummary.textContent = "";
    return;
  }
  dom.manualEvalSummary.classList.remove("hidden");
  dom.manualEvalSummary.textContent = "";
  const title = document.createElement("strong");
  title.textContent = "全部图片已评测完成";
  dom.manualEvalSummary.appendChild(title);
  [
    `${state.selectedModels[0]} 胜出：${summary.counts[state.selectedModels[0]]}`,
    `${state.selectedModels[1]} 胜出：${summary.counts[state.selectedModels[1]]}`,
    `平局：${summary.counts.tie}`,
  ].forEach((text) => {
    const line = document.createElement("div");
    line.textContent = text;
    dom.manualEvalSummary.appendChild(line);
  });
}

function createCell(title, imagePath, noteText = "", options = {}) {
  const { isWinner = false, isTie = false, winnerTag = "" } = options;
  const cell = document.createElement("div");
  cell.className = "cell";
  if (isWinner) {
    cell.classList.add("cell-winner");
  }
  if (isTie) {
    cell.classList.add("cell-tie");
  }

  const titleDiv = document.createElement("div");
  titleDiv.className = "cell-title";
  const titleText = document.createElement("span");
  titleText.textContent = title;
  titleDiv.appendChild(titleText);
  if (winnerTag) {
    const tag = document.createElement("span");
    tag.className = "cell-winner-tag";
    tag.textContent = winnerTag;
    titleDiv.appendChild(tag);
  }
  cell.appendChild(titleDiv);

  const bodyDiv = document.createElement("div");
  bodyDiv.className = "cell-body";
  const imageWrap = document.createElement("div");
  imageWrap.className = "cell-image-wrap";
  const img = document.createElement("img");
  img.src = imagePath;
  img.loading = "lazy";
  img.alt = title;
  img.addEventListener("click", () => {
    openImageModal(imagePath, title);
  });
  const resolutionBadge = document.createElement("span");
  resolutionBadge.className = "cell-resolution";
  resolutionBadge.textContent = "—";
  const updateResolutionBadge = () => {
    if (img.naturalWidth && img.naturalHeight) {
      resolutionBadge.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
    }
  };
  img.addEventListener("load", updateResolutionBadge);
  if (img.complete) {
    updateResolutionBadge();
  }
  imageWrap.appendChild(img);
  imageWrap.appendChild(resolutionBadge);
  bodyDiv.appendChild(imageWrap);

  if (noteText) {
    const note = document.createElement("div");
    note.className = "small-text";
    note.textContent = noteText;
    bodyDiv.appendChild(note);
  }

  cell.appendChild(bodyDiv);
  return cell;
}

function clampScale(scale) {
  return Math.min(state.modal.maxScale, Math.max(state.modal.minScale, scale));
}

function updateModalScale(nextScale) {
  const scale = clampScale(nextScale);
  const viewport = dom.imageModalViewport;
  const previousScrollWidth = viewport.scrollWidth || 1;
  const previousScrollHeight = viewport.scrollHeight || 1;
  const centerX = (viewport.scrollLeft + viewport.clientWidth / 2) / previousScrollWidth;
  const centerY = (viewport.scrollTop + viewport.clientHeight / 2) / previousScrollHeight;

  state.modal.scale = scale;
  if (state.modal.naturalWidth && state.modal.naturalHeight) {
    dom.imageModalImg.style.width = `${state.modal.naturalWidth * scale}px`;
    dom.imageModalImg.style.height = `${state.modal.naturalHeight * scale}px`;
  }
  dom.zoomResetBtn.textContent = `${Math.round(scale * 100)}%`;

  requestAnimationFrame(() => {
    viewport.scrollLeft = viewport.scrollWidth * centerX - viewport.clientWidth / 2;
    viewport.scrollTop = viewport.scrollHeight * centerY - viewport.clientHeight / 2;
  });
}

function openImageModal(imagePath, title = "") {
  state.modal.open = true;
  state.modal.dragging = false;
  state.modal.naturalWidth = 0;
  state.modal.naturalHeight = 0;
  dom.imageModalImg.style.width = "";
  dom.imageModalImg.style.height = "";
  dom.imageModalImg.src = imagePath;
  dom.imageModalTitle.textContent = title || "图片预览";
  dom.imageModal.classList.remove("hidden");
  dom.imageModal.setAttribute("aria-hidden", "false");
  dom.imageModalViewport.classList.remove("dragging");
  dom.imageModalImg.onload = () => {
    state.modal.naturalWidth = dom.imageModalImg.naturalWidth;
    state.modal.naturalHeight = dom.imageModalImg.naturalHeight;
    updateModalScale(1);
  };
}

function closeImageModal() {
  if (!state.modal.open) {
    return;
  }
  state.modal.open = false;
  state.modal.dragging = false;
  dom.imageModal.classList.add("hidden");
  dom.imageModal.setAttribute("aria-hidden", "true");
  dom.imageModalViewport.classList.remove("dragging");
  dom.imageModalImg.onload = null;
  dom.imageModalImg.style.width = "";
  dom.imageModalImg.style.height = "";
  dom.imageModalImg.src = "";
}

function startModalDrag(e) {
  if (!state.modal.open || e.button !== 0) {
    return;
  }
  state.modal.dragging = true;
  state.modal.dragStartX = e.clientX;
  state.modal.dragStartY = e.clientY;
  state.modal.scrollStartLeft = dom.imageModalViewport.scrollLeft;
  state.modal.scrollStartTop = dom.imageModalViewport.scrollTop;
  dom.imageModalViewport.classList.add("dragging");
}

function moveModalDrag(e) {
  if (!state.modal.dragging) {
    return;
  }
  e.preventDefault();
  const dx = e.clientX - state.modal.dragStartX;
  const dy = e.clientY - state.modal.dragStartY;
  dom.imageModalViewport.scrollLeft = state.modal.scrollStartLeft - dx;
  dom.imageModalViewport.scrollTop = state.modal.scrollStartTop - dy;
}

function stopModalDrag() {
  if (!state.modal.dragging) {
    return;
  }
  state.modal.dragging = false;
  dom.imageModalViewport.classList.remove("dragging");
}

function renderEvalWinnerBanner(record, evaluation) {
  if (!isShowEvalResultActive()) {
    dom.evalWinnerBanner.classList.add("hidden");
    dom.evalWinnerBanner.textContent = "";
    return;
  }

  const total = state.validRecords.length;
  const evaluatedCount = selectedPairEvaluations().length;
  dom.evalWinnerBanner.classList.remove("hidden", "pending", "winner", "tie");

  if (!evaluation?.winner) {
    dom.evalWinnerBanner.classList.add("pending");
    dom.evalWinnerBanner.innerHTML = `当前样本 <strong>${record.key}</strong> 尚未评测（进度 ${evaluatedCount}/${total}）`;
    return;
  }

  if (evaluation.winner === "tie") {
    dom.evalWinnerBanner.classList.add("tie");
    dom.evalWinnerBanner.innerHTML = `当前样本 <strong>${record.key}</strong> 评测结果：<strong>平局</strong>（进度 ${evaluatedCount}/${total}）`;
    return;
  }

  dom.evalWinnerBanner.classList.add("winner");
  const winnerLabel = displayModelName(evaluation.winner, record);
  dom.evalWinnerBanner.innerHTML = `当前样本 <strong>${record.key}</strong> 评测结果：<strong>${winnerLabel}</strong> 胜出（进度 ${evaluatedCount}/${total}）`;
}

function renderGrid() {
  dom.grid.innerHTML = "";
  if (!state.validRecords.length) {
    dom.evalWinnerBanner.classList.add("hidden");
    dom.emptyState.classList.remove("hidden");
    return;
  }
  dom.emptyState.classList.add("hidden");

  const record = state.validRecords[state.recordIndex];
  const displayModels = displayModelsForRecord(record);
  const blindMode = state.manualEvalEnabled && state.selectedModels.length === 2;
  const showResult = isShowEvalResultActive();
  const evaluation = showResult ? currentEvaluation(record) : null;
  const winner = evaluation?.winner ?? null;
  const showRealNames = !blindMode || (showResult && Boolean(winner));
  const columnCount = 1 + displayModels.length;
  dom.grid.style.gridTemplateColumns = `repeat(${columnCount}, minmax(200px, 1fr))`;

  renderEvalWinnerBanner(record, evaluation);

  dom.grid.appendChild(createCell("原图", record.original, record.key));
  displayModels.forEach((model, index) => {
    const imagePath = record.candidates[model];
    const title = showRealNames
      ? displayModelName(model, record)
      : index === 0
        ? "左侧"
        : "右侧";
    const isWinner = showResult && Boolean(winner && winner !== "tie" && winner === model);
    const isTie = showResult && winner === "tie";
    let winnerTag = "";
    if (isWinner) {
      winnerTag = "胜出";
    } else if (isTie) {
      winnerTag = "平局";
    }
    dom.grid.appendChild(
      createCell(title, imagePath, record.key, {
        isWinner,
        isTie,
        winnerTag,
      }),
    );
  });
}

function updateZhBlockVisibility() {
  if (state.showZh) {
    dom.promptZhBlock.classList.remove("hidden");
  } else {
    dom.promptZhBlock.classList.add("hidden");
  }
}

function renderHeaderInfo() {
  const total = state.validRecords.length;
  if (!total) {
    dom.promptText.textContent = "当前模型组合下没有可对比样本。";
    dom.promptZhText.textContent = "";
    dom.promptZhText.classList.add("empty");
    updateZhBlockVisibility();
    dom.indexInput.value = "";
    dom.indexTotal.textContent = "-";
    dom.validInfo.textContent = `有效样本: 0`;
    return;
  }
  const record = state.validRecords[state.recordIndex];
  dom.promptText.textContent = record.prompt || "(无 prompt)";
  const zh = lookupZh(record);
  if (zh) {
    dom.promptZhText.textContent = zh;
    dom.promptZhText.classList.remove("empty");
  } else {
    dom.promptZhText.textContent = "（暂无中文翻译）";
    dom.promptZhText.classList.add("empty");
  }
  updateZhBlockVisibility();
  dom.indexInput.value = String(state.recordIndex + 1);
  dom.indexTotal.textContent = String(total);
  dom.validInfo.textContent = `有效样本: ${total}`;
}

function render() {
  buildModelSelectors();
  renderHeaderInfo();
  renderGrid();
  renderManualEval();
}

function setJumpMessage(message) {
  dom.jumpMsg.textContent = message;
}

function gotoPrev() {
  if (!state.validRecords.length) {
    return;
  }
  state.recordIndex =
    (state.recordIndex - 1 + state.validRecords.length) % state.validRecords.length;
  state.evalMessage = "";
  render();
}

function gotoNext() {
  if (!state.validRecords.length) {
    return;
  }
  state.recordIndex = (state.recordIndex + 1) % state.validRecords.length;
  state.evalMessage = "";
  render();
}

async function saveManualEvaluation(winner) {
  const record = state.validRecords[state.recordIndex];
  if (
    state.savingEvaluation ||
    !state.manualEvalEnabled ||
    !record ||
    state.selectedModels.length !== 2
  ) {
    return;
  }

  state.savingEvaluation = true;
  state.evalMessage = "保存中";
  renderManualEval();

  const payload = {
    sample_key: record.key,
    models: state.selectedModels,
    winner,
  };

  try {
    const resp = await fetch("/api/human-evaluations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    state.evaluations[data.evaluation.id] = data.evaluation;
    state.evalMessage = "已保存";
  } catch (err) {
    state.evalMessage = `保存失败，请通过 python web/server.py 启动页面（${err.message}）`;
  } finally {
    state.savingEvaluation = false;
    render();
  }
}

function jumpToSampleById(rawInput) {
  const target = (rawInput || "").trim();
  if (!target) {
    setJumpMessage("请输入样本ID。");
    return;
  }
  if (!state.validRecords.length) {
    setJumpMessage("当前没有可跳转样本。");
    return;
  }

  const exactIndex = state.validRecords.findIndex((r) => r.key === target);
  const fallbackIndex =
    exactIndex >= 0
      ? exactIndex
      : state.validRecords.findIndex(
          (r) => r.key.toLowerCase() === target.toLowerCase(),
        );

  if (fallbackIndex < 0) {
    setJumpMessage("未找到该样本ID（或当前模型组合下被过滤）。");
    return;
  }
  state.recordIndex = fallbackIndex;
  state.evalMessage = "";
  setJumpMessage(`已跳转到 ${state.validRecords[fallbackIndex].key}`);
  render();
}

function jumpToIndex(rawValue) {
  const text = String(rawValue ?? "").trim();
  const total = state.validRecords.length;
  if (!total) {
    setJumpMessage("当前没有可跳转样本。");
    return;
  }
  if (!text) {
    setJumpMessage("请输入样本序号。");
    dom.indexInput.value = String(state.recordIndex + 1);
    return;
  }
  const num = Number(text);
  if (!Number.isInteger(num)) {
    setJumpMessage("样本序号需为整数。");
    dom.indexInput.value = String(state.recordIndex + 1);
    return;
  }
  if (num < 1 || num > total) {
    setJumpMessage(`样本序号范围为 1 到 ${total}。`);
    dom.indexInput.value = String(state.recordIndex + 1);
    return;
  }
  state.recordIndex = num - 1;
  state.evalMessage = "";
  setJumpMessage(`已跳转到第 ${num} 个有效样本。`);
  render();
}

function bindEvents() {
  dom.prevBtn.addEventListener("click", gotoPrev);
  dom.nextBtn.addEventListener("click", gotoNext);
  dom.jumpBtn.addEventListener("click", () => {
    jumpToSampleById(dom.sampleIdInput.value);
  });
  dom.sampleIdInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      jumpToSampleById(dom.sampleIdInput.value);
    }
  });
  dom.indexInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      jumpToIndex(dom.indexInput.value);
    }
  });
  dom.indexInput.addEventListener("blur", () => {
    jumpToIndex(dom.indexInput.value);
  });
  dom.promptZhToggle.addEventListener("change", (e) => {
    state.showZh = Boolean(e.target.checked);
    updateZhBlockVisibility();
  });
  dom.showEvalResultToggle.addEventListener("change", (e) => {
    if (!isManualEvalAvailable()) {
      e.target.checked = false;
      state.showEvalResult = false;
      return;
    }
    state.showEvalResult = Boolean(e.target.checked);
    renderGrid();
  });
  dom.manualEvalToggle.addEventListener("change", (e) => {
    if (!isManualEvalAvailable()) {
      e.target.checked = false;
      state.manualEvalEnabled = false;
      return;
    }
    state.manualEvalEnabled = Boolean(e.target.checked);
    state.evalMessage = "";
    render();
  });
  dom.addModelBtn.addEventListener("click", () => {
    const candidates = availableUnusedModels();
    if (!candidates.length) {
      return;
    }
    state.selectedModels.push(candidates[0]);
    recomputeValidRecords();
    ensureManualEvalState();
    render();
  });
  dom.closeImageModalBtn.addEventListener("click", closeImageModal);
  dom.imageModalBackdrop.addEventListener("click", closeImageModal);
  dom.zoomInBtn.addEventListener("click", () => {
    updateModalScale(state.modal.scale + state.modal.step);
  });
  dom.zoomOutBtn.addEventListener("click", () => {
    updateModalScale(state.modal.scale - state.modal.step);
  });
  dom.zoomResetBtn.addEventListener("click", () => {
    updateModalScale(1);
  });
  dom.imageModalViewport.addEventListener(
    "wheel",
    (e) => {
      if (!state.modal.open) {
        return;
      }
      e.preventDefault();
      const direction = e.deltaY < 0 ? 1 : -1;
      updateModalScale(state.modal.scale + direction * state.modal.step);
    },
    { passive: false },
  );
  dom.imageModalViewport.addEventListener("mousedown", startModalDrag);
  window.addEventListener("mousemove", moveModalDrag);
  window.addEventListener("mouseup", stopModalDrag);
  dom.imageModalViewport.addEventListener("mouseleave", stopModalDrag);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.modal.open) {
      closeImageModal();
      return;
    }
    if (state.modal.open) {
      if (e.key === "+" || e.key === "=") {
        updateModalScale(state.modal.scale + state.modal.step);
      } else if (e.key === "-") {
        updateModalScale(state.modal.scale - state.modal.step);
      } else if (e.key === "0") {
        updateModalScale(1);
      }
      return;
    }
    if (e.target === dom.sampleIdInput || e.target === dom.indexInput) {
      return;
    }
    if (e.key === "ArrowLeft") {
      gotoPrev();
    } else if (e.key === "ArrowRight") {
      gotoNext();
    } else if (state.manualEvalEnabled && state.selectedModels.length === 2) {
      const displayModels = displayModelsForRecord(state.validRecords[state.recordIndex]);
      if (e.key === "1") {
        saveManualEvaluation(displayModels[0]);
      } else if (e.key === "2") {
        saveManualEvaluation(displayModels[1]);
      } else if (e.key === "3") {
        saveManualEvaluation("tie");
      }
    }
  });
}

async function bootstrap() {
  try {
    const [manifest, translationCache, evaluations] = await Promise.all([
      loadManifest(),
      loadTranslationCache(),
      loadEvaluations(),
    ]);
    state.manifest = manifest;
    state.translationCache = translationCache;
    state.evaluations = evaluations;
    state.showZh = dom.promptZhToggle.checked;
    state.showEvalResult = dom.showEvalResultToggle.checked;
    if (!state.manifest.models?.length || !state.manifest.records?.length) {
      dom.promptText.textContent = "manifest 数据为空，请先运行 prepare_data.py。";
      return;
    }
    initSelection(state.manifest.models);
    recomputeValidRecords();
    bindEvents();
    render();
  } catch (err) {
    dom.promptText.textContent = `加载失败：${err.message}`;
  }
}

bootstrap();
