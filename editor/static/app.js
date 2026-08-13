const $ = (id) => document.getElementById(id);

const translations = {
  en: {
    brandHome: "MaskFlow Editor home",
    checkingRuntime: "Checking runtime",
    switchLanguage: "Switch to Chinese",
    maskTitle: "Paint the area you want to edit",
    uploadImage: "Upload image",
    maskTools: "Mask drawing tools",
    draw: "Draw",
    erase: "Erase",
    brush: "Brush",
    undo: "Undo",
    clear: "Clear",
    saveMask: "Save mask",
    startUpload: "Upload an image to begin",
    fileSupport: "PNG, JPG, and WebP supported. White mask pixels define the edit area.",
    chooseImage: "Choose image",
    sourceAlt: "Source image to edit",
    canvasAria: "Mask drawing canvas",
    canvasTip: "The magenta overlay is the edit area. The mask keeps the source image dimensions.",
    generateTitle: "Describe the result you want",
    promptPlaceholder: "For example: Replace the selected area with a red ceramic vase.",
    optional: "Optional",
    negativePlaceholder: "Content you do not want to appear",
    steps: "Inference steps",
    seed: "Seed",
    advanced: "Advanced settings",
    advancedCaption: "Model, LoRA & pipeline",
    modelLora: "Model & LoRA",
    baseModel: "Base model",
    device: "Device",
    precision: "Precision",
    sftPath: "SFT LoRA path / repository",
    sftWeight: "SFT weight file",
    dmdPath: "DMD LoRA path / repository",
    dmdWeight: "DMD weight file",
    pipeline: "Pipeline",
    maskDilation: "Mask dilation",
    maskBlurKernel: "Mask blur kernel",
    maskBlurSigma: "Mask blur sigma",
    maskEdgeWidth: "Mask edge width",
    poissonIterations: "Poisson iterations",
    loadModel: "Load model & LoRA only",
    startGenerate: "Generate",
    preparingTask: "Preparing task",
    loadingModel: "Loading model",
    loadingHint: "The first model load may take a while. Later runs reuse it.",
    generationHint: "Generation progress follows the denoising steps.",
    editComplete: "Edit complete",
    saveResult: "Save result",
    resultAlt: "MaskFlow edit result",
    modelReady: "Model loaded · {device}",
    gpuReady: "GPU available · {device}",
    noGpu: "No GPU detected · Mask editing available",
    statusUnavailable: "Service status unavailable",
    imageLoaded: "Loaded a {width} × {height} image",
    jobCreateFailed: "Could not create the task",
    imageComplete: "Image generation complete",
    loraLoaded: "Model and LoRA loaded",
    jobFailed: "Task failed",
    processing: "Processing",
    needImage: "Upload an image first",
    needPrompt: "Enter a prompt",
    errorNoGpu: "No CUDA GPU was detected. You can still create and save masks; run inference in a GPU environment.",
    stageQueued: "Waiting",
    stagePrepareModel: "Preparing model",
    stageLoadModel: "Loading base model and LoRA",
    stageReuseModel: "Reusing loaded model",
    stageModelReady: "Model ready",
    stageModelLoaded: "Model loaded",
    stagePrepareInput: "Preparing inputs",
    stageGenerating: "MaskFlow is generating",
    stageGeneratingStep: "MaskFlow is generating · {step}/{total}",
    stageSaving: "Saving result",
    stageComplete: "Generation complete",
    stageLoadFailed: "Model loading failed",
    stageFailed: "Generation failed",
  },
  zh: {
    brandHome: "MaskFlow Editor 首页",
    checkingRuntime: "正在检查运行环境",
    switchLanguage: "Switch to English",
    maskTitle: "圈出你想编辑的区域",
    uploadImage: "上传图像",
    maskTools: "Mask 绘制工具",
    draw: "绘制",
    erase: "擦除",
    brush: "笔刷",
    undo: "撤销",
    clear: "清空",
    saveMask: "保存 Mask",
    startUpload: "上传一张图像开始",
    fileSupport: "支持 PNG、JPG 和 WebP；白色区域将作为模型编辑范围。",
    chooseImage: "选择图像",
    sourceAlt: "待编辑原图",
    canvasAria: "Mask 绘制画布",
    canvasTip: "紫红色覆盖层为编辑区域，Mask 会保持与原图相同尺寸。",
    generateTitle: "描述你想要的结果",
    promptPlaceholder: "例如：将选中区域替换为一只红色陶瓷花瓶。",
    optional: "可选",
    negativePlaceholder: "不希望出现的内容",
    steps: "推理步数",
    seed: "随机种子",
    advanced: "高级设置",
    advancedCaption: "模型、LoRA 与 Pipeline",
    modelLora: "模型与 LoRA",
    baseModel: "基础模型",
    device: "设备",
    precision: "精度",
    sftPath: "SFT LoRA 路径 / 仓库",
    sftWeight: "SFT 权重文件",
    dmdPath: "DMD LoRA 路径 / 仓库",
    dmdWeight: "DMD 权重文件",
    pipeline: "Pipeline 参数",
    maskDilation: "Mask 膨胀核",
    maskBlurKernel: "Mask 模糊核",
    maskBlurSigma: "Mask 模糊 Sigma",
    maskEdgeWidth: "Mask 边缘宽度",
    poissonIterations: "Poisson 迭代次数",
    loadModel: "仅加载模型与 LoRA",
    startGenerate: "开始生成",
    preparingTask: "准备任务",
    loadingModel: "正在加载模型",
    loadingHint: "首次加载可能需要一些时间，后续生成会直接复用。",
    generationHint: "生成进度与模型去噪步数同步。",
    editComplete: "编辑完成",
    saveResult: "保存结果",
    resultAlt: "MaskFlow 编辑结果",
    modelReady: "模型已加载 · {device}",
    gpuReady: "GPU 可用 · {device}",
    noGpu: "未检测到 GPU · Mask 编辑可用",
    statusUnavailable: "服务状态不可用",
    imageLoaded: "已载入 {width} × {height} 图像",
    jobCreateFailed: "任务创建失败",
    imageComplete: "图像生成完成",
    loraLoaded: "模型与 LoRA 已加载",
    jobFailed: "任务失败",
    processing: "正在处理",
    needImage: "请先上传图像",
    needPrompt: "请填写 Prompt",
    errorNoGpu: "未检测到可用的 CUDA GPU。你仍可制作和保存 mask，推理请在 GPU 环境中启动。",
    stageQueued: "等待处理",
    stagePrepareModel: "准备模型",
    stageLoadModel: "加载基础模型与 LoRA",
    stageReuseModel: "复用已加载模型",
    stageModelReady: "模型已就绪",
    stageModelLoaded: "模型已加载",
    stagePrepareInput: "准备输入",
    stageGenerating: "MaskFlow 正在生成",
    stageGeneratingStep: "MaskFlow 正在生成 · {step}/{total}",
    stageSaving: "保存结果",
    stageComplete: "生成完成",
    stageLoadFailed: "加载失败",
    stageFailed: "生成失败",
  },
};

let currentLanguage = "en";
let lastStatus = null;
let lastProgressData = null;
let isBusy = false;

function t(key, values = {}) {
  let value = translations[currentLanguage][key] || translations.en[key] || key;
  for (const [name, replacement] of Object.entries(values)) {
    value = value.replace(`{${name}}`, replacement);
  }
  return value;
}

function applyLanguage() {
  document.documentElement.lang = currentLanguage === "en" ? "en" : "zh-CN";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    element.placeholder = t(element.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.i18nAria));
  });
  document.querySelectorAll("[data-i18n-alt]").forEach((element) => {
    element.alt = t(element.dataset.i18nAlt);
  });
  const toggle = $("languageToggle");
  toggle.textContent = currentLanguage === "en" ? "中文" : "EN";
  toggle.setAttribute("aria-label", t("switchLanguage"));
  if (lastStatus) renderStatus(lastStatus);
  if (lastProgressData) showProgress(lastProgressData);
  setBusy(isBusy);
}

$("languageToggle").addEventListener("click", () => {
  currentLanguage = currentLanguage === "en" ? "zh" : "en";
  applyLanguage();
});

const imageInput = $("imageInput");
const sourceImage = $("sourceImage");
const drawCanvas = $("drawCanvas");
const drawContext = drawCanvas.getContext("2d");
const maskCanvas = document.createElement("canvas");
const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
const canvasStage = $("canvasStage");
const emptyState = $("emptyState");
const brushCursor = $("brushCursor");
const history = [];

let sourceDataUrl = "";
let drawing = false;
let mode = "draw";
let previousPoint = null;
let pollTimer = null;

function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.add("visible");
  clearTimeout(element.hideTimer);
  element.hideTimer = setTimeout(() => element.classList.remove("visible"), 2800);
}

async function checkStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    lastStatus = data;
    renderStatus(data);
  } catch {
    lastStatus = { unavailable: true };
    renderStatus(lastStatus);
  }
}

function renderStatus(data) {
  const state = $("deviceState");
  if (data.unavailable) {
    state.className = "device-state warning";
    state.lastElementChild.textContent = t("statusUnavailable");
    return;
  }
  state.className = `device-state ${data.cuda_available ? "ready" : "warning"}`;
  if (data.model_loaded) {
    state.lastElementChild.textContent = t("modelReady", { device: data.gpu_name || "CPU" });
  } else if (data.cuda_available) {
    state.lastElementChild.textContent = t("gpuReady", { device: data.gpu_name });
  } else {
    state.lastElementChild.textContent = t("noGpu");
  }
}

function resetMask() {
  maskContext.globalCompositeOperation = "source-over";
  maskContext.fillStyle = "black";
  maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  drawContext.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  history.length = 0;
  updateButtons();
}

function updateButtons() {
  const ready = Boolean(sourceDataUrl);
  $("undoButton").disabled = !history.length;
  $("clearButton").disabled = !ready;
  $("saveMaskButton").disabled = !ready;
  $("generateButton").disabled = isBusy || !ready;
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    sourceDataUrl = reader.result;
    sourceImage.onload = () => {
      const { naturalWidth: width, naturalHeight: height } = sourceImage;
      drawCanvas.width = maskCanvas.width = width;
      drawCanvas.height = maskCanvas.height = height;
      resetMask();
      emptyState.hidden = true;
      canvasStage.hidden = false;
      imageInput.value = "";
      toast(t("imageLoaded", { width, height }));
    };
    sourceImage.src = sourceDataUrl;
  };
  reader.readAsDataURL(file);
});

function setMode(nextMode) {
  mode = nextMode;
  for (const [button, value] of [[$("drawTool"), "draw"], [$("eraseTool"), "erase"]]) {
    const active = value === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active);
  }
}

$("drawTool").addEventListener("click", () => setMode("draw"));
$("eraseTool").addEventListener("click", () => setMode("erase"));
$("brushSize").addEventListener("input", (event) => {
  $("brushValue").textContent = `${event.target.value} px`;
  updateCursorSize();
});

function pointFromEvent(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * drawCanvas.width / rect.width,
    y: (event.clientY - rect.top) * drawCanvas.height / rect.height,
  };
}

function brushWidth() {
  return Number($("brushSize").value) * drawCanvas.width / drawCanvas.getBoundingClientRect().width;
}

function drawLine(context, from, to, width, color, operation = "source-over") {
  context.save();
  context.globalCompositeOperation = operation;
  if (from.x === to.x && from.y === to.y) {
    context.fillStyle = color;
    context.beginPath();
    context.arc(from.x, from.y, width / 2, 0, Math.PI * 2);
    context.fill();
    context.restore();
    return;
  }
  context.strokeStyle = color;
  context.lineWidth = width;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
  context.restore();
}

function startDrawing(event) {
  if (!sourceDataUrl || event.button > 0) return;
  drawing = true;
  drawCanvas.setPointerCapture(event.pointerId);
  history.push(maskContext.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  if (history.length > 20) history.shift();
  previousPoint = pointFromEvent(event);
  paint(previousPoint);
  updateButtons();
}

function paint(nextPoint) {
  const width = brushWidth();
  if (mode === "draw") {
    drawLine(maskContext, previousPoint, nextPoint, width, "white");
    drawLine(drawContext, previousPoint, nextPoint, width, "rgb(214, 67, 138)");
  } else {
    drawLine(maskContext, previousPoint, nextPoint, width, "black");
    drawLine(drawContext, previousPoint, nextPoint, width, "black", "destination-out");
  }
  previousPoint = nextPoint;
}

function moveDrawing(event) {
  updateCursor(event);
  if (drawing) paint(pointFromEvent(event));
}

function stopDrawing(event) {
  if (!drawing) return;
  drawing = false;
  previousPoint = null;
  if (drawCanvas.hasPointerCapture(event.pointerId)) drawCanvas.releasePointerCapture(event.pointerId);
}

drawCanvas.addEventListener("pointerdown", startDrawing);
drawCanvas.addEventListener("pointermove", moveDrawing);
drawCanvas.addEventListener("pointerup", stopDrawing);
drawCanvas.addEventListener("pointercancel", stopDrawing);
drawCanvas.addEventListener("pointerenter", (event) => { brushCursor.style.opacity = "1"; updateCursor(event); });
drawCanvas.addEventListener("pointerleave", () => { if (!drawing) brushCursor.style.opacity = "0"; });

function updateCursor(event) {
  const rect = canvasStage.getBoundingClientRect();
  brushCursor.style.left = `${event.clientX - rect.left}px`;
  brushCursor.style.top = `${event.clientY - rect.top}px`;
}

function updateCursorSize() {
  const size = Number($("brushSize").value);
  brushCursor.style.width = `${size}px`;
  brushCursor.style.height = `${size}px`;
}

function redrawOverlay() {
  const mask = maskContext.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  const overlay = drawContext.createImageData(mask.width, mask.height);
  for (let index = 0; index < mask.data.length; index += 4) {
    if (mask.data[index] > 127) {
      overlay.data[index] = 214;
      overlay.data[index + 1] = 67;
      overlay.data[index + 2] = 138;
      overlay.data[index + 3] = 255;
    }
  }
  drawContext.putImageData(overlay, 0, 0);
}

$("undoButton").addEventListener("click", () => {
  const snapshot = history.pop();
  if (!snapshot) return;
  maskContext.putImageData(snapshot, 0, 0);
  redrawOverlay();
  updateButtons();
});

$("clearButton").addEventListener("click", () => {
  history.push(maskContext.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  if (history.length > 20) history.shift();
  maskContext.fillStyle = "black";
  maskContext.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  drawContext.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  updateButtons();
});

function downloadCanvas(canvas, filename) {
  canvas.toBlob((blob) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  }, "image/png");
}

$("saveMaskButton").addEventListener("click", () => downloadCanvas(maskCanvas, "maskflow-mask.png"));

function numberValue(id) { return Number($(id).value); }
function settings() {
  return {
    pretrained_model: $("pretrainedModel").value.trim(),
    device: $("device").value,
    dtype: $("dtype").value,
    sft_path: $("sftPath").value.trim(),
    sft_weight_name: $("sftWeight").value.trim(),
    dmd_path: $("dmdPath").value.trim(),
    dmd_weight_name: $("dmdWeight").value.trim(),
    mask_dilation_kernel: numberValue("maskDilation"),
    mask_blur_kernel: numberValue("maskBlur"),
    mask_blur_sigma: numberValue("maskBlurSigma"),
    mask_edge_width: numberValue("maskEdge"),
    enable_poisson_infer: $("poisson").checked,
    enable_local_denoise_infer: $("localDenoise").checked,
    enable_pixel_blend: $("pixelBlend").checked,
    poisson_num_iter: numberValue("poissonIter"),
  };
}

function runtime() {
  return {
    seed: numberValue("seed"),
    num_inference_steps: numberValue("steps"),
    text_cfg_scale: numberValue("textCfg"),
    mask_cfg_scale: numberValue("maskCfg"),
  };
}

async function createJob(endpoint, payload) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(translateError(data.error) || t("jobCreateFailed"));
  return data;
}

function translateStage(stage) {
  const stepMatch = stage.match(/^MaskFlow 正在生成 · (\d+)\/(\d+)$/);
  if (stepMatch) return t("stageGeneratingStep", { step: stepMatch[1], total: stepMatch[2] });
  const stageKeys = {
    "等待处理": "stageQueued",
    "准备模型": "stagePrepareModel",
    "加载基础模型与 LoRA": "stageLoadModel",
    "复用已加载模型": "stageReuseModel",
    "模型已就绪": "stageModelReady",
    "模型已加载": "stageModelLoaded",
    "准备输入": "stagePrepareInput",
    "MaskFlow 正在生成": "stageGenerating",
    "保存结果": "stageSaving",
    "生成完成": "stageComplete",
    "加载失败": "stageLoadFailed",
    "生成失败": "stageFailed",
  };
  return stageKeys[stage] ? t(stageKeys[stage]) : stage;
}

function translateError(error) {
  if (!error) return "";
  if (error.includes("未检测到可用的 CUDA GPU")) return t("errorNoGpu");
  return error;
}

function showProgress(data) {
  lastProgressData = data;
  if (data.kind === "model" && data.status === "done") {
    $("loadingCard").hidden = true;
    $("progressCard").hidden = true;
    return;
  }
  const isLoading = data.phase === "loading" || data.phase === undefined && data.progress < 40;
  $("loadingCard").hidden = !isLoading;
  $("progressCard").hidden = isLoading;
  if (isLoading) {
    $("loadingStage").textContent = data.status === "error"
      ? translateError(data.error) || t("jobFailed")
      : translateStage(data.stage);
    return;
  }
  $("progressStage").textContent = data.status === "error"
    ? translateError(data.error) || t("jobFailed")
    : translateStage(data.stage);
  $("progressValue").textContent = `${data.progress}%`;
  $("progressBar").style.width = `${data.progress}%`;
  $("progressBar").classList.toggle("running", data.phase === "generating");
}

function pollJob(job, isInference) {
  clearTimeout(pollTimer);
  showProgress(job);
  if (job.status === "done") {
    if (isInference) {
      const resultUrl = `${job.result_url}?t=${Date.now()}`;
      $("resultImage").src = resultUrl;
      $("saveResult").href = resultUrl;
      $("resultCard").hidden = false;
      $("resultCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
      toast(t("imageComplete"));
    } else {
      toast(t("loraLoaded"));
      checkStatus();
    }
    setBusy(false);
    return;
  }
  if (job.status === "error") {
    setBusy(false);
    const message = translateError(job.error) || t("jobFailed");
    toast(message);
    $("progressStage").textContent = message;
    return;
  }
  pollTimer = setTimeout(async () => {
    try {
      const response = await fetch(`/api/jobs/${job.id}`);
      pollJob(await response.json(), isInference);
    } catch (error) {
      setBusy(false);
      toast(error.message);
    }
  }, 700);
}

function setBusy(busy) {
  isBusy = busy;
  $("generateButton").disabled = busy || !sourceDataUrl;
  $("loadModelButton").disabled = busy;
  $("generateButtonText").textContent = busy ? t("processing") : t("startGenerate");
}

$("loadModelButton").addEventListener("click", async () => {
  try {
    setBusy(true);
    const job = await createJob("/api/model/load", { settings: settings() });
    pollJob(job, false);
  } catch (error) {
    setBusy(false);
    toast(error.message);
  }
});

$("generateForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!sourceDataUrl) return toast(t("needImage"));
  if (!$("prompt").value.trim()) return toast(t("needPrompt"));
  try {
    setBusy(true);
    $("resultCard").hidden = true;
    const job = await createJob("/api/jobs", {
      source: sourceDataUrl,
      mask: maskCanvas.toDataURL("image/png"),
      prompt: $("prompt").value,
      negative_prompt: $("negativePrompt").value,
      runtime: runtime(),
      settings: settings(),
    });
    pollJob(job, true);
  } catch (error) {
    setBusy(false);
    toast(error.message);
  }
});

updateCursorSize();
updateButtons();
applyLanguage();
checkStatus();
