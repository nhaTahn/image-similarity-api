const state = {
  file1: null,
  file2: null,
  lastScore: null,
};

const elements = {
  input1: document.getElementById("image1"),
  input2: document.getElementById("image2"),
  dropzone1: document.getElementById("dropzone1"),
  dropzone2: document.getElementById("dropzone2"),
  preview1: document.getElementById("preview1"),
  preview2: document.getElementById("preview2"),
  fileName1: document.getElementById("fileName1"),
  fileName2: document.getElementById("fileName2"),
  compareBtn: document.getElementById("compareBtn"),
  swapBtn: document.getElementById("swapBtn"),
  threshold: document.getElementById("threshold"),
  thresholdValue: document.getElementById("thresholdValue"),
  modeSelect: document.getElementById("modeSelect"),
  modelSelect: document.getElementById("modelSelect"),
  scoreValue: document.getElementById("scoreValue"),
  scoreLabel: document.getElementById("scoreLabel"),
  scoreBar: document.getElementById("scoreBar"),
  statusMsg: document.getElementById("statusMsg"),
  device: document.getElementById("device"),
  model: document.getElementById("model"),
};

function setStatus(message, tone = "") {
  elements.statusMsg.textContent = message;
  elements.statusMsg.classList.remove("status--error", "status--success");
  if (tone === "error") {
    elements.statusMsg.classList.add("status--error");
  }
  if (tone === "success") {
    elements.statusMsg.classList.add("status--success");
  }
}

function setInputFile(input, file) {
  const transfer = new DataTransfer();
  if (file) {
    transfer.items.add(file);
  }
  input.files = transfer.files;
}

function updatePreview(slot, file) {
  const preview = slot === 1 ? elements.preview1 : elements.preview2;
  const dropzone = slot === 1 ? elements.dropzone1 : elements.dropzone2;
  const fileName = slot === 1 ? elements.fileName1 : elements.fileName2;

  if (!file) {
    preview.src = "";
    dropzone.classList.remove("has-image");
    fileName.textContent = "No file selected";
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    dropzone.classList.add("has-image");
  };
  reader.readAsDataURL(file);
  fileName.textContent = file.name;
}

function bindDropzone(input, dropzone, slot) {
  input.addEventListener("change", () => {
    const file = input.files && input.files[0] ? input.files[0] : null;
    state[`file${slot}`] = file;
    updatePreview(slot, file);
  });

  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    const file = event.dataTransfer.files && event.dataTransfer.files[0] ? event.dataTransfer.files[0] : null;
    if (!file) {
      return;
    }
    setInputFile(input, file);
    state[`file${slot}`] = file;
    updatePreview(slot, file);
  });
}

function updateThresholdLabel() {
  elements.thresholdValue.textContent = Number(elements.threshold.value).toFixed(2);
  if (state.lastScore !== null) {
    updateScore(state.lastScore);
  }
}

function populateSelect(select, options, defaultName) {
  select.innerHTML = "";
  options.forEach((optionItem) => {
    const option = document.createElement("option");
    option.value = optionItem.name;
    option.textContent = optionItem.label || optionItem.name;
    if (optionItem.name === defaultName) {
      option.selected = true;
    }
    select.appendChild(option);
  });
}

function populateModelSelect(models, defaultName) {
  elements.modelSelect.innerHTML = "";
  populateSelect(elements.modelSelect, models, defaultName);
  elements.model.textContent = elements.modelSelect.value || defaultName;
}

function populateModeSelect(modes, defaultName) {
  populateSelect(elements.modeSelect, modes, defaultName);
}

async function loadModels() {
  const fallbackModels = [
    { name: "openai/clip-vit-base-patch32", label: "CLIP ViT-B/32 (fast)" },
    { name: "openai/clip-vit-large-patch14", label: "CLIP ViT-L/14 (higher quality)" },
  ];
  const fallbackModes = [
    { name: "semantic", label: "Semantic (CLIP)" },
    { name: "strict", label: "Strict (image hash)" },
    { name: "hybrid", label: "Hybrid (CLIP + hash)" },
  ];

  try {
    const response = await fetch("/models");
    if (!response.ok) {
      throw new Error("Failed to load models");
    }
    const data = await response.json();
    populateModelSelect(data.models || fallbackModels, data.default || fallbackModels[0].name);
    populateModeSelect(data.modes || fallbackModes, data.default_mode || fallbackModes[0].name);
  } catch (error) {
    populateModelSelect(fallbackModels, fallbackModels[0].name);
    populateModeSelect(fallbackModes, fallbackModes[0].name);
  }
}

function updateScore(score) {
  state.lastScore = score;
  const normalized = Math.max(-1, Math.min(1, score));
  const percent = ((normalized + 1) / 2) * 100;
  const threshold = Number(elements.threshold.value);
  elements.scoreValue.textContent = score.toFixed(4);
  elements.scoreBar.style.width = `${percent}%`;
  elements.scoreLabel.textContent = score >= threshold ? "Likely similar" : "Likely different";
}

async function compareImages() {
  if (!state.file1 || !state.file2) {
    setStatus("Please upload two images before comparing.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image1", state.file1, state.file1.name);
  formData.append("image2", state.file2, state.file2.name);
  formData.append("model_name", elements.modelSelect.value);
  formData.append("mode", elements.modeSelect.value);

  elements.compareBtn.disabled = true;
  elements.compareBtn.classList.add("loading");
  setStatus("Comparing images...");

  try {
    const response = await fetch("/compare", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const data = await response.json();
        detail = data.detail || detail;
      } catch (error) {
        const text = await response.text();
        detail = text || detail;
      }
      throw new Error(detail);
    }

    const data = await response.json();
    updateScore(data.similarity_score);
    elements.device.textContent = data.device;
    elements.model.textContent = data.model_name;
    const parts = [];
    if (data.semantic_similarity !== null && data.semantic_similarity !== undefined) {
      parts.push(`semantic: ${data.semantic_similarity.toFixed(4)}`);
    }
    if (data.hash_similarity !== null && data.hash_similarity !== undefined) {
      parts.push(`hash: ${data.hash_similarity.toFixed(4)}`);
    }
    const extra = parts.length ? ` (${parts.join(", ")})` : "";
    setStatus(`Done. Adjust the threshold to label the result.${extra}`, "success");
  } catch (error) {
    setStatus(`Error: ${error.message}`, "error");
  } finally {
    elements.compareBtn.disabled = false;
    elements.compareBtn.classList.remove("loading");
  }
}

elements.swapBtn.addEventListener("click", () => {
  const temp = state.file1;
  state.file1 = state.file2;
  state.file2 = temp;
  setInputFile(elements.input1, state.file1);
  setInputFile(elements.input2, state.file2);
  updatePreview(1, state.file1);
  updatePreview(2, state.file2);
});

elements.compareBtn.addEventListener("click", compareImages);
elements.threshold.addEventListener("input", updateThresholdLabel);
elements.modelSelect.addEventListener("change", () => {
  elements.model.textContent = elements.modelSelect.value;
});

bindDropzone(elements.input1, elements.dropzone1, 1);
bindDropzone(elements.input2, elements.dropzone2, 2);
updateThresholdLabel();
loadModels();
