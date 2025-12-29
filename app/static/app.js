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
    setStatus("Done. Adjust the threshold to label the result.", "success");
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

bindDropzone(elements.input1, elements.dropzone1, 1);
bindDropzone(elements.input2, elements.dropzone2, 2);
updateThresholdLabel();
