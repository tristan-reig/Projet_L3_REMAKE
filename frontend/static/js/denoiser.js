const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");
const imgBefore = document.getElementById("imgBefore");
const imgAfter = document.getElementById("imgAfter");
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
  loading.classList.add("hidden");
}

function reset() {
  result.classList.add("hidden");
  errorBox.classList.add("hidden");
  dropzone.classList.remove("hidden");
  fileInput.value = "";
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Fichier image invalide.");
    return;
  }
  errorBox.classList.add("hidden");
  dropzone.classList.add("hidden");
  result.classList.add("hidden");
  loading.classList.remove("hidden");
  imgBefore.src = URL.createObjectURL(file);

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/denoiser/denoise", { method: "POST", body: fd });
    if (!res.ok) {
      let detail = "Erreur " + res.status;
      try {
        const j = await res.json();
        if (j.detail) detail = j.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const url = URL.createObjectURL(await res.blob());
    imgAfter.src = url;
    downloadBtn.href = url;
    loading.classList.add("hidden");
    result.classList.remove("hidden");
  } catch (e) {
    showError("Échec : " + e.message);
    dropzone.classList.remove("hidden");
  }
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

resetBtn.addEventListener("click", reset);
