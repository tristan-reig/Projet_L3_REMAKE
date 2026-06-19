const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");
const imgInput = document.getElementById("imgInput");
const scores = document.getElementById("scores");
const resetBtn = document.getElementById("resetBtn");
const classList = document.getElementById("classList");

// Charge la liste des classes reconnues (colonne de gauche)
async function loadClasses() {
  try {
    const res = await fetch("/classifier/classes");
    if (!res.ok) return;
    const data = await res.json();
    classList.innerHTML = "";
    data.classes.forEach((name) => {
      const li = document.createElement("li");
      li.className = "flex items-center gap-2 text-muted";
      li.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-indigo-400/60 shrink-0"></span>
        <span class="truncate">${name}</span>`;
      classList.appendChild(li);
    });
  } catch (_) {
    /* silencieux : la liste reste vide si l'API ne répond pas */
  }
}
loadClasses();

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
  scores.innerHTML = "";
}

function renderScores(predictions) {
  scores.innerHTML = "";
  predictions.forEach((p, idx) => {
    const pct = Math.round(p.score * 100);
    const isTop = idx === 0;
    const row = document.createElement("div");

    const header = document.createElement("div");
    header.className = "flex items-baseline justify-between gap-3 mb-1";
    header.innerHTML = `
      <span class="text-sm truncate ${isTop ? "font-semibold" : "text-muted"}">${p.classe}</span>
      <span class="font-mono text-xs shrink-0 ${isTop ? "text-indigo-300" : "text-muted"}">${pct}%</span>`;

    const track = document.createElement("div");
    track.className = "h-2 rounded-full bg-white/10 overflow-hidden";
    const fill = document.createElement("div");
    fill.className = "h-full rounded-full " + (isTop
      ? "bg-gradient-to-r from-indigo-400 to-fuchsia-400"
      : "bg-white/25");
    fill.style.width = "0%";
    track.appendChild(fill);

    row.appendChild(header);
    row.appendChild(track);
    scores.appendChild(row);

    // Animation de remplissage (référence directe, plus robuste)
    requestAnimationFrame(() => {
      fill.style.transition = "width 0.6s ease";
      fill.style.width = pct + "%";
    });
  });
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
  imgInput.src = URL.createObjectURL(file);

  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/classifier/predict", { method: "POST", body: fd });
    if (!res.ok) {
      let detail = "Erreur " + res.status;
      try {
        const j = await res.json();
        if (j.detail) detail = j.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const data = await res.json();
    renderScores(data.predictions);
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
