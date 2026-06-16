const dropzone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadForm = document.getElementById("upload-form");
const deletebtn = document.getElementById("delete-btn");
const submitbtn = document.getElementById("submit-btn");
const radiobox = document.getElementById("radiobox");
const buttong = document.getElementById("button-g");
const previewImg = document.getElementById("preview-img");
const previewContainer = document.getElementById("preview-container");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault(); // Empêche le rechargement de la page

  const formData = new FormData();
  const file = fileInput.files[0];
  const model = radiobox.querySelector('input[name="model-select"]:checked')?.value;

  if (!file) {
    alert("Veuillez choisir une image d'abord");
    return -1;
  }
  if (!model) {
    alert("Veuillez selectionner un modèle");
    return -1;
  }

  formData.append("file", file);
  formData.append("model", model);

  try {
    const response = await fetch("/class_predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.result) {
      alert("Résultat : " + data.result + "\nCertitude : " + (data.score*100).toFixed(2) + "%"); // Affiche "cat" + certitude
    } else {
      alert("Erreur : " + data.error);
    }
  } catch (error) {
    console.error("Erreur lors de l'envoi :", error);
  }
});

function handleFiles(files) {
  const previewImg = document.getElementById("preview-img");
  const previewContainer = document.getElementById("preview-container");
  const file = files[0];

  if (!file || !file.type.startsWith("image/")) {
    alert("Veuillez sélectionner une image");
    return;
  }

  // Mettre l'image dans le file input pour le formulaire
  fileInput.files = files; // Important pour Flask !

  const imageURL = URL.createObjectURL(file);
  previewImg.src = imageURL;
  previewContainer.style.display = "block";
  dropzone.style.display = "none";
  buttong.style.display = "block";
  radiobox.style.display = "block";
}

function resetDropZone(event) {
  if (event) event.preventDefault();
  if (previewImg.src && previewImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(previewImg.src);
  }
  previewImg.src = "";
  previewContainer.style.display = "none";
  dropzone.style.display = "flex";
  buttong.style.display = "none";
  radiobox.style.display = "none";
  fileInput.value = ""; 
  try {
    fileInput.files = new DataTransfer().files;
  } catch (e) {
    if (fileInput.value) fileInput.type = "text", fileInput.type = "file";
  }
}

["dragenter", "dragover", "dragleave", "drop"].forEach(evt => {
  dropzone.addEventListener(evt, e => e.preventDefault());
});

dropzone.addEventListener("click", () => fileInput.click());

["dragenter", "dragover"].forEach(evt => dropzone.addEventListener(evt, () => dropzone.classList.add("dragover")));
["dragleave", "drop"].forEach(evt => dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover")));

fileInput.addEventListener("change", () => handleFiles(fileInput.files));
dropzone.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));

const deleteBtn = document.getElementById("delete-btn");
if (deleteBtn) {
    deleteBtn.addEventListener("click", resetDropZone);
}