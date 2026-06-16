const dropzone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadForm = document.getElementById("upload-form");
const deletebtn = document.getElementById("delete-btn");
const submitbtn = document.getElementById("submit-btn");
const buttong = document.getElementById("button-g");
const previewImg = document.getElementById("preview-img");
const previewRes = document.getElementById("preview-img-res")
const previewContainer = document.getElementById("preview-container");
const loader = document.getElementById("loaderid");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault(); // Empêche le rechargement de la page

  const formData = new FormData();
  const file = fileInput.files[0];
  buttong.style.display = "none";
  loader.style.display = "block";

  if (!file) {
    alert("Veuillez choisir une image d'abord");
    return -1;
  }

  formData.append("file", file);

  try {
    const response = await fetch("/coloriser_predict", {
      method: "POST",
      body: formData,
    });

    if(response.ok){
        const blob = await response.blob();

        const imageURL = URL.createObjectURL(blob);
        previewRes.src = imageURL;
        previewRes.onload = () => URL.revokeObjectURL(imageURL);
        previewRes.style.display = "block";
    }
    else {
        const errorData = await response.json();
        alert("Erreur : " + errorData.error);
    }
  } catch (error) {
    console.error("Erreur lors de l'envoi :", error);
  }
  buttong.style.display = "block";
  loader.style.display = "none";
});

function handleFiles(files) {
  const file = files[0];

  if (!file || !file.type.startsWith("image/")) {
    alert("Veuillez sélectionner une image");
    return;
  }

  // Mettre l'image dans le file input pour le formulaire
  fileInput.files = files; // Important pour Flask !

  const imageURL = URL.createObjectURL(file);
  previewImg.src = imageURL;
  previewContainer.style.display = "flex";
  dropzone.style.display = "none";
  buttong.style.display = "block";
}

function resetDropZone(event) {
  if (event) event.preventDefault();
  if (previewImg.src && previewImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(previewImg.src);
  }
  previewImg.src = "";
  previewRes.src = "";
  previewContainer.style.display = "none";
  previewRes.style.display = "none";
  dropzone.style.display = "flex";
  buttong.style.display = "none";
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