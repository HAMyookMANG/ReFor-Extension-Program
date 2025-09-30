const script = document.createElement("script");
script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js";
script.onload = () => {
  console.log("onnxruntime-web 로드 완료 (UMD):", ort.version);
};
document.head.appendChild(script);
