(async () => {
  const ort = await import(chrome.runtime.getURL("lib/ort.min.mjs"));
  console.log("onnxruntime-web 버전:", ort.version);
  console.log("AI Image Detector Content Script 시작");

  // ============================
  // ONNX 모델 로드/실행 유틸
  // ============================
  const modelSessions = {};
  const MODEL_PATHS = {
    binary: "models/binary_ViT.onnx",
    binary_residual: "models/binary_ReFor.onnx"
  };

  // softmax 함수
  function softmax(arr) {
    const maxVal = Math.max(...arr);
    const expArr = arr.map(v => Math.exp(v - maxVal));
    const sum = expArr.reduce((a, b) => a + b, 0);
    return expArr.map(v => v / sum);
  }

  // 원본 이미지 URL 추출
  function getOriginalImageUrl(thumbnailUrl, imgElement) {
    // 1. srcset이 있으면 가장 큰 이미지 선택
    if (imgElement && imgElement.srcset) {
      const candidates = imgElement.srcset.split(",").map(s => s.trim().split(" "));
      return candidates[candidates.length - 1][0];
    }
    // 2. data-src나 data-iurl 속성 있는 경우
    if (imgElement && imgElement.dataset) {
      if (imgElement.dataset.iurl) return imgElement.dataset.iurl;
      if (imgElement.dataset.src) return imgElement.dataset.src;
    }
    // 3. gstatic 썸네일이면 그대로 사용 (원본 찾기 실패)
    if (thumbnailUrl.includes("gstatic.com")) {
      console.warn("썸네일 URL만 있음, 원본 링크를 찾을 수 없음 → 그대로 사용");
      return thumbnailUrl;
    }
    return thumbnailUrl;
  }

  // 모델 세션 로드
  async function getModelSession(modelKey) {
    if (!MODEL_PATHS[modelKey]) throw new Error(`알 수 없는 모델 키: ${modelKey}`);
    if (!modelSessions[modelKey]) {
      const modelUrl = chrome.runtime.getURL(MODEL_PATHS[modelKey]);
      console.log(`[ONNX] ${modelKey} 모델 로드 시작: ${modelUrl}`);
      modelSessions[modelKey] = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"]
      });
      console.log(`[ONNX] ${modelKey} 모델 로드 완료`);
    }
    return modelSessions[modelKey];
  }

  // Binary 실행
  async function runViTClassifier(tensorData, tensorShape) {
    const session = await getModelSession("binary");  // vit_ai_image_detector.onnx
    const input = new ort.Tensor("float32", tensorData, tensorShape);
    const results = await session.run({ [session.inputNames[0]]: input });
    return Array.from(results[session.outputNames[0]].data);
  }


  async function runBinaryResidualClassifier(tensorData, tensorShape) {
    const session = await getModelSession("binary_residual");
    const input = new ort.Tensor("float32", tensorData, tensorShape);
    const results = await session.run({ [session.inputNames[0]]: input });
    return Array.from(results[session.outputNames[0]].data);
  }


  // ============================
  // 메시지 리스너
  // ============================
  chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
    if (message.action === "detectAI") {
      showLoadingDialog();

      try {
        // 0) 원본 이미지 URL 찾기
        const imgElement = document.querySelector(`img[src="${message.imageUrl}"]`);
        const realUrl = getOriginalImageUrl(message.imageUrl, imgElement);

        // 1) background에서 fetch → dataURL 받기
        const fetched = await chrome.runtime.sendMessage({
          action: "fetchImage",
          url: realUrl
        });
        if (!fetched || !fetched.ok || !fetched.dataUrl) {
          throw new Error("이미지 가져오기 실패");
        }

        // 2) dataURL을 실제 <img>로 로드
        const img = await loadImage(fetched.dataUrl);

        // 3) ViT 전처리 → binary_full
        const vitData = preprocessImageViT(img);
        const bin1 = await runViTClassifier(vitData.data, [1,3, vitData.height, vitData.width]);
        const probs1 = softmax(bin1);
        const isFake1 = probs1[1] > probs1[0];

        // 4) Residual 전처리 → binary_full_residual
        const resData = preprocessImageResidual(img);
        const bin2 = await runBinaryResidualClassifier(resData.data, [1, 3, resData.height, resData.width]);
        const isFake2 = bin2[1] > bin2[0];

        // 5) 최종 판정
        const finalIsFake = isFake1 || isFake2;
        let finalResult = {
          isAI: finalIsFake,
          modelType: finalIsFake ? "AI 생성 (Fake)" : "실제 (Real)",
          // confidence: (Math.max(
          //   finalIsFake ? Math.max(bin1[1], bin2[1]) : Math.max(bin1[0], bin2[0])
          // ) * 100).toFixed(2),
          // predictions: {
          //   binary_full: { real: (bin1[0]*100).toFixed(2)+"%", fake: (bin1[1]*100).toFixed(2)+"%" },
          //   binary_full_residual: { real: (bin2[0]*100).toFixed(2)+"%", fake: (bin2[1]*100).toFixed(2)+"%" }
          // }
        };

        // 6) 결과 다이얼로그 표시
        showResultDialog(finalResult);

      } catch (err) {
        console.error("분석 실패:", err);
        showErrorDialog("분석 중 오류: " + err.message);
      }

      sendResponse({ success: true });
      return true; // 비동기 응답
    }
});

  // ============================
  // 이미지 전처리 
  // ============================
  // Residual 전처리
  function preprocessImageResidual(img) {
    const size = 224;

    // 1) Resize
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = size;
    resizeCanvas.height = size;
    const ctx = resizeCanvas.getContext("2d");
    ctx.drawImage(img, 0, 0, size, size);

    const imageData = ctx.getImageData(0, 0, size, size);
    const gray = new Float32Array(size * size);

    // 2) Grayscale
    for (let i = 0; i < size * size; i++) {
      const r = imageData.data[i * 4 + 0];
      const g = imageData.data[i * 4 + 1];
      const b = imageData.data[i * 4 + 2];
      gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // 3) Blur (Gaussian 5x5, sigma=1.2)
    function makeGaussianKernel(k, sigma) {
      const kernel = [];
      let sum = 0;
      const half = Math.floor(k / 2);
      for (let y = -half; y <= half; y++) {
        for (let x = -half; x <= half; x++) {
          const v = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
          kernel.push(v);
          sum += v;
        }
      }
      return kernel.map(v => v / sum);
    }

    const ksize = 5, sigma = 1.2;
    const kernel = makeGaussianKernel(ksize, sigma);
    const half = Math.floor(ksize / 2);

    const blur = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let acc = 0;
        for (let ky = -half; ky <= half; ky++) {
          for (let kx = -half; kx <= half; kx++) {
            const iy = Math.min(size - 1, Math.max(0, y + ky));
            const ix = Math.min(size - 1, Math.max(0, x + kx));
            const val = gray[iy * size + ix];
            const kval = kernel[(ky + half) * ksize + (kx + half)];
            acc += val * kval;
          }
        }
        blur[y * size + x] = acc;
      }
    }

    // 4) Residual = gray - blur
    const residual = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
      residual[i] = gray[i] - blur[i];
    }

    // 5) Normalize (mean/std)
    let mean = 0, std = 0;
    for (let v of residual) mean += v;
    mean /= residual.length;
    for (let v of residual) std += (v - mean) ** 2;
    std = Math.sqrt(std / residual.length) + 1e-6;

    // 6) 3채널 복제
    const float32Data = new Float32Array(size * size * 3);
    let idx = 0;
    for (let i = 0; i < residual.length; i++) {
      const v = (residual[i] - mean) / std;
      float32Data[idx++] = v;
      float32Data[idx++] = v;
      float32Data[idx++] = v;
    }


    return { data: float32Data, width: size, height: size };
  }

  function preprocessImageViT(img) {
    const size = 224;   // ViT 기본 입력 사이즈
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, size, size);

    const imageData = ctx.getImageData(0, 0, size, size);
    const float32Data = new Float32Array(size * size * 3);

    // ImageNet 정규화 (ViT 순서)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < size * size; i++) {
      const r = imageData.data[i * 4 + 0] / 255.0;
      const g = imageData.data[i * 4 + 1] / 255.0;
      const b = imageData.data[i * 4 + 2] / 255.0;

      float32Data[i] = (r - mean[0]) / std[0];
      float32Data[size * size + i] = (g - mean[1]) / std[1];
      float32Data[2 * size * size + i] = (b - mean[2]) / std[2];
    }

    return { data: float32Data, width: size, height: size };
  }


  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }


  // ============================
  // UI 다이얼로그
  // ============================
  function showLoadingDialog() {
    removeLoadingDialog();
    const dialog = document.createElement("div");
    dialog.id = "ai-detector-dialog";
    dialog.innerHTML = `
      <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:9999">
        <div style="background:#fff;padding:20px;border-radius:10px;text-align:center">
          <div>🔄 이미지 분석 중...</div>
          <pre style="text-align:left;font-size:12px;white-space:pre-wrap;word-break:break-word;color:#000;background:#f8f8f8;padding:8px;border-radius:6px;max-height:200px;overflow:auto">
        </div>
      </div>`;
    document.body.appendChild(dialog);
  }

  function showResultDialog(result) {
    removeLoadingDialog();
    const dialog = document.createElement("div");
    dialog.id = "ai-detector-dialog";
    dialog.innerHTML = `
      <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:9999">
        <div style="background:#fff;padding:20px;border-radius:10px;max-width:400px;text-align:center;color:#333;">
          <h3 style="color:#333;margin-top:0;">AI 이미지 판별 결과</h3>
          <p style="color:#333;"><b>${result.modelType}</b></p>
          <button id="ai-detector-close-btn">닫기</button>
        </div>
      </div>`;
    document.body.appendChild(dialog);
    document.getElementById("ai-detector-close-btn").addEventListener("click", removeLoadingDialog);
  }

  function showErrorDialog(msg) {
    removeLoadingDialog();
    const dialog = document.createElement("div");
    dialog.id = "ai-detector-dialog";
    dialog.innerHTML = `
      <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:9999">
        <div style="background:#fff;padding:20px;border-radius:10px;text-align:center;color:red">
          <h3>오류 발생</h3>
          <p>${msg}</p>
          <button onclick="document.getElementById('ai-detector-dialog').remove()">닫기</button>
        </div>
      </div>`;
    document.body.appendChild(dialog);
  }

  function removeLoadingDialog() {
    const existing = document.getElementById("ai-detector-dialog");
    if (existing) existing.remove();
  }

  console.log("AI Image Detector Content Script 로드 완료");
})();
