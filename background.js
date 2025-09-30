// background.js
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "ai-image-detector",
    title: "AI 이미지 판별",
    contexts: ["image"]
  });
  console.log("🚀 AI Image Detector 초기화 완료 (background)");
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "ai-image-detector") {
    console.log("🖱️ 컨텍스트 메뉴 클릭:", info.srcUrl);
    
    try {
      await chrome.tabs.sendMessage(tab.id, {
        action: "detectAI",
        imageUrl: info.srcUrl
      });
    } catch (error) {
      console.error("Content script 메시지 전송 실패:", error);
      
      // Content script가 로드되지 않은 경우 스크립트 주입
      try {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['content.js']
        });
        
        // 잠시 대기 후 다시 메시지 전송
        setTimeout(async () => {
          try {
            await chrome.tabs.sendMessage(tab.id, {
              action: "detectAI",
              imageUrl: info.srcUrl
            });
          } catch (retryError) {
            console.error("재시도 실패:", retryError);
          }
        }, 1000);
        
      } catch (scriptError) {
        console.error("스크립트 주입 실패:", scriptError);
      }
    }
  }
});

// 메시지 수신 처리
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("Background 메시지 수신:", message);

  // popup.js에서 모델 상태 요청 처리
  if (message.action === "getModelStatus") {
    // 현재 활성 탭의 content script로 메시지 전달
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (tabs[0]) {
        try {
          const response = await chrome.tabs.sendMessage(tabs[0].id, message);
          sendResponse(response || {
            isInitialized: false,
            totalModels: 2,
            loadedModels: [],
            loadSuccess: 0,
            loadFailed: 0,
            failedModels: []
          });
        } catch (error) {
          console.log("Content script 응답 없음, 기본값 반환");
          sendResponse({
            isInitialized: false,
            totalModels: 2,
            loadedModels: [],
            loadSuccess: 0,
            loadFailed: 0,
            failedModels: []
          });
        }
      } else {
        sendResponse({
          isInitialized: false,
          totalModels: 2,
          loadedModels: [],
          loadSuccess: 0,
          loadFailed: 0,
          failedModels: []
        });
      }
    });
    return true;
  }

  if (message.action === "reloadModels") {
    // content script로 메시지 전달
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (tabs[0]) {
        try {
          const response = await chrome.tabs.sendMessage(tabs[0].id, message);
          sendResponse(response || { ok: true });
        } catch (error) {
          console.error("모델 재로드 실패:", error);
          sendResponse({ ok: false, error: "Content script에 접근할 수 없습니다." });
        }
      } else {
        sendResponse({ ok: false, error: "활성 탭을 찾을 수 없습니다." });
      }
    });
    return true;
  }

  // 이미지 fetch → CORS 방지
  if (message.action === "fetchImage") {
    console.log("이미지 fetch 요청:", message.url);
    
    fetch(message.url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.blob();
      })
      .then((blob) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          console.log("이미지 fetch 성공");
          sendResponse({ ok: true, dataUrl: reader.result });
        };
        reader.onerror = () => {
          console.error("FileReader 실패");
          sendResponse({ ok: false, error: "파일 읽기 실패" });
        };
        reader.readAsDataURL(blob);
      })
      .catch((err) => {
        console.error("❌ fetch 실패:", err);
        sendResponse({ ok: false, error: err.message });
      });
    return true;
  }
});