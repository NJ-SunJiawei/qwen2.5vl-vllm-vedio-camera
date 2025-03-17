// 本地视频逻辑
const localSessionId = Date.now().toString();
let local_ws;
let localFPSCount = 0, localFPSLastTime = Date.now(), localCurrentFPS = 0;
let local_lastDetection = null, local_lastDetectionTimestamp = 0;
let lastThumbnailTime = 0;
let fileThumbnailList = [];

// 建立本地视频 WebSocket
function createLocalWS() {
  const wsUrl = `ws://${window.location.host}/ws?session_id=${localSessionId}`;
  local_ws = new WebSocket(wsUrl);
  local_ws.onopen = () => {
    console.log("Local video WebSocket connected, session_id:", localSessionId);
  };
  local_ws.onmessage = async (evt) => {
    let msg;
    if (typeof evt.data === "string") {
      msg = JSON.parse(evt.data);
    } else {
      const text = await evt.data.text();
      msg = JSON.parse(text);
    }
    localFPSCount++;
    if (Date.now() - localFPSLastTime >= 1000) {
      localCurrentFPS = localFPSCount;
      localFPSCount = 0;
      localFPSLastTime = Date.now();
      document.getElementById("fileFpsDisplay").innerText = "FPS: " + localCurrentFPS;
    }
    if (msg.image) {
      const blob = await base64ToBlob(msg.image);
      if (msg.result && msg.result.bboxes && msg.result.bboxes.length > 0) {
        local_lastDetection = msg.result;
        local_lastDetectionTimestamp = Date.now();
        console.log("Local detection received:", msg.result.bboxes);
        if (Date.now() - lastThumbnailTime > 3000) {
          addFileThumbnail();
          lastThumbnailTime = Date.now();
        }
      }
      drawLocalFrame(blob, local_lastDetection);
    }
  };
  local_ws.onclose = () => {
    console.log("Local video WebSocket closed");
  };
  local_ws.onerror = (err) => {
    console.error("Local video WebSocket error:", err);
  };
}

// 绘制本地视频帧
function drawLocalFrame(blob, detection) {
  createImageBitmap(blob).then((bitmap) => {
    const canvasW = 640, canvasH = 480;
    const videoCanvas = document.getElementById("fileVideoCanvas");
    const overlayCanvas = document.getElementById("fileOverlayCanvas");
    videoCanvas.width = canvasW;
    videoCanvas.height = canvasH;
    overlayCanvas.width = canvasW;
    overlayCanvas.height = canvasH;

    const ctx = videoCanvas.getContext("2d");
    ctx.clearRect(0, 0, canvasW, canvasH);
    const ratio = Math.min(canvasW / bitmap.width, canvasH / bitmap.height);
    const newWidth = bitmap.width * ratio;
    const newHeight = bitmap.height * ratio;
    const offsetX = (canvasW - newWidth) / 2;
    const offsetY = (canvasH - newHeight) / 2;
    ctx.drawImage(bitmap, offsetX, offsetY, newWidth, newHeight);

    const overlayCtx = overlayCanvas.getContext("2d");
    overlayCtx.clearRect(0, 0, canvasW, canvasH);

    if (detection && detection.bboxes && detection.bboxes.length > 0) {
      if (Date.now() - local_lastDetectionTimestamp > 2000) detection = null;
      else {
        detection.bboxes.forEach(det => {
          if (!det.bbox_2d || det.bbox_2d.length < 4) return;
          const [x1, y1, x2, y2] = det.bbox_2d;
          const boxX = offsetX + x1 * ratio;
          const boxY = offsetY + y1 * ratio;
          const boxW = (x2 - x1) * ratio;
          const boxH = (y2 - y1) * ratio;
          overlayCtx.strokeStyle = "lime";
          overlayCtx.lineWidth = 2;
          overlayCtx.font = "14px Arial";
          overlayCtx.fillStyle = "lime";
          overlayCtx.strokeRect(boxX, boxY, boxW, boxH);
          overlayCtx.fillText(det.label, boxX, boxY - 5);
          console.log("Local detection bbox:", det.bbox_2d, "label:", det.label);
        });
      }
    }
  }).catch(err => console.error("Error drawing local frame:", err));
}

// 缩略图相关
function addFileThumbnail() {
  if (!local_lastDetection || !local_lastDetection.bboxes || local_lastDetection.bboxes.length === 0) return;
  // 延时100毫秒再捕获，确保canvas已绘制完成
  setTimeout(() => {
    const fileVideoCanvas = document.getElementById('fileVideoCanvas');
    const dataURL = fileVideoCanvas.toDataURL("image/jpeg", 0.5);
    const labels = local_lastDetection.bboxes.map(det => det.label);
    const uniqueLabels = [...new Set(labels)];
    const labelText = uniqueLabels.join(", ");

    const thumbnailItem = document.createElement("div");
    thumbnailItem.className = "thumbnail-item";

    const img = document.createElement("img");
    img.src = dataURL;
    fileThumbnailList.push(dataURL);
    img.dataset.index = fileThumbnailList.length - 1;
    img.addEventListener("click", () => {
      showModal(fileThumbnailList, parseInt(img.dataset.index));
    });
    thumbnailItem.appendChild(img);

    const labelDiv = document.createElement("div");
    labelDiv.className = "thumbnail-label";
    labelDiv.innerText = labelText;
    thumbnailItem.appendChild(labelDiv);

    const container = document.getElementById('fileThumbnailContainer');
    container.appendChild(thumbnailItem);

    while (fileThumbnailList.length > 6) {
      fileThumbnailList.shift();
      container.removeChild(container.firstChild);
    }
    updateThumbnailIndices(container);
  }, 100);
}

function updateThumbnailIndices(container) {
  Array.from(container.children).forEach((child, index) => {
    const img = child.querySelector('img');
    if (img) img.dataset.index = index;
  });
}

// 文件上传
document.getElementById('fileUploadForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('videoInput');
  if (fileInput.files.length === 0) {
    alert("请选择视频文件");
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('video', file);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload');
  xhr.upload.onprogress = function (event) {
    if (event.lengthComputable) {
      const percent = Math.round((event.loaded / event.total) * 100);
      document.getElementById('progressBar').style.width = percent + "%";
      document.getElementById('progressText').innerText = percent + "%";
    }
  };
  xhr.onload = function () {
    if (xhr.status === 200) {
      const result = JSON.parse(xhr.responseText);
      document.getElementById('fileAnalyzeSection').classList.remove('hidden');
      document.getElementById('fileAnalyzeButton').dataset.filename = result.filename;
    } else {
      alert("上传视频失败，请重试");
    }
  };
  xhr.onerror = function () {
    alert("上传视频失败，请检查网络");
  };
  xhr.send(formData);
});

// 启动本地视频识别
document.getElementById('fileAnalyzeButton').addEventListener('click', async () => {
  const objectStr = document.getElementById('fileObjectInput').value;
  if (!objectStr) {
    alert("请输入要检测的对象");
    return;
  }
  const filename = document.getElementById('fileAnalyzeButton').dataset.filename;
  if (!filename) {
    alert("请先上传视频");
    return;
  }
  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        session_id: localSessionId, 
        object_str: objectStr, 
        filename 
      })
    });
    if (!response.ok) throw new Error("网络响应异常");
    console.log("本地视频识别已启动");
  } catch (error) {
    alert("启动识别失败，请重试");
  }
});

// 全屏按钮
document.getElementById("fileFullScreenButton").addEventListener("click", (e) => {
  e.stopPropagation();
  toggleFullScreen("fileVideoContainer");
});

// 初始化
createLocalWS();
