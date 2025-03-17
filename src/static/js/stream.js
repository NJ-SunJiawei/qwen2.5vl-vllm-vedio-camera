// ========== 多路互联网流相关代码 ==========
const fixedCanvasW = 400;
const fixedCanvasH = 300;
let channelCounter = 0;
const containerStream = document.getElementById("multiStreamContainer");

// 新增卡片和恢复默认布局按钮事件
document.getElementById("addChannelBtn").addEventListener("click", () => {
  addStreamCard();
});
document.getElementById("resetLayoutBtn").addEventListener("click", () => {
  resetLayout();
});

// 恢复所有卡片原来排版（删除拖拽、调整尺寸的内联样式）
function resetLayout() {
  const cards = containerStream.querySelectorAll(".stream-card");
  cards.forEach(card => {
    card.style.transform = "";
    card.removeAttribute("data-x");
    card.removeAttribute("data-y");
    card.style.width = "";
    card.style.height = "";
    const videoWrapper = card.querySelector('.video-wrapper');
    if (videoWrapper) {
      const canvas = videoWrapper.querySelector("canvas");
      if (canvas) {
        canvas.width = videoWrapper.clientWidth;
        canvas.height = videoWrapper.clientHeight;
      }
    }
  });
}

function addStreamCard() {
  channelCounter++;
  const sessionId = "stream_session_" + Date.now() + "_" + channelCounter;
  const cardDiv = document.createElement("div");
  cardDiv.className = "stream-card";
  // 用于记录流状态：未启动、已启动、暂停
  cardDiv._streamRunning = false;
  cardDiv._paused = false;

  // 卡片头部：显示标题和删除按钮（标题设为可编辑）
  const headerDiv = document.createElement("div");
  headerDiv.className = "card-header";

  const titleEl = document.createElement("div");
  titleEl.className = "card-title";
  titleEl.contentEditable = "true";
  titleEl.innerText = "通道 " + channelCounter;
  headerDiv.appendChild(titleEl);

  const btnDelete = document.createElement("button");
  btnDelete.className = "btn-delete";
  btnDelete.innerText = "删除";
  btnDelete.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await stopStream(sessionId, statusDiv);
    } catch (error) {
      console.error("Error stopping stream:", error);
    } finally {
      cardDiv.remove();
    }
  });
  headerDiv.appendChild(btnDelete);
  cardDiv.appendChild(headerDiv);

  // 创建状态显示（离线/在线）--稍后添加到视频播放区域内
  const statusDiv = document.createElement("div");
  statusDiv.className = "card-status status-offline";
  statusDiv.innerText = "离线";

  // 输入框区域
  const addrInput = document.createElement("input");
  addrInput.className = "card-input";
  addrInput.placeholder = "请输入流地址 (rtsp/rtmp...)";
  cardDiv.appendChild(addrInput);

  const detectInput = document.createElement("input");
  detectInput.className = "card-input";
  detectInput.placeholder = "请输入检测目标(多个用逗号)";
  cardDiv.appendChild(detectInput);

  // 按钮组，合并拉流/停止和暂停/恢复按钮
  const btnGroup = document.createElement("div");
  btnGroup.className = "btn-group";
  cardDiv.appendChild(btnGroup);

  // 拉流/停止按钮
  const btnToggleStream = document.createElement("button");
  // 初始状态使用 btn-start 样式，显示“拉流”
  btnToggleStream.className = "btn-start";
  btnToggleStream.innerText = "拉流";
  btnToggleStream.onclick = async () => {
    const url = addrInput.value.trim();
    if (!url) {
      alert("请先输入流地址");
      return;
    }
    if (!cardDiv._streamRunning) {
      await startStream(sessionId, url, statusDiv);
      cardDiv._streamRunning = true;
      btnToggleStream.className = "btn-stop";
      btnToggleStream.innerText = "停止";
      // 启动后使暂停/恢复按钮可用
      btnTogglePause.disabled = false;
    } else {
      await stopStream(sessionId, statusDiv);
      cardDiv._streamRunning = false;
      cardDiv._paused = false;
      btnToggleStream.className = "btn-start";
      btnToggleStream.innerText = "拉流";
      // 重置暂停/恢复按钮状态
      btnTogglePause.className = "btn-pause";
      btnTogglePause.innerText = "暂停";
      btnTogglePause.disabled = true;
    }
  };
  btnGroup.appendChild(btnToggleStream);

  // 暂停/恢复按钮
  const btnTogglePause = document.createElement("button");
  btnTogglePause.className = "btn-pause";
  btnTogglePause.innerText = "暂停";
  btnTogglePause.disabled = true;
  btnTogglePause.onclick = async () => {
    if (!cardDiv._paused) {
      await pauseStream(sessionId);
      cardDiv._paused = true;
      btnTogglePause.className = "btn-resume";
      btnTogglePause.innerText = "恢复";
    } else {
      await resumeStream(sessionId);
      cardDiv._paused = false;
      btnTogglePause.className = "btn-pause";
      btnTogglePause.innerText = "暂停";
    }
  };
  btnGroup.appendChild(btnTogglePause);

  // AI识别按钮
  const btnDetect = document.createElement("button");
  btnDetect.className = "btn-detect";
  btnDetect.innerText = "AI识别";
  btnDetect.onclick = () => {
    const objs = detectInput.value.trim();
    if (objs) setAnalyze(sessionId, objs);
  };
  btnGroup.appendChild(btnDetect);

  // 视频播放区域 ：video-wrapper填满卡片内容宽度，保持4:3比例
  const videoWrapper = document.createElement("div");
  videoWrapper.className = "video-wrapper";
  const canvas = document.createElement("canvas");
  canvas.width = fixedCanvasW;
  canvas.height = fixedCanvasH;
  canvas.style.backgroundColor = "#000";
  videoWrapper.appendChild(canvas);

// 将FPS显示和状态显示添加到视频播放区域内（分别位于左上和右上）
  const fpsDiv = document.createElement("div");
  fpsDiv.className = "card-fps";
  fpsDiv.innerText = "FPS: 0";
  videoWrapper.appendChild(fpsDiv);
  videoWrapper.appendChild(statusDiv);

  cardDiv.appendChild(videoWrapper);
  containerStream.appendChild(cardDiv);

  cardDiv._sessionId = sessionId;
  cardDiv._statusDiv = statusDiv;
  cardDiv._canvas = canvas;
  cardDiv._ctx = canvas.getContext("2d");
  cardDiv._fpsDisplay = fpsDiv;
  cardDiv._fpsCount = 0;
  cardDiv._fpsLastTime = Date.now();
  cardDiv._currentFPS = 0;
  cardDiv._lastDetection = null;
  cardDiv._lastDetectionTimestamp = 0;

  // 拖拽 & 调整尺寸
  interact(cardDiv)
    .draggable({
      inertia: true,
      modifiers: [
        interact.modifiers.restrictRect({
          restriction: 'parent',
          endOnly: true
        })
      ],
      listeners: {
        move (event) {
          const target = event.target;
          let x = (parseFloat(target.getAttribute('data-x')) || 0) + event.dx;
          let y = (parseFloat(target.getAttribute('data-y')) || 0) + event.dy;
          target.style.transform = 'translate(' + x + 'px, ' + y + 'px)';
          target.setAttribute('data-x', x);
          target.setAttribute('data-y', y);
        }
      }
    })
    .resizable({
      edges: { left: true, right: true, bottom: true, top: true },
      modifiers: [
        interact.modifiers.restrictSize({
          min: { width: 350, height: 300 }
        })
      ],
      inertia: true,
      listeners: {
        move (event) {
          const target = event.target;
          let x = (parseFloat(target.getAttribute('data-x')) || 0);
          let y = (parseFloat(target.getAttribute('data-y')) || 0);
          target.style.width = event.rect.width + 'px';
          target.style.height = event.rect.height + 'px';
          x += event.deltaRect.left;
          y += event.deltaRect.top;
          target.style.transform = 'translate(' + x + 'px, ' + y + 'px)';
          target.setAttribute('data-x', x);
          target.setAttribute('data-y', y);

          const videoWrapper = target.querySelector('.video-wrapper');
          if (videoWrapper) {
            videoWrapper.style.width = "100%";
            const canvas = videoWrapper.querySelector("canvas");
            if (canvas) {
              canvas.width = videoWrapper.clientWidth;
              canvas.height = videoWrapper.clientHeight;
            }
          }
        }
      }
    });
}

async function startStream(session_id, stream_url, statusEl) {
  try {
    const res = await fetch("/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, stream_url })
    });
    const result = await res.json();
    console.log("startStream:", result);
    if (result.status === "success") {
      createStreamWS(session_id, statusEl);
    } else {
      alert(result.message || "启动拉流失败");
    }
  } catch (e) {
    console.error(e);
    alert("拉流请求失败");
  }
}

async function pauseStream(session_id) {
  try {
    const res = await fetch("/stream/pause", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id })
    });
    const r = await res.json();
    console.log("pauseStream:", r);
  } catch (e) {
    console.error(e);
  }
}

async function resumeStream(session_id) {
  try {
    const res = await fetch("/stream/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id })
    });
    const r = await res.json();
    console.log("resumeStream:", r);
  } catch (e) {
    console.error(e);
  }
}

async function stopStream(session_id, statusEl) {
  try {
    const res = await fetch("/stream/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id })
    });
    const r = await res.json();
    console.log("stopStream:", r);
    statusEl.innerText = "离线";
    statusEl.classList.remove("status-live");
    statusEl.classList.add("status-offline");
  } catch (e) {
    console.error(e);
  }
}

async function setAnalyze(session_id, object_str) {
  try {
    const res = await fetch("/stream/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, object_str })
    });
    const r = await res.json();
    console.log("setAnalyze:", r);
  } catch (e) {
    console.error(e);
  }
}

function createStreamWS(session_id, statusEl) {
  const wsProto = (location.protocol === "https:") ? "wss:" : "ws:";
  const wsUrl = `${wsProto}//${window.location.host}/ws/stream?session_id=${session_id}`;
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    console.log("WebSocket connected for session:", session_id);
    statusEl.innerText = "Live";
    statusEl.classList.remove("status-offline");
    statusEl.classList.add("status-live");
  };
  ws.onmessage = async (evt) => {
    let msg;
    if (typeof evt.data === "string") {
      msg = JSON.parse(evt.data);
    } else {
      const text = await evt.data.text();
      msg = JSON.parse(text);
    }
    if (msg.image) {
      const blob = await base64ToBlob(msg.image);
      const card = findCardBySession(session_id);
      if (card) {
        card._fpsCount++;
        const now = Date.now();
        if (now - card._fpsLastTime >= 1000) {
          card._currentFPS = card._fpsCount;
          card._fpsCount = 0;
          card._fpsLastTime = now;
          if (card._fpsDisplay) {
            card._fpsDisplay.innerText = "FPS: " + card._currentFPS;
          }
        }
        if (msg.result && msg.result.bboxes && msg.result.bboxes.length > 0) {
          card._lastDetection = msg.result;
          card._lastDetectionTimestamp = Date.now();
          console.log("Stream detection received for session", session_id, msg.result.bboxes);
        }
        drawStreamFrame(card, blob, card._lastDetection);
      }
    }
  };
  ws.onclose = () => {
    console.log("WebSocket closed for session:", session_id);
  };
  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
  window.stream_sessions[session_id] = ws;
}

function findCardBySession(session_id) {
  const cards = containerStream.querySelectorAll(".stream-card");
  for (const c of cards) {
    if (c._sessionId === session_id) return c;
  }
  return null;
}

function drawStreamFrame(card, blob, detection) {
  createImageBitmap(blob).then((bitmap) => {
    const canvasW = card._canvas.width;
    const canvasH = card._canvas.height;
    const ctx = card._ctx;
    ctx.clearRect(0, 0, canvasW, canvasH);
    const ratio = Math.min(canvasW / bitmap.width, canvasH / bitmap.height);
    const newWidth = bitmap.width * ratio;
    const newHeight = bitmap.height * ratio;
    const offsetX = (canvasW - newWidth) / 2;
    const offsetY = (canvasH - newHeight) / 2;
    ctx.drawImage(bitmap, offsetX, offsetY, newWidth, newHeight);

    if (detection && detection.bboxes && detection.bboxes.length > 0) {
      if (Date.now() - card._lastDetectionTimestamp > 1200) detection = null;
      else {
        detection.bboxes.forEach(det => {
          if (!det.bbox_2d || det.bbox_2d.length < 4) return;
          const [x1, y1, x2, y2] = det.bbox_2d;
          const boxX = offsetX + x1 * ratio;
          const boxY = offsetY + y1 * ratio;
          const boxW = (x2 - x1) * ratio;
          const boxH = (y2 - y1) * ratio;
          ctx.strokeStyle = "lime";
          ctx.lineWidth = 2;
          ctx.font = "14px Arial";
          ctx.fillStyle = "lime";
          ctx.strokeRect(boxX, boxY, boxW, boxH);
          ctx.fillText(det.label, boxX, boxY - 5);
        });
      }
    }
  }).catch(err => console.error("Error drawing stream frame:", err));
}

// ====== 以下为多路互联网流卡片筛选功能 ======

// 筛选函数：遍历所有流卡片，判断是否显示
function filterStreamCards() {
  const channelNameFilter = document.getElementById('channelNameFilter').value.trim().toLowerCase();
  const statusFilter = document.getElementById('statusFilter').value; // 值为 ""、"live" 或 "offline"

  // 遍历 containerStream 内所有卡片
  const cards = containerStream.querySelectorAll(".stream-card");
  cards.forEach(card => {
    // 获取通道名称（假设标题在 .card-title 内）
    const titleEl = card.querySelector('.card-title');
    const channelName = titleEl ? titleEl.innerText.trim().toLowerCase() : "";

    // 获取状态（假设状态在 .card-status 内，状态文字 "Live" 表示在线，"离线" 表示离线）
    const statusEl = card.querySelector('.card-status');
    let cardStatus = "";
    if (statusEl) {
      // 可根据实际情况调整判断（例如 "Live" 表示在线）
      cardStatus = statusEl.innerText.trim().toLowerCase();
      // 统一转换为 "live" 或 "offline"
      if (cardStatus === "live") {
        cardStatus = "live";
      } else {
        cardStatus = "offline";
      }
    }

    // 判断是否满足筛选条件
    let show = true;
    if (channelNameFilter && !channelName.includes(channelNameFilter)) {
      show = false;
    }
    if (statusFilter && cardStatus !== statusFilter) {
      show = false;
    }
    // 显示或隐藏卡片
    card.style.display = show ? "" : "none";
  });
}

// 清除筛选：将筛选输入框置空，同时显示所有卡片
function clearStreamFilters() {
  document.getElementById('channelNameFilter').value = "";
  document.getElementById('statusFilter').value = "";
  // 恢复显示所有卡片
  const cards = containerStream.querySelectorAll(".stream-card");
  cards.forEach(card => {
    card.style.display = "";
  });
}

// 绑定筛选按钮事件
document.getElementById('applyFilterBtn').addEventListener('click', () => {
  filterStreamCards();
});

// 绑定清除筛选按钮事件
document.getElementById('clearFilterBtn').addEventListener('click', () => {
  clearStreamFilters();
});

// 也可以为筛选输入框绑定键盘事件（例如回车键）
// document.getElementById('channelNameFilter').addEventListener('keyup', (e) => {
//   if (e.key === 'Enter') filterStreamCards();
// });
