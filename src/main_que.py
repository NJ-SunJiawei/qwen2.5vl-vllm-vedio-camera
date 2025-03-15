import asyncio
import base64
import cv2
import json
import numpy as np
import os
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from fastapi import FastAPI, WebSocket, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# 请确保 openai 模块已正确安装和配置
from openai import OpenAI

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --------------------
# 配置及全局变量
# --------------------
WORKER_COUNT = 4             # 处理帧的工作线程数量
SEND_WORKER_COUNT = 4        # 发送任务的工作线程数量
TARGET_FPS = 20              # 分析帧率
FRAME_SKIP = 50              # 基础跳帧数
TARGET_SEND_FPS = 15         # 发送帧率（用于控制发送速度，保证前端平稳）

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

client = OpenAI(
    base_url="http://124.220.202.52:8000/v1",
    api_key="EMPTY"
)

# 全局工作队列，每个工作线程对应一个 asyncio.Queue
worker_queues = [asyncio.Queue() for _ in range(WORKER_COUNT)]
# 增加线程池的最大线程数，便于处理更多并发任务
thread_executor = ThreadPoolExecutor(max_workers=10)

# 为发送任务采用多个队列，利用 hash 分发
file_send_queues = [asyncio.Queue() for _ in range(SEND_WORKER_COUNT)]
stream_send_queues = [asyncio.Queue() for _ in range(SEND_WORKER_COUNT)]

# 多用户会话管理（session_id 映射到 WebSocket 连接）
file_sessions = {}    # 本地视频会话
stream_sessions = {}  # 互联网流会话

# 针对互联网流，每个 session 可能有不同的检测目标
stream_detection = {}  # 键：session_id，值：检测目标字符串

# 为防止并发调用 OpenAI API 导致连接错误，增加全局信号量，限制同时并发调用数量
analysis_semaphore = threading.Semaphore(10)

# --------------------
# 公共函数
# --------------------
def get_effective_skip(queue_size, base_skip=FRAME_SKIP):
    if queue_size > 50:
        return base_skip * 2
    elif queue_size > 20:
        return int(base_skip * 1.5)
    else:
        return base_skip

async def analyze_frame(frame_np: bytes, object_str: str):
    base64_image = base64.b64encode(frame_np).decode('utf-8')
    prompt_str = f"""
    Analyze the image and extract the bounding box coordinates for the objects specified in the list: {object_str}.
    For each object found, return a JSON array of detections in the following format:
    [{{"bbox_2d": [x1, y1, x2, y2], "label": "object_label"}}, ...]
    If an object is not found, do not include it. If none found, return an empty list.
    """
    print("Sending request to OpenAI...")
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt_str},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}],
            max_tokens=1024,
        )
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        try:
            data = json.loads(response_text)
            if not isinstance(data, list):
                data = []
            return {"bboxes": data}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {"bboxes": []}
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

def process_frame_wrapper(frame_np, object_str):
    # 增加重试机制，最多重试 2 次
    for attempt in range(2):
        try:
            # 限制并发调用 API
            with analysis_semaphore:
                future = thread_executor.submit(lambda: asyncio.run(analyze_frame(frame_np, object_str)))
                return future.result()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    # 重试失败后返回空检测结果
    return {"bboxes": []}

# --------------------
# 发送任务（采用多线程和 hash 分发，独立发送 worker 处理，通过队列缓存，并限制 FPS）
# --------------------
async def send_file_results_worker(worker_index: int):
    interval = 1 / TARGET_SEND_FPS
    while True:
        start_time = time.time()
        if not file_send_queues[worker_index].empty():
            session_id, binary_frame, result = await file_send_queues[worker_index].get()
            ws = file_sessions.get(session_id)
            if ws:
                try:
                    await ws.send_bytes(binary_frame)
                    await ws.send_json(result)
                except Exception as e:
                    print(f"Error sending file result for session {session_id}: {e}")
            file_send_queues[worker_index].task_done()
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        await asyncio.sleep(sleep_time)

async def send_stream_results_worker(worker_index: int):
    interval = 1 / TARGET_SEND_FPS
    while True:
        start_time = time.time()
        if not stream_send_queues[worker_index].empty():
            session_id, binary_frame, result = await stream_send_queues[worker_index].get()
            ws = stream_sessions.get(session_id)
            if ws:
                try:
                    await ws.send_bytes(binary_frame)
                    await ws.send_json(result)
                except Exception as e:
                    print(f"Error sending stream result for session {session_id}: {e}")
            stream_send_queues[worker_index].task_done()
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        await asyncio.sleep(sleep_time)

# --------------------
# 工作线程任务（各 worker_queue 独立执行）
# --------------------
async def worker_task(worker_id: int, queue: asyncio.Queue):
    loop = asyncio.get_running_loop()
    while True:
        # 任务格式：(session_id, frame_id, binary_frame, object_str, second, mode)
        session_id, frame_id, binary_frame, object_str, second, mode = await queue.get()
        effective_skip = get_effective_skip(queue.qsize(), FRAME_SKIP)
        if frame_id % effective_skip == 0 and object_str:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: process_frame_wrapper(binary_frame, object_str)
                )
            except Exception as e:
                print(f"Frame {frame_id} processing error: {e}")
                result = {"bboxes": []}
        else:
            result = {"bboxes": []}
        # 使用 hash 分发，将结果放入对应的发送队列
        if mode == "file":
            send_worker_index = hash(session_id) % SEND_WORKER_COUNT
            await file_send_queues[send_worker_index].put((session_id, binary_frame, result))
        else:
            send_worker_index = hash(session_id) % SEND_WORKER_COUNT
            await stream_send_queues[send_worker_index].put((session_id, binary_frame, result))
        print(f"Worker {worker_id}: Processed frame {frame_id} for session {session_id}. Queue size: {queue.qsize()}")
        queue.task_done()

# --------------------
# 启动任务：启动 worker 任务和发送任务
# --------------------
@app.on_event("startup")
async def startup_event():
    for worker_id, q in enumerate(worker_queues):
        asyncio.create_task(worker_task(worker_id, q))
    for i in range(SEND_WORKER_COUNT):
        asyncio.create_task(send_file_results_worker(i))
        asyncio.create_task(send_stream_results_worker(i))

# --------------------
# 数据模型
# --------------------
class AnalyzeRequest(BaseModel):
    session_id: str
    object_str: str
    filename: str

class StreamRequest(BaseModel):
    session_id: str
    stream_url: str

class StreamAnalyzeRequest(BaseModel):
    session_id: str
    object_str: str

# --------------------
# WebSocket 接口（要求客户端在 URL 中传递 session_id 参数，并增加心跳处理）
# --------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    file_sessions[session_id] = websocket
    print(f"本地视频 WebSocket connected, session_id: {session_id}")
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                continue
    except Exception as e:
        print(f"WebSocket error (file) session {session_id}: {e}")
    finally:
        file_sessions.pop(session_id, None)

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    stream_sessions[session_id] = websocket
    print(f"互联网流 WebSocket connected, session_id: {session_id}")
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                continue
    except Exception as e:
        print(f"WebSocket error (stream) session {session_id}: {e}")
    finally:
        stream_sessions.pop(session_id, None)

# --------------------
# 接口：上传视频（本地视频模式）
# --------------------
@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    print("视频上传，保存中...")
    video_path = UPLOAD_DIR / video.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    return {"status": "success", "message": "视频上传成功", "filename": video.filename}

# --------------------
# 接口：本地视频分析
# --------------------
@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    session_id = request.session_id
    object_str = request.object_str
    filename = request.filename
    print(f"开始分析本地视频，session: {session_id}, 目标对象: {object_str}")
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return {"status": "error", "message": "视频文件不存在"}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": "无法打开视频文件"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = TARGET_FPS
    frame_interval = max(1, fps // TARGET_FPS)
    print(f"原始fps: {fps}, 目标fps: {TARGET_FPS}")

    frame_id, second = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
            _, buffer = cv2.imencode(".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            binary_frame = buffer.tobytes()
            worker_index = hash(session_id) % WORKER_COUNT
            await worker_queues[worker_index].put((session_id, frame_id, binary_frame, object_str, second, "file"))
            print(f"Frame {frame_id} added to worker {worker_index} queue for session {session_id}")
        frame_id += 1
        second = frame_id // fps
    cap.release()
    os.remove(video_path)
    return {"status": "processing", "message": "视频分析中..."}

# --------------------
# 接口：启动互联网流拉流
# --------------------
@app.post("/stream")
async def start_stream(request: StreamRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id
    stream_url = request.stream_url
    if not stream_url:
        return {"status": "error", "message": "缺少流地址"}
    print(f"启动互联网流拉流，session: {session_id}, 流地址: {stream_url}")
    background_tasks.add_task(stream_video_reader, session_id, stream_url)
    return {"status": "success", "message": "互联网流拉流启动", "stream_url": stream_url}

# --------------------
# 接口：设置互联网流检测目标
# --------------------
@app.post("/stream/analyze")
async def analyze_stream(request: StreamAnalyzeRequest):
    session_id = request.session_id
    object_str = request.object_str
    if not object_str:
        return {"status": "error", "message": "缺少检测对象"}
    stream_detection[session_id] = object_str
    print(f"设置互联网流检测目标，session: {session_id}, 对象: {object_str}")
    return {"status": "success", "message": "互联网流目标检测已启动", "object": object_str}

# --------------------
# 互联网流视频读取任务（支持 RTSP），每个 session 独立拉流
# --------------------
async def stream_video_reader(session_id: str, stream_url: str):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"无法打开网络流: {stream_url}，session: {session_id}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = TARGET_FPS
    frame_interval = max(1, int(fps // TARGET_FPS))
    frame_id = 0
    print(f"开始拉流，session: {session_id}，原始fps: {fps}, 目标fps: {TARGET_FPS}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"读取网络流帧失败，session: {session_id}，等待重试...")
            await asyncio.sleep(0.1)
            continue
        if frame_id % frame_interval == 0:
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
            _, buffer = cv2.imencode(".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            binary_frame = buffer.tobytes()
            second = frame_id // int(fps)
            object_str = stream_detection.get(session_id, "")
            worker_index = hash(session_id) % WORKER_COUNT
            await worker_queues[worker_index].put((session_id, frame_id, binary_frame, object_str, second, "stream"))
        frame_id += 1
        await asyncio.sleep(0)
    cap.release()

# --------------------
# 首页
# --------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
