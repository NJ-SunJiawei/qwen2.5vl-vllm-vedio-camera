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
import logging
import collections

from fastapi import FastAPI, WebSocket, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# 请确保 openai 模块已正确安装和配置
from openai import OpenAI

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --------------------
# 宏开关设置
# --------------------
DIRECT_SEND_MODE = False
WORKER_COUNT = 8             # 处理帧的工作线程数量
SEND_WORKER_COUNT = 8        # 发送任务的工作线程数量（仅在 DIRECT_SEND_MODE=False 时使用）
OPENAI_WORKER_COUNT = 10     # 调用大模型线程池数量
TARGET_FPS = 50              # 发送帧率
ANALYSIS_FPS = 20            # 分析帧率
FRAME_SKIP = 100             # 基础跳帧数

client = OpenAI(
    base_url="http://124.220.202.52:8000/v1",
    api_key="EMPTY",
    max_retries=0  # 只重试 0 次
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

worker_queues = [asyncio.Queue() for _ in range(WORKER_COUNT)]
thread_executor = ThreadPoolExecutor(max_workers=OPENAI_WORKER_COUNT)

file_send_queues = [asyncio.Queue() for _ in range(SEND_WORKER_COUNT)]
stream_send_queues = [asyncio.Queue() for _ in range(SEND_WORKER_COUNT)]

file_sessions = {}    # 本地视频会话
stream_sessions = {}  # 互联网流会话

stream_detection = {}  # 键：session_id，值：检测目标字符串
stream_state = {}      # 键：session_id，值：状态字符串

analysis_semaphore = threading.Semaphore(5)
direct_send_locks = {}       # key: session_id, value: asyncio.Lock()
direct_send_last_time = {}   # key: session_id, value: timestamp (float)

# 新增：记录每个 session 的发送 fps 统计数据
send_fps_data = {}  # key: session_id, value: {"frame_count": int, "last_time": float, "actual_fps": float}

def get_effective_skip(queue_size, base_skip=FRAME_SKIP):
    if queue_size > 50:
        return base_skip * 2
    elif queue_size > 20:
        return int(base_skip * 1.5)
    else:
        return base_skip

def build_combined_result(session_id: str, binary_frame: bytes, result: dict) -> dict:
    """更新指定 session 的 fps 统计，并返回带有实际 fps 数据的发送内容"""
    now = time.time()
    fps_data = send_fps_data.setdefault(session_id, {"frame_count": 0, "last_time": now, "actual_fps": 0})
    fps_data["frame_count"] += 1
    if now - fps_data["last_time"] >= 1.0:
        fps_data["actual_fps"] = fps_data["frame_count"] / (now - fps_data["last_time"])
        logging.info(f"Session {session_id}: actual sending FPS: {fps_data['actual_fps']:.2f}")
        fps_data["frame_count"] = 0
        fps_data["last_time"] = now
    return {
        "image": base64.b64encode(binary_frame).decode("utf-8"),
        "result": result,
        "actual_fps": fps_data["actual_fps"]
    }

async def analyze_frame(session_id: str, frame_id: int, frame_np: bytes, object_str: str):
    base64_image = base64.b64encode(frame_np).decode('utf-8')
    prompt_str = f"""
    Analyze the image and extract the bounding box coordinates for the objects specified in the list: {object_str}.
    For each object found, return a JSON array of detections in the following format:
    [{{"bbox_2d": [x1, y1, x2, y2], "label": "object_label"}}, ...]
    If an object is not found, do not include it. If none found, return an empty list.
    """
    logging.info(f"Sending request to OpenAI, session: {session_id}, frame: {frame_id}")
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
            logging.error(f"JSON decode error: {e}")
            return {"bboxes": []}
    except Exception as e:
        logging.error(f"Error during analysis, session: {session_id}, frame: {frame_id}: {e}")
        raise

def process_frame_wrapper(session_id, frame_id, frame_np, object_str):
    for attempt in range(1):
        try:
            with analysis_semaphore:
                future = thread_executor.submit(
                    lambda: asyncio.run(analyze_frame(session_id, frame_id, frame_np, object_str))
                )
                return future.result()
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed for session {session_id}, frame {frame_id}: {e}")
            time.sleep(1)
    return {"bboxes": []}

async def send_result_direct(session_id: str, binary_frame: bytes, result: dict, mode: str):
    ws = file_sessions.get(session_id) if mode == "file" else stream_sessions.get(session_id)
    if not ws:
        return
    lock = direct_send_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        now = time.time()
        last_time = direct_send_last_time.get(session_id, 0)
        interval = 1 / TARGET_FPS
        sleep_time = interval - (now - last_time)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        combined = build_combined_result(session_id, binary_frame, result)
        try:
            await ws.send_json(combined)
        except Exception as e:
            logging.error(f"Error sending result for session {session_id}: {e}")
        direct_send_last_time[session_id] = time.time()

async def send_file_results_worker(worker_index: int):
    interval = 1 / TARGET_FPS
    while True:
        start_time = time.time()
        if not file_send_queues[worker_index].empty():
            session_id, binary_frame, result = await file_send_queues[worker_index].get()
            ws = file_sessions.get(session_id)
            if ws:
                try:
                    combined = build_combined_result(session_id, binary_frame, result)
                    await ws.send_json(combined)
                except Exception as e:
                    logging.error(f"Error sending file result for session {session_id}: {e}")
            file_send_queues[worker_index].task_done()
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        await asyncio.sleep(sleep_time)

async def send_stream_results_worker(worker_index: int):
    interval = 1 / TARGET_FPS
    while True:
        start_time = time.time()
        if not stream_send_queues[worker_index].empty():
            session_id, binary_frame, result = await stream_send_queues[worker_index].get()
            ws = stream_sessions.get(session_id)
            if ws:
                try:
                    combined = build_combined_result(session_id, binary_frame, result)
                    await ws.send_json(combined)
                except Exception as e:
                    logging.error(f"Error sending stream result for session {session_id}: {e}")
            stream_send_queues[worker_index].task_done()
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        await asyncio.sleep(sleep_time)

async def clear_session_queues(session_id: str):
    for q in worker_queues:
        new_deque = collections.deque(item for item in q._queue if item[0] != session_id)
        q._queue = new_deque
    for q in file_send_queues:
        new_deque = collections.deque(item for item in q._queue if item[0] != session_id)
        q._queue = new_deque
    for q in stream_send_queues:
        new_deque = collections.deque(item for item in q._queue if item[0] != session_id)
        q._queue = new_deque

async def worker_task(worker_id: int, queue: asyncio.Queue):
    loop = asyncio.get_running_loop()
    while True:
        session_id, frame_id, binary_frame, object_str, second, mode = await queue.get()
        effective_skip = get_effective_skip(queue.qsize(), FRAME_SKIP)
        if frame_id % effective_skip == 0 and object_str:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: process_frame_wrapper(session_id, frame_id, binary_frame, object_str)
                )
            except Exception as e:
                logging.error(f"Frame {frame_id} processing error for session {session_id}: {e}")
                result = {"bboxes": []}
        else:
            result = {"bboxes": []}
        if DIRECT_SEND_MODE:
            await send_result_direct(session_id, binary_frame, result, mode)
        else:
            if mode == "file":
                send_worker_index = (abs(hash(session_id)) % SEND_WORKER_COUNT)
                await file_send_queues[send_worker_index].put((session_id, binary_frame, result))
            else:
                send_worker_index = (abs(hash(session_id)) % SEND_WORKER_COUNT)
                await stream_send_queues[send_worker_index].put((session_id, binary_frame, result))
        logging.debug(f"Worker {worker_id}: Processed frame {frame_id} for session {session_id}. Queue size: {queue.qsize()}")
        queue.task_done()

@app.on_event("startup")
async def startup_event():
    for worker_id, q in enumerate(worker_queues):
        asyncio.create_task(worker_task(worker_id, q))
    if not DIRECT_SEND_MODE:
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

class StreamControlRequest(BaseModel):
    session_id: str

# --------------------
# WebSocket 接口
# --------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    file_sessions[session_id] = websocket
    logging.info(f"本地视频 WebSocket connected, session_id: {session_id}")
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                continue
    except Exception as e:
        logging.error(f"WebSocket error (file) session {session_id}: {e}")
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
    logging.info(f"互联网流 WebSocket connected, session_id: {session_id}")
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                continue
    except Exception as e:
        logging.error(f"WebSocket error (stream) session {session_id}: {e}")
    finally:
        stream_sessions.pop(session_id, None)

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    logging.info("视频上传，保存中...")
    video_path = UPLOAD_DIR / video.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    return {"status": "success", "message": "视频上传成功", "filename": video.filename}

@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    session_id = request.session_id
    object_str = request.object_str
    filename = request.filename
    logging.info(f"开始分析本地视频，session: {session_id}, 目标对象: {object_str}")
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return {"status": "error", "message": "视频文件不存在"}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": "无法打开视频文件"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = ANALYSIS_FPS
    frame_interval = max(1, fps // ANALYSIS_FPS)
    logging.info(f"原始fps: {fps}, 目标fps: {ANALYSIS_FPS}")

    frame_id, second = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
            _, buffer = cv2.imencode(".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            binary_frame = buffer.tobytes()
            worker_index = (abs(hash(session_id)) % WORKER_COUNT)
            await worker_queues[worker_index].put((session_id, frame_id, binary_frame, object_str, second, "file"))
            logging.info(f"Frame {frame_id} added to worker {worker_index} queue for session {session_id}")
        frame_id += 1
        second = frame_id // fps
    cap.release()
    os.remove(video_path)
    return {"status": "processing", "message": "视频分析中..."}

@app.post("/stream")
async def start_stream(request: StreamRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id
    stream_url = request.stream_url
    if not stream_url:
        return {"status": "error", "message": "缺少流地址"}
    logging.info(f"启动互联网流拉流，session: {session_id}, 流地址: {stream_url}")
    stream_state[session_id] = "running"
    background_tasks.add_task(stream_video_reader, session_id, stream_url)
    return {"status": "success", "message": "互联网流拉流启动", "stream_url": stream_url}

@app.post("/stream/analyze")
async def analyze_stream(request: StreamAnalyzeRequest):
    session_id = request.session_id
    object_str = request.object_str
    if not object_str:
        return {"status": "error", "message": "缺少检测对象"}
    stream_detection[session_id] = object_str
    logging.info(f"设置互联网流检测目标，session: {session_id}, 对象: {object_str}")
    return {"status": "success", "message": "互联网流目标检测已启动", "object": object_str}

@app.post("/stream/pause")
async def pause_stream(request: StreamControlRequest):
    session_id = request.session_id
    if session_id not in stream_state:
        return {"status": "error", "message": "无效的 session_id"}
    stream_state[session_id] = "paused"
    await clear_session_queues(session_id)
    logging.info(f"暂停互联网流，并清空队列数据，session: {session_id}")
    return {"status": "success", "message": "互联网流已暂停，队列数据已清空"}

@app.post("/stream/resume")
async def resume_stream(request: StreamControlRequest):
    session_id = request.session_id
    if session_id not in stream_state:
        return {"status": "error", "message": "无效的 session_id"}
    stream_state[session_id] = "running"
    logging.info(f"恢复互联网流，session: {session_id}")
    return {"status": "success", "message": "互联网流已恢复"}

@app.post("/stream/stop")
async def stop_stream(request: StreamControlRequest):
    session_id = request.session_id
    if session_id not in stream_state:
        return {"status": "error", "message": "无效的 session_id"}
    stream_state[session_id] = "stopped"
    await clear_session_queues(session_id)
    logging.info(f"停止互联网流，并清空队列数据，session: {session_id}")
    return {"status": "success", "message": "互联网流已停止，队列数据已清空"}

async def stream_video_reader(session_id: str, stream_url: str):
    loop = asyncio.get_running_loop()
    try:
        cap = await loop.run_in_executor(None, cv2.VideoCapture, stream_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    except (asyncio.TimeoutError, cv2.error) as e:
        logging.error(f"打开网络流失败: {stream_url}, session: {session_id}, 错误: {e}")
        return

    if not cap.isOpened():
        logging.error(f"无法打开网络流: {stream_url}，session: {session_id}")
        return

    fps = await loop.run_in_executor(None, cap.get, cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = ANALYSIS_FPS
    frame_interval = max(1, int(fps // ANALYSIS_FPS))
    frame_id = 0
    logging.info(f"开始拉流，session: {session_id}，原始fps: {fps}, 目标fps: {ANALYSIS_FPS}")

    while True:
        current_state = stream_state.get(session_id, "running")
        if current_state == "stopped":
            logging.info(f"拉流停止，session: {session_id}")
            break
        ret, frame = cap.read()
        if not ret:
            logging.error(f"读取网络流帧失败，session: {session_id}，等待重试...")
            await asyncio.sleep(0.1)
            continue
        if current_state == "paused":
            frame_id += 1
            await asyncio.sleep(0.01)
            continue
        if frame_id % frame_interval == 0:
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
            _, buffer = cv2.imencode(".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            binary_frame = buffer.tobytes()
            second = frame_id // int(fps)
            object_str = stream_detection.get(session_id, "")
            worker_index = (abs(hash(session_id)) % WORKER_COUNT)
            await worker_queues[worker_index].put((session_id, frame_id, binary_frame, object_str, second, "stream"))
        frame_id += 1
        await asyncio.sleep(0)
    cap.release()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
