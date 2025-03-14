# apt install ffmpeg

from fastapi import FastAPI, WebSocket, UploadFile, File, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
import cv2
import json
import base64
import numpy as np
from openai import OpenAI
from pathlib import Path
import shutil
import os
import time
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# 初始化 Jinja2 模板
templates = Jinja2Templates(directory="templates")

# OpenAI API 初始化
client = OpenAI(
    base_url="http://124.220.202.52:8000/v1",  # 你的 OpenAI API 地址
    api_key="EMPTY"
)

# WebSocket 客户端存储
file_clients = set()
stream_clients = set()

# 参数设置
TARGET_FPS = 20          # 分析帧率
FRAME_SKIP = 50          # 每处理50帧做一次检测（基础值，动态调整）
TARGET_SEND_FPS = 20     # WebSocket 发送帧率控制

last_send_time = time.time()

# 文件视频处理队列（原有方案）
frame_queue = asyncio.Queue()
ordered_result_queue = asyncio.Queue()  # 用于存储处理结果

# 网络视频流处理队列（新增）
stream_frame_queue = asyncio.Queue()
stream_ordered_result_queue = asyncio.Queue()  # 用于存储网络流处理结果

# WebSocket 客户端存储（新增网络流的单独存储）
stream_clients = set()

# 网络流检测目标（由 /stream/analyze 设置）
stream_object_str = ""

# 上传文件保存目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 定义分析请求体模型（文件视频）
class AnalyzeRequest(BaseModel):
    object_str: str
    filename: str

async def analyze_frame(frame_np: bytes, object_str: str):
    """ 使用二进制图片数据进行目标检测 """
    base64_image = base64.b64encode(frame_np).decode('utf-8')
    prompt_str = f"""
    Analyze the image and extract the bounding box coordinates for the object '{object_str}'.
    Provide the response in this fixed format: 
    {{"bbox_2d": [x1, y1, x2, y2], "label": "{object_str}"}}
    If the object is not found, return an empty bbox_2d.
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
        response_text = response.choices[0].message.content
        print("OpenAI response:", response_text)
        response_text = response_text.strip()
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        try:
            data = json.loads(response_text)
            if isinstance(data, list):
                data = data[0]
            return {
                "bbox": data.get("bbox_2d", []),
                "label": data.get("label", object_str)
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {"bbox": [], "label": object_str}
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"bbox": [], "label": object_str}

# 使用线程池，避免频繁创建进程
thread_executor = ThreadPoolExecutor(max_workers=4)

def process_frame_wrapper(frame_np, object_str):
    """
    使用线程池异步执行 analyze_frame，替代原来的多进程调用。
    """
    future = thread_executor.submit(lambda: asyncio.run(analyze_frame(frame_np, object_str)))
    return future.result()

# 根据当前队列负载动态计算有效跳帧因子
def get_effective_skip(queue_size, base_skip=FRAME_SKIP):
    if queue_size > 50:
        return base_skip * 2
    elif queue_size > 20:
        return int(base_skip * 1.5)
    else:
        return base_skip

# 原有文件视频帧处理任务，修改为动态跳帧和线程池调用
async def frame_worker():
    loop = asyncio.get_running_loop()
    while True:
        frame_info = await frame_queue.get()
        binary_frame, object_str, second, frame_id = frame_info

        # 根据队列当前负载动态调整跳帧因子
        effective_skip = get_effective_skip(frame_queue.qsize(), FRAME_SKIP)
        if frame_id % effective_skip == 0:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: process_frame_wrapper(binary_frame, object_str)
                )
            except Exception as e:
                print(f"Frame {frame_id} processing error: {e}")
                result = {"bbox": [], "label": object_str}
        else:
            result = {"bbox": [], "label": object_str}

        await ordered_result_queue.put((frame_id, binary_frame, result))
        print(f"Frame {frame_id} processed. Queue size: {ordered_result_queue.qsize()}")
        frame_queue.task_done()

async def send_results():
    interval = 1 / TARGET_SEND_FPS
    while True:
        start_time = time.time()
        if not ordered_result_queue.empty():
            frame_id, binary_frame, result = await ordered_result_queue.get()
            for client in file_clients:
                await client.send_bytes(binary_frame)
                await client.send_json(result)
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        await asyncio.sleep(sleep_time)

# 新增网络流视频帧处理任务，修改为动态跳帧和线程池调用
async def stream_frame_worker():
    loop = asyncio.get_running_loop()
    while True:
        frame_info = await stream_frame_queue.get()
        binary_frame, object_str, second, frame_id = frame_info

        effective_skip = get_effective_skip(stream_frame_queue.qsize(), FRAME_SKIP)
        if frame_id % effective_skip == 0 and object_str:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: process_frame_wrapper(binary_frame, object_str)
                )
            except Exception as e:
                print(f"Stream Frame {frame_id} processing error: {e}")
                result = {"bbox": [], "label": object_str}
        else:
            result = {"bbox": [], "label": object_str}

        await stream_ordered_result_queue.put((frame_id, binary_frame, result))
        stream_frame_queue.task_done()

async def send_stream_results():
    interval = 1 / TARGET_SEND_FPS
    while True:
        start_time = time.time()
        if not stream_ordered_result_queue.empty():
            frame_id, binary_frame, result = await stream_ordered_result_queue.get()
            for client in stream_clients:
                await client.send_bytes(binary_frame)
                await client.send_json(result)
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        await asyncio.sleep(sleep_time)

# 网络视频流读取任务（支持RTSP）
async def stream_video_reader(stream_url: str):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("无法打开网络流:", stream_url)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = TARGET_FPS
    frame_interval = max(1, int(fps // TARGET_FPS))
    frame_id = 0
    print(f"开始拉流，原始fps: {fps}, 目标fps: {TARGET_FPS}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取网络流帧失败，等待重试...")
            await asyncio.sleep(0.1)
            continue
        if frame_id % frame_interval == 0:
            # 缩放与压缩
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
            _, buffer = cv2.imencode(".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            binary_frame = buffer.tobytes()
            second = frame_id // int(fps)
            # 使用当前设置的检测对象（由 /stream/analyze 设置）
            await stream_frame_queue.put((binary_frame, stream_object_str, second, frame_id))
        frame_id += 1
        await asyncio.sleep(0)  # 让出控制权
    cap.release()

# WebSocket 端点：本地视频（原有）
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    file_clients.add(websocket)
    print("本地视频 WebSocket connected")
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket error: {e}")
        file_clients.remove(websocket)

# 新增 WebSocket 端点：网络流
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    stream_clients.add(websocket)
    print("网络流 WebSocket connected")
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"网络流 WebSocket error: {e}")
        stream_clients.remove(websocket)

# 上传文件（本地视频）接口（原有）
@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    print("视频上传，保存中...")
    video_path = UPLOAD_DIR / video.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    return {"status": "success", "message": "视频上传成功", "filename": video.filename}

# 本地视频分析接口（原有）
@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    object_str = request.object_str
    filename = request.filename
    print(f"开始分析本地视频，目标对象: {object_str}")
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return {"status": "error", "message": "视频文件不存在"}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": "无法打开视频文件"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = TARGET_FPS
    frame_interval = max(1, fps // target_fps)
    print(f"原始fps: {fps}, 目标fps: {target_fps}")

    frame_id, second = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
            _, buffer = cv2.imencode(".jpg", resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            binary_frame = buffer.tobytes()
            await frame_queue.put((binary_frame, object_str, second, frame_id))
            print(f"Frame {frame_id} added to frame_queue")
        frame_id += 1
        second = frame_id // fps
    cap.release()
    os.remove(video_path)
    return {"status": "processing", "message": "视频分析中..."}

# 新增接口：启动网络流拉流，支持RTSP等流地址
class StreamRequest(BaseModel):
    stream_url: str

@app.post("/stream")
async def start_stream(request: StreamRequest, background_tasks: BackgroundTasks):
    stream_url = request.stream_url
    if not stream_url:
        return {"status": "error", "message": "缺少流地址"}
    print(f"启动网络流拉流，流地址: {stream_url}")
    # 启动后台任务拉取流
    background_tasks.add_task(stream_video_reader, stream_url)
    return {"status": "success", "message": "网络流拉流启动", "stream_url": stream_url}

# 新增接口：设置网络流检测目标
class StreamAnalyzeRequest(BaseModel):
    object_str: str

@app.post("/stream/analyze")
async def analyze_stream(request: StreamAnalyzeRequest):
    global stream_object_str
    object_str = request.object_str
    if not object_str:
        return {"status": "error", "message": "缺少检测对象"}
    stream_object_str = object_str
    print(f"设置网络流检测目标: {object_str}")
    return {"status": "success", "message": "网络流目标检测已启动", "object": object_str}

# 首页
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(frame_worker())
    asyncio.create_task(send_results())
    asyncio.create_task(stream_frame_worker())
    asyncio.create_task(send_stream_results())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
