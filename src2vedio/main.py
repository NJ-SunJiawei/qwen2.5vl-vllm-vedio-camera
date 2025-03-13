from fastapi import FastAPI, WebSocket, UploadFile, File, Request
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
from multiprocessing import Process, Queue

app = FastAPI()

# 初始化 Jinja2 模板
templates = Jinja2Templates(directory="templates")

# OpenAI API 初始化
client = OpenAI(
    base_url="http://43.136.90.245:8000/v1",  # 你的 OpenAI API 地址
    api_key="EMPTY"
)

# WebSocket 客户端存储
clients = set()

# 参数设置
TARGET_FPS = 20          # 分析帧率
FRAME_SKIP = 50          # 每处理5帧做一次检测
TARGET_SEND_FPS = 20     # WebSocket 发送帧率控制

last_send_time = time.time()
frame_queue = asyncio.Queue() #asyncio.Queue(maxsize=QUEUE_SIZE)
ordered_result_queue = asyncio.Queue()  # 用于存储处理结果

# 上传文件保存目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 定义分析请求体模型
class AnalyzeRequest(BaseModel):
    object_str: str
    filename: str

async def analyze_frame(frame_np: np.ndarray, object_str: str):
    """ 使用 numpy 数组进行目标检测，无需写入 .jpg 文件 """
    _, buffer = cv2.imencode(".jpg", frame_np, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    base64_image = base64.b64encode(buffer).decode('utf-8')

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

def process_frame(frame_np, object_str, result_queue):
    """ 进程中处理帧 """
    result = asyncio.run(analyze_frame(frame_np, object_str))
    result_queue.put(result)

def process_frame_wrapper(frame_np, object_str):
    """ 同步包装进程调用 """
    result_queue = Queue()
    process = Process(target=process_frame, args=(frame_np, object_str, result_queue))
    process.start()
    process.join()
    return result_queue.get()

async def frame_worker():
    """ 处理队列中的帧 """
    loop = asyncio.get_running_loop()
    while True:
        frame_info = await frame_queue.get()
        frame_np, object_str, second, frame_id = frame_info

        if frame_id % FRAME_SKIP == 0:
            try:
                result = await loop.run_in_executor(
                    None, 
                    lambda: process_frame_wrapper(frame_np, object_str)
                )
            except Exception as e:
                print(f"Frame {frame_id} processing error: {e}")
                result = {"bbox": [], "label": object_str}
        else:
            result = {"bbox": [], "label": object_str}

        _, buffer = cv2.imencode(".jpg", frame_np, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        binary_frame = buffer.tobytes()
        await ordered_result_queue.put((frame_id, binary_frame, result))
        print(f"Frame {frame_id} processed. Queue size: {ordered_result_queue.qsize()}")
        frame_queue.task_done()

'''
async def send_results():
    """发送结果（新增帧率控制）"""
    global last_send_time
    while True:
        if not ordered_result_queue.empty():
            now = time.time()
            if now - last_send_time >= 1/TARGET_SEND_FPS:
                frame_id, binary_frame, result = await ordered_result_queue.get()
                for client in clients:
                    await client.send_bytes(binary_frame)
                    await client.send_json(result)
                last_send_time = now
        await asyncio.sleep(0.001)
'''

async def send_results():
    interval = 1 / TARGET_SEND_FPS
    while True:
        start_time = time.time()
        if not ordered_result_queue.empty():
            frame_id, binary_frame, result = await ordered_result_queue.get()
            for client in clients:
                await client.send_bytes(binary_frame)
                await client.send_json(result)
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        await asyncio.sleep(sleep_time)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(frame_worker())
    asyncio.create_task(send_results())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("WebSocket connected")
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket error: {e}")
        clients.remove(websocket)

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    print("Video uploaded, saving...")
    video_path = UPLOAD_DIR / video.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    return {"status": "success", "message": "视频上传成功", "filename": video.filename}

@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    object_str = request.object_str
    filename = request.filename
    print(f"Starting analysis for object: {object_str}")
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return {"status": "error", "message": "视频文件不存在"}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": "无法打开视频文件"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = TARGET_FPS #如果原视频 FPS 是 30，而目标是 10，则 frame_interval = 30 // 10 = 3，意味着每 3 帧采样一帧，从而达到降帧处理的目的
    frame_interval = max(1, fps // target_fps)
    print(f"ori_fps: {fps}")
    print(f"target_fps: {target_fps}")

    frame_id, second = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            await frame_queue.put((frame, object_str, second, frame_id))
            print(f"Frame {frame_id} added to frame_queue")
        frame_id += 1
        second = frame_id // fps
    cap.release()
    os.remove(video_path)
    return {"status": "processing", "message": "视频分析中..."}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
