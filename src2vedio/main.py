from fastapi import FastAPI, WebSocket, UploadFile, File, Request, Body
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

# 帧处理参数
FRAME_SKIP = 40  # 每 10 帧抽检一次（根据实际情况调整）
QUEUE_SIZE = 1000  # 最大缓存 100 帧
frame_queue = asyncio.Queue(maxsize=QUEUE_SIZE)

# 上传文件保存目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # 确保上传目录存在

# 定义分析请求体模型
class AnalyzeRequest(BaseModel):
    object_str: str
    filename: str

async def analyze_frame(frame_np: np.ndarray, object_str: str):
    """ 直接使用 `numpy` 数组进行目标检测，无需写入 `.jpg` """
    _, buffer = cv2.imencode(".jpg", frame_np, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 压缩质量为 50
    base64_image = base64.b64encode(buffer).decode('utf-8')

    prompt_str = f"""
    Analyze the image and extract the bounding box coordinates for the object '{object_str}'.
    Provide the response in this fixed format: 
    {{"bbox_2d": [x1, y1, x2, y2], "label": "{object_str}"}}
    If the object is not found, return an empty bbox_2d.
    """

    print("Sending request to OpenAI...")  # 调试日志
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt_str},
                                                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            max_tokens=1024,
        )

        response_text = response.choices[0].message.content
        print("OpenAI response:", response_text)  # 调试日志

        # 预处理响应文本
        response_text = response_text.strip()  # 去除首尾空白字符
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()  # 去除 ```json 和 ```

        # 尝试解析 JSON
        try:
            data = json.loads(response_text)
            if isinstance(data, list):  # 如果返回的是数组，取第一个元素
                data = data[0]
            return {
                "bbox": data.get("bbox_2d", []),  # 返回 bbox_2d
                "label": data.get("label", object_str)  # 返回 label
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {"bbox": [], "label": object_str}  # 返回空 bbox 和默认 label
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"bbox": [], "label": object_str}  # 返回空 bbox 和默认 label

async def frame_worker():
    """ 处理队列中的帧 """
    while True:
        frame_info = await frame_queue.get()
        frame_np, object_str, second, frame_id = frame_info

        # 将帧转换为 base64
        _, buffer = cv2.imencode(".jpg", frame_np, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 压缩质量为 50
        base64_frame = base64.b64encode(buffer).decode('utf-8')

        # 根据 FRAME_SKIP 抽检帧
        if frame_id % FRAME_SKIP == 0:
            result = await analyze_frame(frame_np, object_str)
        else:
            result = {"bbox": [], "label": object_str}  # 未抽检的帧不进行检测

        # 发送 WebSocket 更新
        message = json.dumps({
            "frame": base64_frame,  # 发送 base64 帧
            "second": second,
            "frame_id": frame_id,
            "bbox": result["bbox"],  # 发送 bbox
            "label": result["label"]  # 发送 label
        })
        await asyncio.gather(*[client.send_text(message) for client in clients])
        frame_queue.task_done()

@app.on_event("startup")
async def startup_event():
    """ 确保 `frame_worker` 在 FastAPI 生命周期内运行 """
    asyncio.create_task(frame_worker())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """ WebSocket 处理函数 """
    await websocket.accept()
    clients.add(websocket)
    print("WebSocket connected")  # 调试日志
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket error: {e}")  # 调试日志
        clients.remove(websocket)

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """ 处理视频上传 """
    print("Video uploaded, saving...")  # 调试日志

    # 保存上传的视频文件
    video_path = UPLOAD_DIR / video.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    return {"status": "success", "message": "视频上传成功", "filename": video.filename}

@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    """ 处理视频分析 """
    object_str = request.object_str
    filename = request.filename
    print(f"Starting analysis for object: {object_str}")  # 调试日志

    # 使用 OpenCV 读取视频文件
    video_path = UPLOAD_DIR / filename  # 使用上传的文件名
    if not video_path.exists():
        return {"status": "error", "message": "视频文件不存在"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": "无法打开视频文件"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 10  # 目标帧率
    frame_interval = max(1, fps // target_fps)  # 计算帧间隔

    frame_id, second = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 根据帧间隔跳过部分帧
        if frame_id % frame_interval == 0:
            await frame_queue.put((frame, object_str, second, frame_id))

        frame_id += 1
        second = frame_id // fps

    cap.release()
    os.remove(video_path)  # 删除上传的视频文件
    return {"status": "processing", "message": "视频分析中..."}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """ 返回渲染后的 HTML 页面 """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    """ 忽略 favicon.ico 请求 """
    return {"message": "No favicon"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)