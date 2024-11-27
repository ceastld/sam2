import gradio as gr
import cv2
import numpy as np
SOURCE_VIDEO_PATH = "exp_data/HSID48/face_crop.mp4"


# 假设 sam2 是你要使用的处理视频的函数
def sam2(video_path):
    # 这里是一个示例，你需要替换成实际的 sam2 处理逻辑
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 假设 sam2 处理的是视频帧，进行一些图像处理操作
        # 这里只是一个简单的例子
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    # 假设我们返回的是处理过的所有帧
    return frames

# 定义 Gradio 接口
def process_video(video):
    result_frames = sam2(video.name)
    return result_frames

# 创建 Gradio 界面
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),  # 允许用户上传视频文件，移除 type="file"
    outputs=gr.Gallery(label="Processed Video Frames"),  # 显示处理后的帧
    title="Video Processing with SAM2",
    description="Upload a video to process with SAM2."
)

# 启动界面
iface.launch()
